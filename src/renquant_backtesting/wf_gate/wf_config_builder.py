#!/usr/bin/env python
"""Build production-semantic walk-forward eval configs.

Walk-forward configs need a few evaluation-only paths (manifest, static
placeholder artifact, calibration artifact), but their decision semantics must
match production exactly. Hand-edited side configs drifted on buy floors,
tax-lot method, sector maps, and regime params; this builder makes the allowed
differences explicit.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

try:
    from .wf_config_parity import evaluate_wf_config_parity
except ImportError:
    from wf_config_parity import evaluate_wf_config_parity

try:
    from .artifact_loader import load_artifact_payload
except ImportError:
    from artifact_loader import load_artifact_payload


def _resolve_repo_root() -> Path:
    env_root = os.environ.get("RENQUANT_REPO_ROOT")
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend([Path.cwd(), *Path(__file__).resolve().parents])
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "backtesting" / "renquant_104").is_dir():
            return root
    return Path(env_root).expanduser().resolve() if env_root else Path.cwd().resolve()


REPO = _resolve_repo_root()
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"

EXPERIMENT_OVERRIDE_PATHS = (
    "rotation.joint_actions.qp_admission_gate.max_sigma",
    "rotation.joint_actions.qp_admission_gate.max_sigma_by_regime",
    "rotation.joint_actions.qp_admission_gate.topup_max_sigma",
    "rotation.joint_actions.qp_admission_gate.topup_max_sigma_by_regime",
    "rotation.joint_actions.qp_admission_gate.min_expected_return",
    "rotation.joint_actions.qp_admission_gate.min_expected_return_by_regime",
    "rotation.joint_actions.qp_admission_gate.min_expected_excess_return",
    "rotation.joint_actions.qp_admission_gate.min_expected_excess_return_by_regime",
    "rotation.joint_actions.qp_admission_gate.topup_min_expected_return",
    "rotation.joint_actions.qp_admission_gate.topup_min_expected_return_by_regime",
    "rotation.joint_actions.qp_admission_gate.topup_min_expected_excess_return",
    "rotation.joint_actions.qp_admission_gate.topup_min_expected_excess_return_by_regime",
)


def _get_path(obj: dict[str, Any], dotted: str) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _set_path(obj: dict[str, Any], dotted: str, value: Any) -> None:
    cur: Any = obj
    parts = dotted.split(".")
    for part in parts[:-1]:
        nxt = cur.setdefault(part, {})
        if not isinstance(nxt, dict):
            raise TypeError(f"cannot set {dotted}: {part} is not a mapping")
        cur = nxt
    cur[parts[-1]] = value


# Scorer-kind vocabulary, kept aligned with the parity guard in
# ``wf_config_parity.py`` (``PATCHTST_KINDS`` and ``_normalize_kind``). Used by
# ``select_prod_reference_for_candidate`` to pick the production config whose
# scorer kind matches the candidate, NOT to mutate any config's declared kind.
_PATCHTST_KIND = "hf_patchtst"
_GBDT_KIND = "xgb"
_PATCHTST_KINDS = {"hf_patchtst", "patchtst", "patchtst_panel"}

_PROD_CONFIG_NAMES = ("strategy_config.json", "strategy_config.shadow.json")


def _resolve_prod_reference_by_kind(
    strategy_dir: Path = STRATEGY_DIR,
) -> dict[str, str]:
    """Scan production configs and map scorer kind → config filename.

    Survives primary/shadow lineup swaps without code changes: the mapping is
    derived from each config's declared ``ranking.panel_scoring.kind``, not
    from a hardcoded filename assumption.
    """
    mapping: dict[str, str] = {}
    for name in _PROD_CONFIG_NAMES:
        path = strategy_dir / name
        if not path.exists():
            continue
        try:
            cfg = json.loads(path.read_text())
            kind = _normalize_kind(
                ((cfg.get("ranking") or {}).get("panel_scoring") or {}).get("kind")
            )
            if kind and kind not in mapping:
                mapping[kind] = name
        except Exception:
            continue
    return mapping


def _normalize_kind(kind: Any) -> str:
    """Collapse a scorer kind to the parity-guard vocabulary.

    Mirrors ``wf_config_parity._normalize_kind`` so ``panel_ltr_xgboost`` /
    ``xgboost`` collapse to ``xgb`` and PatchTST aliases collapse to
    ``hf_patchtst``. Returns ``""`` for an empty value.
    """
    value = str(kind or "").lower()
    if value in _PATCHTST_KINDS:
        return _PATCHTST_KIND
    aliases = {
        "panel_ltr_xgboost": _GBDT_KIND,
        "xgboost": _GBDT_KIND,
    }
    return aliases.get(value, value)


def select_prod_reference_for_candidate(
    candidate_kind: Any,
    *,
    strategy_dir: Path = STRATEGY_DIR,
    env_override: str | None = None,
) -> Path:
    """Select the production reference config matched to the candidate kind.

    The ONE parity contract shared with the umbrella rollback path
    (``scripts/run_wf_gate.py::_prod_config_path``): a candidate is only
    promotable against the production semantics that actually run its scorer
    kind. A GBDT/xgb candidate is compared against the GBDT/shadow config
    (``strategy_config.shadow.json``, ``kind=xgb``); a PatchTST candidate
    against the PatchTST primary (``strategy_config.json``, ``kind=hf_patchtst``).

    ``candidate_kind`` MUST come from the candidate artifact's declared metadata
    (``_load_artifact_payload(...)["kind"]``), never from a path suffix.

    Selection precedence:
      1. ``env_override`` (``RENQUANT_STRATEGY_CONFIG``) when set — but its
         declared ``panel_scoring.kind`` is validated to match the candidate
         kind; a mismatch FAILS CLOSED so the env cannot smuggle a wrong
         reference past parity.
      2. ``_resolve_prod_reference_by_kind`` scans both configs by declared kind.

    Raises ``ValueError`` (fail closed) when the candidate kind is unknown/empty
    or no matched production reference exists. The caller must treat that as a
    non-promotable run, not silently pass.
    """
    kind = _normalize_kind(candidate_kind)
    if not kind:
        raise ValueError(
            "cannot select a production reference: candidate artifact has no "
            "declared scorer kind (load it from artifact metadata, not a path "
            "suffix); fail closed."
        )

    if env_override:
        path = Path(env_override).expanduser()
        if not path.is_absolute():
            path = strategy_dir / path
        path = path.resolve()
        if not path.exists():
            raise ValueError(
                f"RENQUANT_STRATEGY_CONFIG points at a missing prod config: {path}"
            )
        ref_kind = _normalize_kind(
            ((json.loads(path.read_text()).get("ranking") or {})
             .get("panel_scoring") or {}).get("kind")
        )
        if ref_kind != kind:
            raise ValueError(
                "RENQUANT_STRATEGY_CONFIG selects a production reference whose "
                f"scorer kind ({ref_kind!r}) does not match the candidate kind "
                f"({kind!r}); refusing to compare a candidate against a "
                "non-matching production reference. Fail closed."
            )
        return path

    by_kind = _resolve_prod_reference_by_kind(strategy_dir)
    ref_name = by_kind.get(kind)
    if not ref_name:
        raise ValueError(
            f"no production reference config is registered for candidate kind "
            f"{kind!r}; scanned configs declared kinds: {by_kind}. "
            "Fail closed — a candidate cannot be promoted without a "
            "kind-matched production reference."
        )
    ref_path = (strategy_dir / ref_name).resolve()
    if not ref_path.exists():
        raise ValueError(
            f"production reference for kind {kind!r} not found: {ref_path}"
        )
    return ref_path


def _first_manifest_artifact(manifest_path: Path) -> str | None:
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text())
    rows = payload.get("retrains", []) if isinstance(payload, dict) else payload
    if not rows:
        return None
    raw = str((rows[0] or {}).get("artifact_uri") or "")
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        try:
            return str(path.relative_to(STRATEGY_DIR))
        except ValueError:
            return str(path)
    return raw


def _resolve_strategy_path(raw: str | None, strategy_dir: Path) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else strategy_dir / path


def _semantic_overrides_in_base(
    prod_config: dict[str, Any],
    base_wf_config: dict[str, Any],
) -> list[str]:
    """Return explicit experiment overrides that differ from production.

    ``build_wf_config_from_prod`` intentionally starts from production
    semantics. If an operator hands it a side config with an explicit
    experiment knob, silently dropping that knob creates a false A/B result.
    Require a separate opt-in before preserving those semantic differences.
    """
    out: list[str] = []
    for dotted in EXPERIMENT_OVERRIDE_PATHS:
        base_value = _get_path(base_wf_config, dotted)
        if base_value is None:
            continue
        prod_value = _get_path(prod_config, dotted)
        if base_value != prod_value:
            out.append(dotted)
    return out


def build_wf_config_from_prod(
    prod_config: dict[str, Any],
    *,
    manifest_path: str,
    base_wf_config: dict[str, Any] | None = None,
    strategy_dir: Path = STRATEGY_DIR,
    preserve_experiment_overrides: bool = False,
) -> dict[str, Any]:
    """Return a WF eval config with production decision semantics.

    The derived config inherits ``ranking.panel_scoring.kind`` UNCHANGED from
    ``prod_config``. The builder never mutates the scorer kind to match a
    candidate artifact — doing so would convert a genuine prod-vs-candidate
    mismatch into a passing config and defeat the parity guard. To evaluate a
    non-PatchTST candidate, the caller must pass the kind-matched production
    reference (see ``select_prod_reference_for_candidate``): a GBDT/xgb
    candidate is derived from the GBDT/shadow config, so the inherited kind is
    already ``xgb`` and parity passes; a GBDT candidate handed the PatchTST
    primary STILL fails parity, as it should.

    Allowed non-semantic differences:
      - ``walkforward`` manifest dispatch.
      - ``ranking.panel_scoring.artifact_path`` placeholder for diagnostics;
        SimAdapter uses the manifest when walkforward is enabled.
      - ``ranking.panel_scoring.global_calibration.*`` artifact paths. These
        are evaluation artifacts and must remain point-in-time / sim-scoped,
        not production full-sample paths.
      - ``regime.gmm_artifact`` and ``regime.correlation_artifact`` because
        historical simulations must use point-in-time regime/risk artifacts
        whose ``as_of_date`` is no later than the backtest start.
      - ``ranking.panel_scoring.regime_admission.enabled`` is disabled only
        for WF acceptance runs because that gate consumes the
        ``wf_gate_metadata`` which this run is generating. Live/preflight still
        require the stamped evidence before buy/QP admission.
      - Shadow-model tracking is disabled because it has no trade-decision
        effect and can make WF gates spend hours in auxiliary PyTorch inference.
      - Label/metadata fields starting with ``_`` or ``__``.
    """
    cfg = copy.deepcopy(prod_config)
    base = base_wf_config or {}
    overrides = _semantic_overrides_in_base(prod_config, base)
    if overrides and not preserve_experiment_overrides:
        joined = ", ".join(overrides)
        raise ValueError(
            "base WF config contains semantic experiment override(s) that "
            "would be dropped by production-semantic derivation: "
            f"{joined}. Re-run with preserve_experiment_overrides=True "
            "or the CLI flag --preserve-experiment-overrides for an "
            "explicit diagnostic/non-promotable A/B run."
        )

    wf_base = copy.deepcopy(base.get("walkforward") or {})
    wf_base.update({
        "enabled": True,
        "manifest_path": manifest_path,
        "fail_on_no_model": bool(wf_base.get("fail_on_no_model", True)),
    })
    cfg["walkforward"] = wf_base

    manifest_abs = _resolve_strategy_path(manifest_path, strategy_dir)
    placeholder = _first_manifest_artifact(manifest_abs) if manifest_abs else None
    if placeholder:
        # Only the diagnostic artifact_path placeholder is overwritten;
        # ``ranking.panel_scoring.kind`` is inherited unchanged from prod_config
        # (see docstring). Parity stays meaningful: a candidate is promotable
        # only when derived from its kind-matched production reference.
        _set_path(cfg, "ranking.panel_scoring.artifact_path", placeholder)

    # Preserve point-in-time calibration artifact paths from the WF base config.
    # Production calibrators are usually fitted on the full current panel, so
    # copying them into historical WF would create a subtle look-ahead channel.
    for dotted in (
        "ranking.panel_scoring.global_calibration.artifact_path",
        "ranking.panel_scoring.global_calibration.regime_conditional.artifact_pattern",
        "regime.gmm_artifact",
        "regime.correlation_artifact",
    ):
        value = _get_path(base, dotted)
        if value is not None:
            _set_path(cfg, dotted, value)

    if preserve_experiment_overrides:
        preserved: list[str] = []
        for dotted in overrides:
            value = _get_path(base, dotted)
            if value is not None:
                _set_path(cfg, dotted, copy.deepcopy(value))
                preserved.append(dotted)
        if preserved:
            cfg["_experiment_overrides_preserved"] = preserved
            cfg["_experiment_overrides_note"] = (
                "Diagnostic WF A/B only. These semantic differences mean the "
                "generated config is not production-equivalent and cannot be "
                "promotion evidence until production config carries the same "
                "knobs and parity passes."
            )

    panel_cfg = cfg.setdefault("ranking", {}).setdefault("panel_scoring", {})
    regime_admission = panel_cfg.setdefault("regime_admission", {})
    regime_admission["enabled"] = False
    regime_admission["_wf_disabled_reason"] = (
        "WF acceptance generates trade_monotonicity/sanity_regime_ic metadata; "
        "requiring pre-existing wf_gate_metadata here creates a circular "
        "zero-trade evaluation. Live/preflight keep the production fail-closed "
        "admission contract."
    )
    if "shadow_models" in panel_cfg:
        panel_cfg["shadow_models"] = []
        panel_cfg["_wf_shadow_disabled_reason"] = (
            "WF acceptance validates production trade decisions; shadow models "
            "are non-decision diagnostics and are disabled to avoid auxiliary "
            "PyTorch inference stalls."
        )

    for key in ("__label", "_side_config_label", "_generated_note", "_backtest_start_note"):
        if key in base:
            cfg[key] = base[key]
    cfg["_generated_note"] = (
        "Generated by scripts/wf_config_builder.py from production config; "
        "only walkforward/artifact/calibration/regime paths may differ."
    )
    cfg["_generated_at_utc"] = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prod-config", default=None,
        help="Explicit production reference override. If omitted and "
             "--candidate-artifact is given, the kind-matched reference is "
             "selected automatically via select_prod_reference_for_candidate "
             "(the same contract runner.main() and pipelines.py use) — this "
             "is what lets the builder survive a primary/shadow lineup swap. "
             "If omitted with no candidate, defaults to strategy_config.json.")
    parser.add_argument("--base-wf-config", required=True,
                        help="Existing WF config to borrow manifest/calibrator paths from.")
    parser.add_argument("--out", required=True,
                        help="Output config path. Relative paths resolve under strategy dir.")
    parser.add_argument("--candidate-artifact", default=None,
                        help="Optional candidate artifact for feature-contract parity.")
    parser.add_argument("--preserve-experiment-overrides", action="store_true",
                        help="Explicitly carry whitelisted diagnostic semantic "
                             "overrides from the base WF config. This is for "
                             "A/B exploration and will normally fail prod/WF "
                             "parity until production config matches.")
    args = parser.parse_args(argv)

    candidate = Path(args.candidate_artifact) if args.candidate_artifact else None
    candidate_kind = load_artifact_payload(candidate).get("kind") if candidate else None

    if args.prod_config is not None:
        prod_path = Path(args.prod_config)
        if not prod_path.is_absolute():
            prod_path = STRATEGY_DIR / prod_path
        if candidate_kind:
            # Explicit override still must not smuggle a mismatched reference
            # past parity — same fail-closed validation
            # select_prod_reference_for_candidate applies to its own
            # RENQUANT_STRATEGY_CONFIG env_override.
            ref_kind = _normalize_kind(
                ((json.loads(prod_path.read_text()).get("ranking") or {})
                 .get("panel_scoring") or {}).get("kind")
            )
            if ref_kind != _normalize_kind(candidate_kind):
                raise SystemExit(
                    "--prod-config explicitly selects a reference whose "
                    f"scorer kind ({ref_kind!r}) does not match the candidate "
                    f"kind ({candidate_kind!r}); refusing to compare a "
                    "candidate against a non-matching production reference. "
                    "Fail closed."
                )
    elif candidate_kind:
        prod_path = select_prod_reference_for_candidate(
            candidate_kind,
            strategy_dir=STRATEGY_DIR,
            env_override=os.environ.get("RENQUANT_STRATEGY_CONFIG"),
        )
    else:
        prod_path = STRATEGY_DIR / "strategy_config.json"

    base_path = Path(args.base_wf_config)
    if not base_path.is_absolute():
        base_path = STRATEGY_DIR / base_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = STRATEGY_DIR / out_path

    prod = json.loads(prod_path.read_text())
    base = json.loads(base_path.read_text())
    manifest = ((base.get("walkforward") or {}).get("manifest_path"))
    if not manifest:
        raise SystemExit(f"base WF config has no walkforward.manifest_path: {base_path}")

    cfg = build_wf_config_from_prod(
        prod,
        manifest_path=str(manifest),
        base_wf_config=base,
        strategy_dir=STRATEGY_DIR,
        preserve_experiment_overrides=args.preserve_experiment_overrides,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cfg, indent=2, sort_keys=False) + "\n")

    result = evaluate_wf_config_parity(
        prod_path,
        out_path,
        candidate_artifact=candidate,
        strategy_dir=STRATEGY_DIR,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
