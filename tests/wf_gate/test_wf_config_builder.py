"""Tests for the converged WF-gate config-parity contract (active path).

The ONE contract (shared with umbrella ``scripts/run_wf_gate.py``): select the
production reference whose scorer kind MATCHES the candidate's declared kind
(read from artifact metadata, never a path suffix), and use that SAME selected
reference for BOTH derivation and the parity check. The builder must NOT mutate
``ranking.panel_scoring.kind`` to force parity — that would convert a genuine
prod-vs-candidate mismatch into a passing config.

Coverage:
  * the builder inherits ``kind`` unchanged from whatever prod config it is
    handed (no suffix inference, no mutation);
  * ``select_prod_reference_for_candidate`` picks the kind-matched reference,
    honors a validated env override, and FAILS CLOSED on unknown / mismatched
    kinds;
  * NEGATIVE: a GBDT candidate with NO GBDT reference selected (PatchTST prod)
    stays non-promotable — parity fails;
  * POSITIVE: a GBDT candidate WITH the GBDT/shadow reference selected passes,
    using the same selected reference for derivation and parity.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from renquant_backtesting.wf_gate import wf_config_builder
from renquant_backtesting.wf_gate.wf_config_builder import (
    build_wf_config_from_prod,
    derive_xgb_reference_from_blend,
    main as wf_config_builder_main,
    select_prod_reference_for_candidate,
)
from renquant_backtesting.wf_gate.wf_config_parity import evaluate_wf_config_parity


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _prod_config(*, kind: str, artifact_path: str) -> dict:
    return {
        "ranking": {
            "panel_scoring": {
                "enabled": True,
                "kind": kind,
                "artifact_path": artifact_path,
                "buy_floor": "adaptive_mean_std",
            },
        },
    }


def _write_manifest(path: Path, artifact_uri: str) -> None:
    _write_json(path, {"retrains": [{"artifact_uri": artifact_uri}]})


#: The pinned production xgb leg's config fingerprint (matches the real
#: ``panel-ltr.alpha158_fund.json`` artifact + the blend component pin).
_XGB_PIN = "sha256:f8fb2259b2bf1537"


def _gbdt_artifact(
    path: Path, cols: list[str], *, config_fingerprint: str | None = _XGB_PIN
) -> Path:
    payload = {"kind": "panel_ltr_xgboost", "feature_cols": cols}
    if config_fingerprint is not None:
        payload["config_fingerprint"] = config_fingerprint
    _write_json(path, payload)
    return path


# ── builder no longer mutates kind ───────────────────────────────────────────


def test_builder_inherits_kind_unchanged_from_prod(tmp_path: Path) -> None:
    """The derived config carries the prod config's kind verbatim — no mutation.

    Even when the manifest artifact is a GBDT ``panel-ltr.json``, deriving from a
    PatchTST prod config keeps ``kind=hf_patchtst``. The builder must never infer
    a kind from the artifact suffix; selecting the correct reference is the
    caller's job.
    """
    gbdt_artifact = _gbdt_artifact(tmp_path / "artifacts" / "sim" / "panel-ltr.json", ["a", "b"])
    manifest = tmp_path / "artifacts" / "wf" / "manifest.json"
    _write_manifest(manifest, str(gbdt_artifact.relative_to(tmp_path)))

    prod_cfg = _prod_config(kind="hf_patchtst", artifact_path="artifacts/prod/model.pt")
    base_cfg = {"walkforward": {"manifest_path": str(manifest.relative_to(tmp_path))}}

    derived = build_wf_config_from_prod(
        prod_cfg,
        manifest_path=str(manifest.relative_to(tmp_path)),
        base_wf_config=base_cfg,
        strategy_dir=tmp_path,
    )

    panel = derived["ranking"]["panel_scoring"]
    assert panel["kind"] == "hf_patchtst"  # inherited, NOT mutated to xgb
    assert panel["artifact_path"].endswith("panel-ltr.json")


def test_builder_inherits_xgb_kind_from_gbdt_prod(tmp_path: Path) -> None:
    """Derived config inherits ``xgb`` when the prod config IS the GBDT reference."""
    gbdt_artifact = _gbdt_artifact(tmp_path / "artifacts" / "sim" / "panel-ltr.json", ["a", "b"])
    manifest = tmp_path / "artifacts" / "wf" / "manifest.json"
    _write_manifest(manifest, str(gbdt_artifact.relative_to(tmp_path)))

    prod_cfg = _prod_config(kind="xgb", artifact_path="artifacts/prod/panel-ltr.json")
    base_cfg = {"walkforward": {"manifest_path": str(manifest.relative_to(tmp_path))}}

    derived = build_wf_config_from_prod(
        prod_cfg,
        manifest_path=str(manifest.relative_to(tmp_path)),
        base_wf_config=base_cfg,
        strategy_dir=tmp_path,
    )

    assert derived["ranking"]["panel_scoring"]["kind"] == "xgb"


# ── candidate-matched reference selection ─────────────────────────────────────


def _seed_references(strategy_dir: Path) -> None:
    _write_json(
        strategy_dir / "strategy_config.json",
        _prod_config(kind="hf_patchtst", artifact_path="artifacts/prod/model.pt"),
    )
    _write_json(
        strategy_dir / "strategy_config.shadow.json",
        _prod_config(kind="xgb", artifact_path="artifacts/prod/panel-ltr.json"),
    )


def test_select_reference_maps_gbdt_to_shadow(tmp_path: Path) -> None:
    _seed_references(tmp_path)
    ref = select_prod_reference_for_candidate("panel_ltr_xgboost", strategy_dir=tmp_path)
    assert ref == (tmp_path / "strategy_config.shadow.json").resolve()


def test_select_reference_maps_patchtst_to_primary(tmp_path: Path) -> None:
    _seed_references(tmp_path)
    ref = select_prod_reference_for_candidate("hf_patchtst", strategy_dir=tmp_path)
    assert ref == (tmp_path / "strategy_config.json").resolve()


def test_select_reference_unknown_kind_fails_closed(tmp_path: Path) -> None:
    _seed_references(tmp_path)
    with pytest.raises(ValueError, match="no production reference"):
        select_prod_reference_for_candidate("some_unknown_scorer", strategy_dir=tmp_path)


def test_select_reference_empty_kind_fails_closed(tmp_path: Path) -> None:
    _seed_references(tmp_path)
    with pytest.raises(ValueError, match="no declared scorer kind"):
        select_prod_reference_for_candidate(None, strategy_dir=tmp_path)


def test_select_reference_env_override_validated(tmp_path: Path) -> None:
    """A validated env override matching the candidate kind is honored."""
    _seed_references(tmp_path)
    ref = select_prod_reference_for_candidate(
        "panel_ltr_xgboost",
        strategy_dir=tmp_path,
        env_override="strategy_config.shadow.json",
    )
    assert ref == (tmp_path / "strategy_config.shadow.json").resolve()


def test_select_reference_env_override_mismatch_fails_closed(tmp_path: Path) -> None:
    """An env override whose kind != candidate kind FAILS CLOSED (no smuggling)."""
    _seed_references(tmp_path)
    with pytest.raises(ValueError, match="does not match the candidate kind"):
        select_prod_reference_for_candidate(
            "panel_ltr_xgboost",
            strategy_dir=tmp_path,
            env_override="strategy_config.json",  # PatchTST primary, kind=hf_patchtst
        )


# ── same-selected-reference contract: negative then positive ──────────────────


def test_gbdt_candidate_against_patchtst_reference_stays_non_promotable(tmp_path: Path) -> None:
    """NEGATIVE: GBDT candidate vs PatchTST prod (no GBDT ref) → parity FAILS.

    This is the genuine mismatch the previous kind-mutation defeated. With the
    converged contract the builder inherits ``kind=hf_patchtst`` from the
    PatchTST primary, the candidate artifact is GBDT, and the SAME PatchTST
    reference is used for both derivation and parity, so the kind/artifact guard
    fires and the run is non-promotable.
    """
    gbdt_artifact = _gbdt_artifact(tmp_path / "artifacts" / "sim" / "panel-ltr.json", ["a", "b"])
    manifest = tmp_path / "artifacts" / "wf" / "manifest.json"
    _write_manifest(manifest, str(gbdt_artifact.relative_to(tmp_path)))
    _seed_references(tmp_path)

    # Caller WRONGLY selects the PatchTST primary for a GBDT candidate (or no
    # GBDT reference exists). The same reference is used for derivation + parity.
    prod_ref = tmp_path / "strategy_config.json"
    prod_cfg = json.loads(prod_ref.read_text())
    base_cfg = {"walkforward": {"manifest_path": str(manifest.relative_to(tmp_path))}}

    derived = build_wf_config_from_prod(
        prod_cfg,
        manifest_path=str(manifest.relative_to(tmp_path)),
        base_wf_config=base_cfg,
        strategy_dir=tmp_path,
    )
    assert derived["ranking"]["panel_scoring"]["kind"] == "hf_patchtst"

    wf_path = tmp_path / "wf_config.json"
    _write_json(wf_path, derived)

    result = evaluate_wf_config_parity(
        prod_ref,
        wf_path,
        candidate_artifact=gbdt_artifact,
        strategy_dir=tmp_path,
    )
    assert result["passed"] is False
    # The PatchTST-kind-vs-JSON-artifact guard fires on the derived config.
    assert any(
        i.get("path", "").endswith("ranking.panel_scoring.artifact_path")
        for i in result["issues"]
    ), result["issues"]


def test_gbdt_candidate_against_selected_gbdt_reference_passes(tmp_path: Path) -> None:
    """POSITIVE: GBDT candidate WITH the selected GBDT/shadow reference passes.

    Same selected reference for derivation and parity; no kind mutation needed
    because the GBDT/shadow prod config already declares ``kind=xgb``.
    """
    gbdt_artifact = _gbdt_artifact(tmp_path / "artifacts" / "sim" / "panel-ltr.json", ["a", "b"])
    manifest = tmp_path / "artifacts" / "wf" / "manifest.json"
    _write_manifest(manifest, str(gbdt_artifact.relative_to(tmp_path)))
    _seed_references(tmp_path)

    # The selector chooses the GBDT/shadow reference for the GBDT candidate.
    prod_ref = select_prod_reference_for_candidate(
        gbdt_artifact and json.loads(gbdt_artifact.read_text())["kind"],
        strategy_dir=tmp_path,
    )
    assert prod_ref == (tmp_path / "strategy_config.shadow.json").resolve()

    # Point the shadow reference at the candidate artifact so the feature
    # contract matches (the real shadow config points at the prod GBDT artifact).
    prod_cfg = json.loads(prod_ref.read_text())
    prod_cfg["ranking"]["panel_scoring"]["artifact_path"] = str(gbdt_artifact)
    _write_json(prod_ref, prod_cfg)

    base_cfg = {"walkforward": {"manifest_path": str(manifest.relative_to(tmp_path))}}
    derived = build_wf_config_from_prod(
        prod_cfg,
        manifest_path=str(manifest.relative_to(tmp_path)),
        base_wf_config=base_cfg,
        strategy_dir=tmp_path,
    )
    assert derived["ranking"]["panel_scoring"]["kind"] == "xgb"

    wf_path = tmp_path / "wf_config.json"
    _write_json(wf_path, derived)

    result = evaluate_wf_config_parity(
        prod_ref,  # SAME selected reference used for derivation
        wf_path,
        candidate_artifact=gbdt_artifact,
        strategy_dir=tmp_path,
    )
    assert result["passed"] is True, result["issues"]


# ── end-to-end: main() survives a swapped lineup ─────────────────────────────


def test_main_survives_swapped_lineup_without_explicit_prod_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Swapped-lineup scenario, driven through the REAL CLI entrypoint.

    Candidate kind=xgb, primary config kind=hf_patchtst, shadow config
    kind=xgb. Before the fix, ``main()`` used ``--prod-config`` directly
    (defaulting to ``strategy_config.json``, the PatchTST primary) and never
    called ``select_prod_reference_for_candidate`` — so the parity check ran
    against a mismatched reference regardless of the candidate's declared
    kind. This drives ``main()`` itself (not the selector in isolation) with
    NO ``--prod-config`` flag, proving the wiring — not just the function —
    resolves to the kind-matched shadow config.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RENQUANT_STRATEGY_CONFIG", raising=False)
    monkeypatch.setattr(wf_config_builder, "STRATEGY_DIR", tmp_path)

    _seed_references(tmp_path)
    # Point the shadow (xgb) reference's artifact_path at the candidate so a
    # correctly-selected reference produces a passing (not just non-crashing)
    # parity result — mirrors test_gbdt_candidate_against_selected_gbdt_reference_passes.
    gbdt_artifact = _gbdt_artifact(tmp_path / "artifacts" / "sim" / "panel-ltr.json", ["a", "b"])
    shadow_path = tmp_path / "strategy_config.shadow.json"
    shadow_cfg = json.loads(shadow_path.read_text())
    shadow_cfg["ranking"]["panel_scoring"]["artifact_path"] = str(gbdt_artifact)
    _write_json(shadow_path, shadow_cfg)

    manifest = tmp_path / "artifacts" / "wf" / "manifest.json"
    _write_manifest(manifest, str(gbdt_artifact.relative_to(tmp_path)))
    base_path = tmp_path / "base_wf_config.json"
    _write_json(base_path, {"walkforward": {"manifest_path": str(manifest.relative_to(tmp_path))}})
    out_path = tmp_path / "wf_config.json"

    exit_code = wf_config_builder_main([
        "--base-wf-config", str(base_path),
        "--out", str(out_path),
        "--candidate-artifact", str(gbdt_artifact),
        # deliberately NO --prod-config: this is the lineup-swap-survives case
    ])

    assert exit_code == 0
    derived = json.loads(out_path.read_text())
    assert derived["ranking"]["panel_scoring"]["kind"] == "xgb", (
        "main() derived from the PatchTST primary instead of the xgb shadow "
        "reference — the selector is not wired into the CLI path"
    )


def test_main_explicit_prod_config_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit --prod-config that mismatches the candidate kind must
    still fail closed, exactly like select_prod_reference_for_candidate's own
    env_override validation — an explicit override cannot smuggle a wrong
    reference past parity either."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RENQUANT_STRATEGY_CONFIG", raising=False)
    _seed_references(tmp_path)

    gbdt_artifact = _gbdt_artifact(tmp_path / "artifacts" / "sim" / "panel-ltr.json", ["a", "b"])
    manifest = tmp_path / "artifacts" / "wf" / "manifest.json"
    _write_manifest(manifest, str(gbdt_artifact.relative_to(tmp_path)))
    base_path = tmp_path / "base_wf_config.json"
    _write_json(base_path, {"walkforward": {"manifest_path": str(manifest.relative_to(tmp_path))}})
    out_path = tmp_path / "wf_config.json"

    with pytest.raises(SystemExit):
        wf_config_builder_main([
            "--base-wf-config", str(base_path),
            "--out", str(out_path),
            "--candidate-artifact", str(gbdt_artifact),
            "--prod-config", str(tmp_path / "strategy_config.json"),  # PatchTST, mismatched
        ])


# ── orch#799: derive an xgb reference from a blend prod ───────────────────────


def _blend_config(*, component0_artifact: str, pin: str | None = _XGB_PIN) -> dict:
    """A z-blend prod: component[0] = xgb panel leg, component[1] = momentum.

    ``pin=None`` omits the component's ``expected_config_fingerprint`` (to test
    the missing-pin fail-closed path)."""
    component0: dict = {
        "artifact_path": component0_artifact,
        "_role": "component 0 = PRODUCTION panel scorer (rank:pairwise xgb)",
    }
    if pin is not None:
        component0["expected_config_fingerprint"] = pin
    return {
        "ranking": {
            "panel_scoring": {
                "enabled": True,
                "kind": "blend",
                "artifact_path": component0_artifact,
                "buy_floor": "adaptive_mean_std",
                "_zblend_fullbook_note": "OPERATOR OVERRIDE 2026-08-04 z-blend进prod",
                "components": [
                    component0,
                    {
                        "kind": "momentum_residual",
                        "artifact_path": "artifacts/momentum/momentum_artifact_ledger.jsonl",
                    },
                ],
            },
        },
    }


def test_derive_xgb_reference_from_blend_produces_xgb_shaped_ref(tmp_path: Path) -> None:
    """A blend whose component[0] artifact DECLARES xgb yields an xgb reference."""
    art = _gbdt_artifact(tmp_path / "artifacts" / "prod" / "panel-ltr.alpha158_fund.json", ["a", "b"])
    blend = _blend_config(component0_artifact=str(art.relative_to(tmp_path)))

    derived = derive_xgb_reference_from_blend(blend, strategy_dir=tmp_path)

    dps = derived["ranking"]["panel_scoring"]
    assert wf_config_builder._normalize_kind(dps["kind"]) == "xgb"  # normalizes to xgb
    assert dps["kind"] == "panel_ltr_xgboost"  # the artifact's OWN declared kind, verbatim
    assert dps["artifact_path"] == str(art.relative_to(tmp_path))
    assert "components" not in dps  # no longer a blend
    assert "_zblend_fullbook_note" not in dps
    # The pin lands at the canonical single-scorer field downstream validation
    # consumes — not merely echoed into the provenance sidecar.
    assert dps["expected_config_fingerprint"] == _XGB_PIN
    # ADVERSARIAL provenance: the derived ref pins the SAME component fingerprint,
    # so parity compares the live xgb leg (not a phantom).
    assert (
        dps["_orch799_derived_from_blend"]["component_config_fingerprint"] == _XGB_PIN
    )
    # original blend config is untouched (deepcopy, not mutated in place)
    assert blend["ranking"]["panel_scoring"]["kind"] == "blend"
    assert "components" in blend["ranking"]["panel_scoring"]


def test_derived_ref_kind_matches_an_xgb_candidate_for_parity(tmp_path: Path) -> None:
    """The derived reference kind matches an xgb candidate — parity would pass."""
    art = _gbdt_artifact(tmp_path / "artifacts" / "prod" / "panel-ltr.alpha158_fund.json", ["a", "b"])
    blend = _blend_config(component0_artifact=str(art.relative_to(tmp_path)))
    derived = derive_xgb_reference_from_blend(blend, strategy_dir=tmp_path)
    ref_kind = wf_config_builder._normalize_kind(derived["ranking"]["panel_scoring"]["kind"])
    assert ref_kind == wf_config_builder._normalize_kind("panel_ltr_xgboost")  # candidate kind


def test_derive_refuses_non_xgb_component_fail_closed(tmp_path: Path) -> None:
    """ADVERSARIAL: a component whose artifact declares a NON-xgb kind is refused.

    The kind is read from the artifact, never mutated to pass, so a PatchTST (or
    any non-xgb) component cannot be laundered into a passing xgb reference.
    """
    art = tmp_path / "artifacts" / "prod" / "patchtst.json"
    _write_json(art, {"kind": "hf_patchtst", "feature_cols": ["a"]})
    blend = _blend_config(component0_artifact=str(art.relative_to(tmp_path)))
    with pytest.raises(ValueError, match="does not normalize"):
        derive_xgb_reference_from_blend(blend, strategy_dir=tmp_path)


def test_derive_refuses_when_component_artifact_missing(tmp_path: Path) -> None:
    """Fail closed when the component artifact can't be loaded (kind unknowable)."""
    blend = _blend_config(component0_artifact="artifacts/prod/does-not-exist.json")
    with pytest.raises(ValueError):
        derive_xgb_reference_from_blend(blend, strategy_dir=tmp_path)


def test_derive_refuses_when_component_has_no_fingerprint_pin(tmp_path: Path) -> None:
    """ADVERSARIAL (codex #112 review): an xgb component with NO
    expected_config_fingerprint pin is refused — an unpinned reference cannot
    prove it is the pinned production leg."""
    art = _gbdt_artifact(tmp_path / "artifacts" / "prod" / "panel-ltr.alpha158_fund.json", ["a", "b"])
    blend = _blend_config(component0_artifact=str(art.relative_to(tmp_path)), pin=None)
    with pytest.raises(ValueError, match="no expected_config_fingerprint"):
        derive_xgb_reference_from_blend(blend, strategy_dir=tmp_path)


def test_derive_refuses_when_artifact_fingerprint_mismatches_pin(tmp_path: Path) -> None:
    """ADVERSARIAL (codex #112 review): the component declares the xgb kind and a
    pin, but the on-disk artifact's config_fingerprint does NOT match the pin —
    i.e. the configured path was swapped for a different xgb-alias artifact.
    Refused: parity would prove model family, but the pin proves identity."""
    art = _gbdt_artifact(
        tmp_path / "artifacts" / "prod" / "panel-ltr.alpha158_fund.json",
        ["a", "b"],
        config_fingerprint="sha256:deadbeefdeadbeef",  # != the component pin
    )
    blend = _blend_config(component0_artifact=str(art.relative_to(tmp_path)))
    with pytest.raises(ValueError, match="does not match the pinned"):
        derive_xgb_reference_from_blend(blend, strategy_dir=tmp_path)


def test_derive_refuses_when_artifact_has_no_fingerprint(tmp_path: Path) -> None:
    """Fail closed when the on-disk artifact declares no config_fingerprint at
    all — the pin cannot be corroborated, so identity is unproven."""
    art = _gbdt_artifact(
        tmp_path / "artifacts" / "prod" / "panel-ltr.alpha158_fund.json",
        ["a", "b"],
        config_fingerprint=None,  # artifact carries no fingerprint to match
    )
    blend = _blend_config(component0_artifact=str(art.relative_to(tmp_path)))
    with pytest.raises(ValueError, match="does not match the pinned"):
        derive_xgb_reference_from_blend(blend, strategy_dir=tmp_path)


def test_derive_refuses_non_blend_config(tmp_path: Path) -> None:
    """An already-xgb (non-blend) prod config is not a derivation target."""
    prod = _prod_config(kind="xgb", artifact_path="artifacts/prod/model.json")
    with pytest.raises(ValueError, match="not kind=blend"):
        derive_xgb_reference_from_blend(prod, strategy_dir=tmp_path)


def test_derive_refuses_blend_without_components(tmp_path: Path) -> None:
    """A blend missing its components list fails closed."""
    blend = {"ranking": {"panel_scoring": {"kind": "blend", "artifact_path": "x"}}}
    with pytest.raises(ValueError, match="no components"):
        derive_xgb_reference_from_blend(blend, strategy_dir=tmp_path)
