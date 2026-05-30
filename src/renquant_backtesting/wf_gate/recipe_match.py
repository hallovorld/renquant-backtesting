"""Recipe-fingerprint helpers — Phase 3b lift out of runner.py.

These four pure functions define the contract by which a candidate artifact's
recipe is matched against pre-trained walk-forward manifest entries. Lifting
them here:

1. removes the recipe logic from the 2525-line runner — easier to review;
2. enables independent unit-testing of edge cases (esp. the 2026-05-27 fix:
   ``feature_source_contract_keys`` instead of hashing the whole prose contract);
3. unblocks Phase 3c lift of ``_manifest_recipe_usage`` (which uses these as
   building blocks).

The original ``runner._semantic_params`` / ``_recipe_*`` symbols remain so the
umbrella copy stays byte-equivalent. Phase 5 flips them to import from here.

The 2026-05-27 incident is preserved here as a behavioural anchor: the
fingerprint must depend only on **structural** keys of ``feature_source_contract``,
never the prose docstrings. Tests pin that.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

#: Params that do NOT participate in the recipe fingerprint — execution
#: controls (thread counts, verbosity) whose change must not invalidate
#: otherwise comparable WF manifests.
EXECUTION_ONLY_PARAM_KEYS: frozenset[str] = frozenset({
    "nthread",
    "n_jobs",
    "num_threads",
    "total_steps",
    "warmup_steps",
    "verbosity",
    "verbose",
    "silent",
})


def semantic_params(params: dict) -> dict:
    """Return learner params that define the statistical recipe.

    Drops execution controls (thread counts, verbosity); keeps learning
    parameters (eta, max_depth, objective, seed, tree_method, …).
    """
    if not isinstance(params, dict):
        return {}
    return {
        k: v for k, v in params.items()
        if str(k) not in EXECUTION_ONLY_PARAM_KEYS
    }


def feature_source_contract_keys(artifact: dict) -> list[str]:
    """Structural projection of the feature-source contract.

    2026-05-27 behavioural anchor: hash only the sorted keys of
    ``feature_source_contract`` (e.g. ``raw``, ``panel``) — never the prose
    values. Editing prose (e.g. during the subrepo refactor) used to break
    recipe matches without any behavioural change.
    """
    contract = artifact.get("feature_source_contract")
    if not isinstance(contract, dict):
        return []
    return sorted(str(k) for k in contract.keys())


def recipe_projection(artifact: dict) -> dict:
    """Return the model-recipe fields a WF manifest must match.

    A current production artifact cannot be replayed into old sim windows
    without look-ahead leakage. For historical walk-forward, we therefore
    validate the *retraining* recipe instead: same model kind, ordered
    feature contract, label horizon, and learner params.
    """
    return {
        "kind": artifact.get("kind"),
        "feature_cols": list(artifact.get("feature_cols") or []),
        "feature_norm_kind": list(artifact.get("feature_norm_kind") or []),
        "feature_source_contract_keys": feature_source_contract_keys(artifact),
        "label_col": artifact.get("label_col"),
        "lookahead_days": int(artifact.get("lookahead_days") or 0),
        "params": semantic_params(artifact.get("params") or {}),
    }


def recipe_fingerprint(artifact: dict) -> str:
    """Compute the stable recipe-fingerprint (sha256 prefix) of an artifact."""
    payload = json.dumps(
        recipe_projection(artifact),
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def manifest_recipe_usage(
    manifest_path: Path | None,
    artifact_path: Path,
    *,
    strategy_dir: Path,
) -> dict:
    """Validate that a WF manifest's per-cutoff artifacts share the candidate's recipe.

    Phase 3b.2 lift. ``strategy_dir`` is now an explicit parameter (was the
    module-global ``STRATEGY_DIR`` in runner.py); pass the umbrella's
    ``backtesting/renquant_104`` for production callers and ``tmp_path`` for tests.

    Returns a dict with ``recipe_validated``, the candidate's recipe fingerprint,
    a per-sample report, and a human-readable ``reason`` — same shape as the
    original ``runner._manifest_recipe_usage``.
    """
    # Imported lazily to keep this module's import surface small.
    from .artifact_loader import load_artifact_payload  # noqa: PLC0415

    if manifest_path is None or not manifest_path.exists():
        return {
            "recipe_validated": False,
            "reason": f"manifest not found: {manifest_path}",
        }
    try:
        payload = json.loads(manifest_path.read_text())
        rows = payload.get("retrains", []) if isinstance(payload, dict) else payload
    except Exception as exc:  # noqa: BLE001
        return {"recipe_validated": False, "reason": f"manifest parse failed: {exc}"}
    if not isinstance(rows, list) or not rows:
        return {"recipe_validated": False, "reason": "manifest has no retrain rows"}

    candidate = load_artifact_payload(artifact_path)
    candidate_fp = recipe_fingerprint(candidate)
    candidate_recipe = recipe_projection(candidate)
    seen: set[str] = set()
    sample_reports: list[dict] = []
    for row in rows:
        uri = str((row or {}).get("artifact_uri") or "")
        if not uri or uri in seen:
            continue
        seen.add(uri)
        p = Path(uri)
        if not p.is_absolute():
            p = strategy_dir / p
        if not p.exists():
            sample_reports.append({
                "artifact_uri": uri,
                "exists": False,
                "recipe_matches": False,
            })
            continue
        try:
            sample = load_artifact_payload(p)
            sample_fp = recipe_fingerprint(sample)
            sample_recipe = recipe_projection(sample)
        except Exception as exc:  # noqa: BLE001
            sample_reports.append({
                "artifact_uri": str(p),
                "exists": True,
                "recipe_matches": False,
                "error": str(exc),
            })
            continue
        sample_reports.append({
            "artifact_uri": str(p),
            "exists": True,
            "recipe_matches": sample_fp == candidate_fp,
            "recipe_fingerprint": sample_fp,
            "n_features": len(sample_recipe["feature_cols"]),
            "missing_features_vs_candidate": sorted(
                set(candidate_recipe["feature_cols"]) - set(sample_recipe["feature_cols"])
            )[:10],
            "extra_features_vs_candidate": sorted(
                set(sample_recipe["feature_cols"]) - set(candidate_recipe["feature_cols"])
            )[:10],
        })

    if not sample_reports:
        return {"recipe_validated": False, "reason": "no sample artifacts found in manifest"}
    all_match = all(r.get("recipe_matches") for r in sample_reports)
    return {
        "recipe_validated": bool(all_match),
        "candidate_recipe_fingerprint": candidate_fp,
        "candidate_n_features": len(candidate_recipe["feature_cols"]),
        "manifest_rows_checked": int(len(rows)),
        "manifest_sample_reports": sample_reports,
        "reason": (
            "manifest artifacts match candidate recipe"
            if all_match else
            "manifest artifacts do not match candidate recipe"
        ),
    }


def matching_manifest_for_recipe(
    *,
    artifact_path: Path,
    preferred_manifest: Path | None,
    strategy_dir: Path,
    search_dir: Path | None = None,
) -> tuple[Path | None, dict]:
    """Return a WF manifest whose artifacts share the candidate's recipe.

    Phase 3b.3 lift. ``strategy_dir`` is now a required explicit parameter
    (was ``STRATEGY_DIR`` runner-global). The selection policy preserved from
    runner._matching_manifest_for_recipe:

    1. If a preferred manifest is supplied: validate it as the evidence
       contract and fail closed on mismatch (do NOT silently substitute).
    2. Otherwise auto-discover ``walkforward_manifest*.json`` in ``search_dir``
       (defaults to ``strategy_dir/artifacts/sim``).
    3. Pick the recipe-matching candidate with the most rows; tiebreak by URI.
    4. If nothing matches, return the first checked manifest (so the gate can
       still fail closed with informative metadata).
    """
    checked: list[tuple[Path, dict]] = []
    seen: set[Path] = set()

    def add(path: Path | None) -> None:
        if path is None:
            return
        p = path if path.is_absolute() else strategy_dir / path
        try:
            key = p.resolve()
        except Exception:  # noqa: BLE001
            key = p
        if key in seen:
            return
        seen.add(key)
        checked.append((p, manifest_recipe_usage(p, artifact_path, strategy_dir=strategy_dir)))

    add(preferred_manifest)
    if preferred_manifest is not None and checked:
        p, usage = checked[0]
        usage = dict(usage)
        usage["checked_manifest_count"] = 1
        usage["manifest_selection_policy"] = "preferred_manifest_required"
        return p, usage

    root = search_dir or (strategy_dir / "artifacts" / "sim")
    if root.exists():
        for candidate in sorted(root.glob("walkforward_manifest*.json")):
            add(candidate)

    matches = [(p, usage) for p, usage in checked if bool(usage.get("recipe_validated"))]
    if matches:
        matches.sort(
            key=lambda item: (
                int(item[1].get("manifest_rows_checked") or 0),
                str(item[0]),
            ),
            reverse=True,
        )
        return matches[0]

    if checked:
        p, usage = checked[0]
        usage = dict(usage)
        usage["checked_manifest_count"] = len(checked)
        return p, usage
    return None, {
        "recipe_validated": False,
        "reason": "no walkforward manifests found",
        "checked_manifest_count": 0,
    }
