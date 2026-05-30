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
