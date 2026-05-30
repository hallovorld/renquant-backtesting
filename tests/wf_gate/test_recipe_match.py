"""Tests for the lifted recipe_match helpers — Phase 3b verification.

Pins behavioural invariants that the runner.py version always had:
- nthread / verbosity changes do NOT change the fingerprint
- prose changes to feature_source_contract values do NOT change the fingerprint
  (the 2026-05-27 incident anchor)
- changes to feature_cols / params / kind DO change the fingerprint
"""
from __future__ import annotations

from renquant_backtesting.wf_gate.recipe_match import (
    EXECUTION_ONLY_PARAM_KEYS,
    feature_source_contract_keys,
    recipe_fingerprint,
    recipe_projection,
    semantic_params,
)


def test_semantic_params_drops_execution_keys() -> None:
    p = {"nthread": 8, "n_jobs": 4, "eta": 0.05, "max_depth": 5, "verbosity": 0}
    out = semantic_params(p)
    assert out == {"eta": 0.05, "max_depth": 5}


def test_semantic_params_nondict_returns_empty() -> None:
    assert semantic_params(None) == {}  # type: ignore[arg-type]
    assert semantic_params("xgboost") == {}  # type: ignore[arg-type]


def test_feature_source_contract_keys_returns_sorted_keys_only() -> None:
    art = {"feature_source_contract": {
        "raw": "apply means/stds before scoring",
        "panel": "panel normalisation rule",
    }}
    assert feature_source_contract_keys(art) == ["panel", "raw"]


def test_feature_source_contract_keys_handles_missing() -> None:
    assert feature_source_contract_keys({}) == []
    assert feature_source_contract_keys({"feature_source_contract": "not a dict"}) == []


def test_recipe_fingerprint_stable_across_thread_count() -> None:
    """Execution-only changes must NOT shift the fingerprint."""
    base = {
        "kind": "panel_ltr_xgboost",
        "feature_cols": ["a", "b"],
        "feature_norm_kind": ["global_z", "robust_z"],
        "feature_source_contract": {"raw": "x", "panel": "y"},
        "label_col": "fwd_60d_excess",
        "lookahead_days": 60,
        "params": {"eta": 0.05, "max_depth": 5, "nthread": 8},
    }
    other = dict(base, params={"eta": 0.05, "max_depth": 5, "nthread": 16})
    assert recipe_fingerprint(base) == recipe_fingerprint(other)


def test_recipe_fingerprint_stable_across_prose_changes() -> None:
    """The 2026-05-27 incident: prose change must NOT shift the fingerprint."""
    base = {
        "kind": "panel_ltr_xgboost",
        "feature_cols": ["a"],
        "feature_norm_kind": [],
        "feature_source_contract": {"raw": "apply means/stds before scoring"},
        "label_col": "fwd_60d_excess",
        "lookahead_days": 60,
        "params": {},
    }
    after_doc_edit = dict(
        base,
        feature_source_contract={"raw": "different prose, same structure"},
    )
    assert recipe_fingerprint(base) == recipe_fingerprint(after_doc_edit)


def test_recipe_fingerprint_changes_with_features() -> None:
    base = {"kind": "p", "feature_cols": ["a"], "feature_norm_kind": [],
            "feature_source_contract": {}, "label_col": "y", "lookahead_days": 60,
            "params": {}}
    other = dict(base, feature_cols=["a", "b"])
    assert recipe_fingerprint(base) != recipe_fingerprint(other)


def test_recipe_fingerprint_changes_with_kind() -> None:
    base = {"kind": "panel_ltr_xgboost", "feature_cols": [], "feature_norm_kind": [],
            "feature_source_contract": {}, "label_col": "y", "lookahead_days": 60,
            "params": {}}
    other = dict(base, kind="hf_patchtst")
    assert recipe_fingerprint(base) != recipe_fingerprint(other)


def test_recipe_projection_uses_zero_for_missing_lookahead() -> None:
    assert recipe_projection({}).get("lookahead_days") == 0


def test_execution_only_keys_immutable() -> None:
    assert "nthread" in EXECUTION_ONLY_PARAM_KEYS
    assert "verbosity" in EXECUTION_ONLY_PARAM_KEYS
    assert isinstance(EXECUTION_ONLY_PARAM_KEYS, frozenset)
