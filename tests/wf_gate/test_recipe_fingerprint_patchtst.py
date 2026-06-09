"""Regression guard: PatchTST recipe fingerprint is robust to benign load-path
metadata differences (2026-06-09 orphan-primary mismatch).

An identical PatchTST model loaded two ways must fingerprint the same:
- the contract-sidecar path tags kind=`hf_patchtst_contract_sidecar` and carries
  only the 15 recipe hyperparameters;
- the full-metadata path tags kind=`hf_patchtst` and additionally carries
  runtime/env fields (device, detector_version, epochs, early_stopping_patience,
  shuffle_labels, label_shift_days).
Neither the kind marker nor the runtime fields are part of the statistical
recipe, so they must not change the fingerprint. GBDT fingerprints must be
unaffected by the change.
"""
from __future__ import annotations

from renquant_backtesting.wf_gate.runner import (
    _canonical_recipe_kind,
    _recipe_fingerprint,
    _semantic_params,
)

_RECIPE_PARAMS = {
    "seq_len": 24, "patch_length": 4, "d_model": 64, "n_heads": 4, "n_layers": 2,
    "lr": 0.0001, "weight_decay": 0.3, "lr_scheduler": "cosine", "warmup_ratio": 0.1,
    "nll_loss_weight": 0.5, "ranking_margin": 0.1, "distributional_head": True,
    "film_regime_cond": False, "cross_stock_attn": False, "embargo_days": 60,
}
_RUNTIME_EXTRAS = {
    "epochs": 5, "early_stopping_patience": 2, "device": "mps",
    "shuffle_labels": False, "label_shift_days": 0, "detector_version": "v2026-05-31",
}


def _artifact(kind, params):
    return {
        "kind": kind,
        "feature_cols": [f"f{i}" for i in range(172)],
        "feature_norm_kind": ["zscore"] * 172,
        "label_col": "fwd_60d_excess",
        "lookahead_days": 60,
        "params": params,
    }


def test_contract_sidecar_kind_normalizes():
    assert _canonical_recipe_kind("hf_patchtst_contract_sidecar") == "hf_patchtst"
    assert _canonical_recipe_kind("hf_patchtst") == "hf_patchtst"
    assert _canonical_recipe_kind("panel_ltr_xgboost") == "panel_ltr_xgboost"


def test_orphan_primary_matches_fresh_cut():
    # contract-sidecar load (orphan live primary) vs full-metadata load (fresh cut)
    orphan = _artifact("hf_patchtst_contract_sidecar", dict(_RECIPE_PARAMS))
    fresh = _artifact("hf_patchtst", {**_RECIPE_PARAMS, **_RUNTIME_EXTRAS})
    assert _recipe_fingerprint(orphan) == _recipe_fingerprint(fresh)


def test_runtime_fields_do_not_change_patchtst_fingerprint():
    base = _semantic_params(dict(_RECIPE_PARAMS), "hf_patchtst")
    withextras = _semantic_params({**_RECIPE_PARAMS, **_RUNTIME_EXTRAS}, "hf_patchtst")
    assert base == withextras
    for k in _RUNTIME_EXTRAS:
        assert k not in withextras


def test_real_recipe_change_still_changes_fingerprint():
    a = _artifact("hf_patchtst", dict(_RECIPE_PARAMS))
    b = _artifact("hf_patchtst", {**_RECIPE_PARAMS, "seq_len": 48})  # genuine recipe diff
    assert _recipe_fingerprint(a) != _recipe_fingerprint(b)


def test_gbdt_fingerprint_unaffected():
    # non-patchtst: the execution-only denylist path is used; runtime fields that
    # are NOT in the denylist still participate (behavior unchanged by this fix).
    p1 = _semantic_params({"max_depth": 6, "eta": 0.1, "n_jobs": 8}, "panel_ltr_xgboost")
    assert p1 == {"max_depth": 6, "eta": 0.1}  # n_jobs is execution-only
    # a GBDT-relevant param still matters
    p2 = _semantic_params({"max_depth": 8, "eta": 0.1}, "panel_ltr_xgboost")
    assert _semantic_params({"max_depth": 6, "eta": 0.1}, "panel_ltr_xgboost") != p2
