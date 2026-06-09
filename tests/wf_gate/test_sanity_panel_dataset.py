"""Fix A regression: the sanity panel must come from the model's OWN training
dataset (training_contract.dataset), not a hardcoded one.

The 2026-06-09 false-negative: a PatchTST model trained on transformer_v4_wl200
scored IC -0.017 on the hardcoded alpha158_291 panel vs its true +0.11, because a
CSRankNorm model on a different dataset sees out-of-distribution features. See
renquant-model doc #36.
"""
from __future__ import annotations

import pandas as pd
import pytest

from renquant_backtesting.wf_gate.runner import _load_sanity_panel


def _self_contained(tmp_path, feats, label="fwd_60d_excess"):
    p = tmp_path / "transformer_v4_wl200_clean.parquet"
    n = 30
    df = pd.DataFrame({
        "ticker": ["AAA"] * n,
        "date": pd.date_range("2025-01-01", periods=n, freq="D"),
        label: [0.01] * n,
        **{f: [0.5] * n for f in feats},
    })
    df.to_parquet(p)
    return p


def test_uses_recorded_training_dataset(tmp_path):
    feats = ["KMID", "KLEN", "mean_sentiment"]
    p = _self_contained(tmp_path, feats)
    panel, meta = _load_sanity_panel(feats, "fwd_60d_excess", dataset_path=p)
    assert meta["sanity_dataset_source"] == "training_contract.dataset"
    assert str(p) == meta["sanity_feature_panel"]
    assert all(f in panel.columns for f in feats)
    assert len(panel) == 30


def test_recorded_dataset_missing_feature_fails_closed(tmp_path):
    # dataset lacks one of the requested features → must NOT silently fall back
    p = _self_contained(tmp_path, ["KMID", "KLEN"])
    with pytest.raises(KeyError, match="missing"):
        _load_sanity_panel(["KMID", "KLEN", "NOT_IN_DATASET"], "fwd_60d_excess", dataset_path=p)


def test_recorded_dataset_missing_label_fails_closed(tmp_path):
    p = _self_contained(tmp_path, ["KMID"], label="fwd_60d_excess")
    with pytest.raises(KeyError):
        _load_sanity_panel(["KMID"], "fwd_20d_excess", dataset_path=p)  # label not present


def test_none_dataset_path_uses_fallback_signature(tmp_path):
    # dataset_path=None must not crash on the new branch; it proceeds to the
    # historical loader (which raises FileNotFoundError when the data is absent).
    with pytest.raises((FileNotFoundError, KeyError)):
        _load_sanity_panel(["KMID"], "fwd_60d_excess", dataset_path=None)
