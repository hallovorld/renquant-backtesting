"""Regression guard: _load_sanity_panel supplements opt-in addendum features.

2026-06-05 Track-B verdict run crashed because addendum features (Track B/C:
mom_carry_12_1, beta_dm, …) live only in the production training panel, not in
the rawlabel or transformer sanity panels. _load_sanity_panel must supplement
ONLY the missing columns from the training panel, leaving the rawlabel base
features untouched so addendum sanity runs stay apples-to-apples with the
non-addendum (baseline) run.
"""
import pandas as pd
import pytest

from renquant_backtesting.wf_gate import runner as wf_runner


def test_load_sanity_panel_supplements_addendum_from_training_panel(
    tmp_path, monkeypatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    dates = pd.bdate_range("2024-01-01", periods=4)
    rows = [(t, d) for t in ("AAA", "BBB") for d in dates]
    raw = pd.DataFrame({
        "ticker": [t for t, _ in rows],
        "date": [d for _, d in rows],
        "alpha_base": [0.1 * i for i in range(len(rows))],
        "fwd_60d_excess_raw": [0.2 * i for i in range(len(rows))],
    })
    raw.to_parquet(data / "alpha158_291_fundamental_dataset_rawlabel.parquet")
    train = raw[["ticker", "date", "alpha_base"]].copy()
    train["mom_carry_12_1"] = [0.5 * i for i in range(len(rows))]
    train.to_parquet(data / "alpha158_291_fundamental_dataset.parquet")

    monkeypatch.setattr(wf_runner, "REPO", tmp_path)

    panel, meta = wf_runner._load_sanity_panel(
        ["alpha_base", "mom_carry_12_1"], "fwd_60d_excess_raw"
    )

    assert "mom_carry_12_1" in panel.columns
    assert panel["mom_carry_12_1"].notna().all()
    assert "alpha_base" in panel.columns
    assert meta["supplement_only_missing"] is True
    assert meta["feature_cols_supplied_by_feature_panel"] == ["mom_carry_12_1"]
    assert "alpha158_291_fundamental_dataset.parquet" in meta["sanity_feature_panel"]


def test_load_sanity_panel_rejects_duplicate_training_panel_keys(
    tmp_path, monkeypatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    dates = pd.bdate_range("2024-01-01", periods=2)
    raw = pd.DataFrame({
        "ticker": ["AAA", "AAA"],
        "date": dates,
        "alpha_base": [0.1, 0.2],
        "fwd_60d_excess_raw": [0.3, 0.4],
    })
    raw.to_parquet(data / "alpha158_291_fundamental_dataset_rawlabel.parquet")
    train = pd.DataFrame({
        "ticker": ["AAA", "AAA", "AAA"],
        "date": [dates[0], dates[0], dates[1]],
        "mom_carry_12_1": [0.5, 0.6, 0.7],
    })
    train.to_parquet(data / "alpha158_291_fundamental_dataset.parquet")
    monkeypatch.setattr(wf_runner, "REPO", tmp_path)

    with pytest.raises(ValueError, match="duplicate"):
        wf_runner._load_sanity_panel(
            ["alpha_base", "mom_carry_12_1"], "fwd_60d_excess_raw"
        )


def test_load_sanity_panel_rejects_incomplete_training_panel_coverage(
    tmp_path, monkeypatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    dates = pd.bdate_range("2024-01-01", periods=2)
    raw = pd.DataFrame({
        "ticker": ["AAA", "AAA"],
        "date": dates,
        "alpha_base": [0.1, 0.2],
        "fwd_60d_excess_raw": [0.3, 0.4],
    })
    raw.to_parquet(data / "alpha158_291_fundamental_dataset_rawlabel.parquet")
    train = pd.DataFrame({
        "ticker": ["AAA"],
        "date": [dates[0]],
        "mom_carry_12_1": [0.5],
    })
    train.to_parquet(data / "alpha158_291_fundamental_dataset.parquet")
    monkeypatch.setattr(wf_runner, "REPO", tmp_path)

    with pytest.raises(ValueError, match="missing values"):
        wf_runner._load_sanity_panel(
            ["alpha_base", "mom_carry_12_1"], "fwd_60d_excess_raw"
        )


def test_load_sanity_panel_drops_tail_edge_coverage_gap(tmp_path, monkeypatch) -> None:
    """A tail-edge coverage gap below the 1% tolerance is DROPPED, not
    hard-failed (rawlabel's last date not yet in the training panel; the model
    scores NaN natively). Regression guard for the 2026-06-06 Track-C specialist
    eval (109/715629 = 0.02% gap).
    """
    data = tmp_path / "data"
    data.mkdir()
    dates = pd.bdate_range("2024-01-01", periods=200)
    raw = pd.DataFrame({
        "ticker": ["AAA"] * 200,
        "date": dates,
        "alpha_base": [0.001 * i for i in range(200)],
        "fwd_60d_excess_raw": [0.002 * i for i in range(200)],
    })
    raw.to_parquet(data / "alpha158_291_fundamental_dataset_rawlabel.parquet")
    train = pd.DataFrame({
        "ticker": ["AAA"] * 199,
        "date": dates[:199],
        "mom_carry_12_1": [0.5 * i for i in range(199)],
    })
    train.to_parquet(data / "alpha158_291_fundamental_dataset.parquet")
    monkeypatch.setattr(wf_runner, "REPO", tmp_path)

    panel, meta = wf_runner._load_sanity_panel(
        ["alpha_base", "mom_carry_12_1"], "fwd_60d_excess_raw"
    )

    assert len(panel) == 199  # the 1 gap row dropped
    assert panel["mom_carry_12_1"].notna().all()
    assert meta["supplement_only_missing"] is True
