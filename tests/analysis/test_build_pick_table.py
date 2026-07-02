"""Unit tests for build_pick_table (--dump-predictions, the durable OOS pick table)."""
from pathlib import Path

import pandas as pd

from renquant_backtesting.analysis.analyze_manifest_sanity_placebo import (
    build_pick_table,
)


def _fixture():
    dates = pd.to_datetime(["2025-01-02"] * 10 + ["2025-01-03"] * 10)
    tickers = [f"T{i}" for i in range(10)] * 2
    val = pd.DataFrame({
        "date": dates,
        "ticker": tickers,
        "fwd_60d_excess": [0.01 * i for i in range(10)] * 2,
    })
    mu = pd.Series([float(i) for i in range(10)] + [float(9 - i) for i in range(10)],
                   index=val.index)
    regimes = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
        "regime": ["BULL_CALM", "BEAR"],
    })
    return val, mu, regimes


def test_columns_and_row_count():
    val, mu, regimes = _fixture()
    out = build_pick_table(val, mu, "fwd_60d_excess", regimes)
    assert list(out.columns) == ["date", "ticker", "fwd_60d_excess", "score",
                                 "regime", "decile_rank"]
    assert len(out) == 20


def test_decile_rank_is_per_date_and_top_is_9():
    val, mu, regimes = _fixture()
    out = build_pick_table(val, mu, "fwd_60d_excess", regimes)
    for d, g in out.groupby("date"):
        assert g["decile_rank"].min() >= 0 and g["decile_rank"].max() == 9
        top = g.loc[g["score"].idxmax()]
        assert top["decile_rank"] == 9
    # day 2 reverses mu, so the top name flips: deciles must be per-date
    d1 = out[out["date"] == "2025-01-02"].nlargest(1, "score")["ticker"].iloc[0]
    d2 = out[out["date"] == "2025-01-03"].nlargest(1, "score")["ticker"].iloc[0]
    assert d1 != d2


def test_regime_joined_and_nan_scores_dropped():
    val, mu, regimes = _fixture()
    mu2 = mu.copy()
    mu2.iloc[0] = float("nan")
    out = build_pick_table(val, mu2, "fwd_60d_excess", regimes)
    assert len(out) == 19
    assert set(out["regime"].unique()) == {"BULL_CALM", "BEAR"}


def test_empty_regimes_yields_none_column():
    val, mu, _ = _fixture()
    out = build_pick_table(val, mu, "fwd_60d_excess", pd.DataFrame())
    assert "regime" in out.columns
    assert out["regime"].isna().all()
