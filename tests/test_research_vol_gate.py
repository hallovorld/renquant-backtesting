"""Tests for the pure helpers of the vol-gate exploratory diagnostic — TRADING-SESSION
fold non-overlap/embargo (the re-homed leakage fix), regime-uniformity fail-closed,
bootstrap CI sanity, and metric computation. (The full backtest needs umbrella data +
xgboost and is run manually; these cover the testable logic.)"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "rvg",
    Path(__file__).resolve().parent.parent / "research" / "research_vol_gate_opportunity_cost.py",
)
rvg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rvg)


def _session_gap(udates, train_end, test_lo):
    """Number of TRADING SESSIONS strictly between train_end and test_lo on the date index."""
    u = np.array(sorted(pd.to_datetime(pd.unique(udates))))
    i_train = int(np.searchsorted(u, np.datetime64(pd.Timestamp(train_end))))
    i_test = int(np.searchsorted(u, np.datetime64(pd.Timestamp(test_lo))))
    return i_test - i_train


def test_purge_is_60_trading_sessions_not_calendar_days():
    """THE leakage fix: with a business-day index, ~60 trading sessions span ~84 calendar
    days. A purge of 60 *calendar* days would leave only ~42 sessions of separation and let
    a fwd_60d label near the cutoff overlap the test interval. Assert the embargo is counted
    in trading sessions: the last training-label-end precedes the first test timestamp by
    >= 60 trading sessions (no label overlap)."""
    dates = pd.to_datetime(pd.date_range("2016-01-01", "2024-12-31", freq="B")).values
    wins = rvg.purged_test_windows(dates, n_folds=5, embargo_sessions=60)
    assert len(wins) >= 1
    udates = np.array(sorted(pd.to_datetime(pd.unique(dates))))
    prev_hi = None
    for train_end, lo, hi in wins:
        assert lo <= hi
        # >= 60 TRADING SESSIONS between the training cutoff and the first test date.
        gap = _session_gap(udates, train_end, lo)
        assert gap >= 60, f"only {gap} trading sessions of embargo (need >= 60)"
        # And the calendar gap is materially LARGER than 60 days (proving sessions != days).
        cal_days = (pd.Timestamp(lo) - pd.Timestamp(train_end)).days
        assert cal_days > 60, f"calendar gap {cal_days}d not > 60 (would be a calendar-day purge)"
        # No training label (ends at train_end, horizon 60 sessions) can reach the test start:
        # train_end + 60 sessions <= the date at index(train_end)+60, which is <= test_lo.
        i_train = int(np.searchsorted(udates, np.datetime64(pd.Timestamp(train_end))))
        label_end_idx = i_train + rvg.LABEL_HORIZON_SESSIONS
        if label_end_idx < len(udates):
            assert pd.Timestamp(udates[label_end_idx]) <= pd.Timestamp(lo)
        # Test folds do NOT share a boundary with the previous fold.
        if prev_hi is not None:
            assert pd.Timestamp(lo) > pd.Timestamp(prev_hi)
        prev_hi = hi


def test_purge_drops_early_fold_without_enough_history():
    """A fold whose test start is fewer than `embargo_sessions` into the index cannot be
    embargoed a full horizon, so it is dropped rather than allowed to leak."""
    dates = pd.to_datetime(pd.date_range("2020-01-01", periods=120, freq="B")).values
    wins = rvg.purged_test_windows(dates, n_folds=5, embargo_sessions=60)
    udates = np.array(sorted(pd.to_datetime(pd.unique(dates))))
    for train_end, lo, _hi in wins:
        assert _session_gap(udates, train_end, lo) >= 60


def test_assert_regime_uniform_passes_when_uniform_and_returns_market_label():
    dates = pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"])
    df = pd.DataFrame({"date": dates, "ticker": ["A", "B", "A", "B"], "regime": [0, 0, 2, 2]})
    out = rvg.assert_regime_uniform_per_date(df)
    assert out.loc[pd.Timestamp("2020-01-02")] == 0
    assert out.loc[pd.Timestamp("2020-01-03")] == 2


def test_assert_regime_uniform_fails_closed_on_conflict():
    dates = pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03"])
    df = pd.DataFrame({"date": dates, "ticker": ["A", "B", "A"], "regime": [0, 1, 2]})
    with pytest.raises(ValueError, match="NOT uniform"):
        rvg.assert_regime_uniform_per_date(df)


def test_annualize_basic_and_small_sample():
    assert rvg.annualize(pd.Series([0.01, 0.02]))["n"] == 2          # <6 → no full metrics
    s = pd.Series([0.01] * 12)
    m = rvg.annualize(s)
    assert m["n"] == 12 and m["ann_ret"] > 0 and m["hit"] == 1.0
    assert m["maxDD"] == 0.0                                          # all-positive → no drawdown


def test_block_bootstrap_ci_brackets_mean_and_flags_zero():
    rng = np.random.default_rng(1)
    # a clearly-positive series: CI should exclude 0
    pos = rng.normal(0.02, 0.005, size=120)
    mean, lo, hi = rvg.block_bootstrap_ci(pos, block=3, n_boot=1000)
    assert lo <= mean <= hi and lo > 0
    # a zero-mean noisy series: CI should include 0
    noise = rng.normal(0.0, 0.05, size=120)
    _, lo2, hi2 = rvg.block_bootstrap_ci(noise, block=3, n_boot=1000)
    assert lo2 <= 0 <= hi2


def test_block_bootstrap_ci_handles_tiny_input():
    mean, lo, hi = rvg.block_bootstrap_ci(np.array([0.01, 0.02]), block=3)
    assert np.isfinite(mean) and np.isnan(lo) and np.isnan(hi)        # too short for blocks
