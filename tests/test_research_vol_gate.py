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


def test_purge_is_trading_sessions_not_calendar_days_strict_no_overlap():
    """THE leakage fix + the Codex boundary point: with a business-day index, ~60 trading
    sessions span ~84 calendar days. A purge of 60 *calendar* days would leave only ~42 sessions
    of separation and let a fwd_60d label near the cutoff overlap the test interval. Assert the
    embargo is counted in trading sessions AND that the last training-label-end falls STRICTLY
    before the first test date (no boundary touch): the default embargo strictly exceeds the
    label horizon, so label_end_idx < test_lo_idx, not <=."""
    dates = pd.to_datetime(pd.date_range("2016-01-01", "2024-12-31", freq="B")).values
    wins = rvg.purged_test_windows(dates, n_folds=5, embargo_sessions=rvg.EMB_SESSIONS)
    assert len(wins) >= 1
    assert rvg.EMB_SESSIONS > rvg.LABEL_HORIZON_SESSIONS  # strict separation is achievable
    udates = np.array(sorted(pd.to_datetime(pd.unique(dates))))
    prev_hi = None
    for train_end, lo, hi in wins:
        assert lo <= hi
        # >= EMB_SESSIONS TRADING SESSIONS between the training cutoff and the first test date.
        gap = _session_gap(udates, train_end, lo)
        assert gap >= rvg.EMB_SESSIONS, f"only {gap} trading sessions of embargo"
        # And the calendar gap is materially LARGER than the session count (sessions != days).
        cal_days = (pd.Timestamp(lo) - pd.Timestamp(train_end)).days
        assert cal_days > rvg.EMB_SESSIONS, f"calendar gap {cal_days}d <= embargo (calendar purge)"
        # STRICT no-overlap: a training label (ends at train_end, horizon 60 sessions) ends
        # strictly BEFORE the test start — label_end_idx < test_lo_idx (not <=).
        i_train = int(np.searchsorted(udates, np.datetime64(pd.Timestamp(train_end))))
        i_test = int(np.searchsorted(udates, np.datetime64(pd.Timestamp(lo))))
        label_end_idx = i_train + rvg.LABEL_HORIZON_SESSIONS
        assert label_end_idx < i_test, (
            f"label end idx {label_end_idx} not strictly < test start idx {i_test} (overlap)"
        )
        # Test folds do NOT share a boundary with the previous fold.
        if prev_hi is not None:
            assert pd.Timestamp(lo) > pd.Timestamp(prev_hi)
        prev_hi = hi


def test_purge_rejects_embargo_not_exceeding_label_horizon():
    """A 60-session embargo on a 60-session label leaves the last training label ending exactly
    ON test_lo (boundary overlap). The function must fail closed, not silently leak."""
    dates = pd.to_datetime(pd.date_range("2016-01-01", "2024-12-31", freq="B")).values
    with pytest.raises(ValueError, match="STRICTLY greater"):
        rvg.purged_test_windows(dates, n_folds=5, embargo_sessions=60,
                                label_horizon_sessions=60)


def test_purge_drops_early_fold_without_enough_history():
    """A fold whose test start is fewer than `embargo_sessions` into the index cannot be
    embargoed a full horizon, so it is dropped rather than allowed to leak."""
    dates = pd.to_datetime(pd.date_range("2020-01-01", periods=120, freq="B")).values
    wins = rvg.purged_test_windows(dates, n_folds=5, embargo_sessions=rvg.EMB_SESSIONS)
    udates = np.array(sorted(pd.to_datetime(pd.unique(dates))))
    for train_end, lo, _hi in wins:
        assert _session_gap(udates, train_end, lo) >= rvg.EMB_SESSIONS


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
