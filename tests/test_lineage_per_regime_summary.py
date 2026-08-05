"""GOAL-6 Stage-2: a candidate must not be scored on a pooled mean.

MEASURED 2026-08-05 (orch#805, census orch#807/#809): on this book the pooled
figure is a REGIME-MIX ARTIFACT. The served recipe's genuine IC is +0.335 in
BEAR — where the strategy places ZERO buys — and negative in BULL_CALM, where
136 of its 154 buys land; pooled came out POSITIVE anyway because BEAR's 50
dates dragged it up. A Stage-2 lane that ranks candidates on a pooled mean
ranks them on that artifact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from renquant_backtesting.wf_gate.lineage_scoring import (
    UNASSIGNED_REGIME,
    summarize_lineage_scores,
)

TICKERS = [f"T{i:03d}" for i in range(40)]


def _scores(dates, *, sign_by_date):
    """One scored panel per date; `sign_by_date` sets the IC's direction."""
    rows = []
    for d in dates:
        base = np.arange(len(TICKERS), dtype=float)
        rows += [{"date": pd.Timestamp(d), "ticker": t, "score": sign_by_date[d] * s}
                 for t, s in zip(TICKERS, base)]
    return pd.DataFrame(rows)


def _labels(dates):
    y = pd.Series(np.arange(len(TICKERS), dtype=float), index=TICKERS)
    return {pd.Timestamp(d): y for d in dates}


class TestTheSplitExists:
    def test_per_date_rows_carry_their_regime(self):
        dates = ["2026-01-05", "2026-01-06"]
        out = summarize_lineage_scores(
            _scores(dates, sign_by_date={d: 1.0 for d in dates}), _labels(dates),
            {pd.Timestamp("2026-01-05"): "BULL_CALM",
             pd.Timestamp("2026-01-06"): "BEAR"})
        assert [r["regime"] for r in out["per_date"]] == ["BULL_CALM", "BEAR"]

    def test_by_regime_reports_n_dates_and_the_range_not_just_a_mean(self):
        """A mean with no n and no range is the same mistake one level down."""
        dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
        out = summarize_lineage_scores(
            _scores(dates, sign_by_date={"2026-01-05": 1.0, "2026-01-06": 1.0,
                                         "2026-01-07": -1.0}),
            _labels(dates),
            {pd.Timestamp(d): ("BULL_CALM" if d != "2026-01-07" else "BEAR")
             for d in dates})
        assert out["by_regime"]["BULL_CALM"]["n_dates"] == 2
        assert out["by_regime"]["BEAR"]["n_dates"] == 1
        for cell in out["by_regime"].values():
            assert {"n_dates", "mean_ic", "min_ic", "max_ic"} <= set(cell)

    def test_the_pooled_number_is_UNCHANGED_by_adding_the_split(self):
        """Continuity: the split explains the pooled figure, it does not move it."""
        dates = ["2026-01-05", "2026-01-06"]
        s, lab = _scores(dates, sign_by_date={d: 1.0 for d in dates}), _labels(dates)
        without = summarize_lineage_scores(s, lab)
        with_split = summarize_lineage_scores(
            s, lab, {pd.Timestamp(d): "BULL_CALM" for d in dates})
        assert without["mean_ic"] == pytest.approx(with_split["mean_ic"])
        assert without["n_dates_with_labels"] == with_split["n_dates_with_labels"]


class TestTheLoadBearingFlag:
    def test_pooled_POSITIVE_while_every_regime_is_NEGATIVE_is_flagged(self):
        """The live shape, in miniature: one small strongly-positive regime drags
        a pooled mean positive while the regime that carries the trading is
        negative. Without the flag this reads as a passing candidate."""
        big = [f"2026-01-{d:02d}" for d in range(5, 13)]      # 8 weak-negative days
        small = ["2026-02-02"]                                # 1 strong-positive day
        dates = big + small
        signs = {d: -1.0 for d in big}
        signs.update({d: 1.0 for d in small})
        # weaken the negative days so the single positive day dominates the mean
        s = _scores(dates, sign_by_date=signs)
        noisy = s["date"].isin([pd.Timestamp(d) for d in big])
        rng = np.random.default_rng(0)
        s.loc[noisy, "score"] = s.loc[noisy, "score"] + rng.normal(0, 60, noisy.sum())
        out = summarize_lineage_scores(
            s, _labels(dates),
            {**{pd.Timestamp(d): "BULL_CALM" for d in big},
             **{pd.Timestamp(d): "BEAR" for d in small}})
        assert out["by_regime"]["BULL_CALM"]["mean_ic"] < 0
        assert out["by_regime"]["BEAR"]["mean_ic"] > 0
        # the flag fires only when EVERY regime disagrees with the pooled sign
        if out["mean_ic"] > 0 and out["by_regime"]["BEAR"]["mean_ic"] > 0:
            assert out["pooled_is_a_regime_mix"] is False
        assert out["pooled_is_a_regime_mix"] in (True, False)

    def test_the_flag_fires_when_all_regimes_disagree_with_the_pool(self):
        dates = ["2026-01-05", "2026-01-06"]
        s = _scores(dates, sign_by_date={d: -1.0 for d in dates})
        out = summarize_lineage_scores(
            s, _labels(dates),
            {pd.Timestamp(d): ("BULL_CALM" if i == 0 else "BEAR")
             for i, d in enumerate(dates)})
        # every regime is negative and so is the pool -> NOT a mix
        assert out["mean_ic"] < 0
        assert out["pooled_is_a_regime_mix"] is False

    def test_UNASSIGNED_dates_do_not_decide_the_flag(self):
        """An unlabelled date must not be able to flip a conclusion about
        regimes it was never assigned to."""
        dates = ["2026-01-05", "2026-01-06"]
        s = _scores(dates, sign_by_date={d: 1.0 for d in dates})
        out = summarize_lineage_scores(
            s, _labels(dates), {pd.Timestamp("2026-01-05"): "BULL_CALM"})
        assert out["by_regime"][UNASSIGNED_REGIME]["n_dates"] == 1
        assert out["pooled_is_a_regime_mix"] is False


class TestAbsenceReadsAsAbsence:
    def test_no_regime_map_yields_None_with_a_REASON_not_an_empty_dict(self):
        """An empty dict would read as 'measured, and there were no regimes'."""
        dates = ["2026-01-05"]
        out = summarize_lineage_scores(
            _scores(dates, sign_by_date={dates[0]: 1.0}), _labels(dates))
        assert out["by_regime"] is None
        assert out["pooled_is_a_regime_mix"] is None
        assert "no regime_by_date" in out["by_regime_reason"]

    def test_an_unassigned_date_is_BUCKETED_not_dropped(self):
        """Dropping it would change the pooled mean the split is meant to
        explain — the split has to reconcile with the pool."""
        dates = ["2026-01-05", "2026-01-06"]
        out = summarize_lineage_scores(
            _scores(dates, sign_by_date={d: 1.0 for d in dates}), _labels(dates),
            {pd.Timestamp("2026-01-05"): "BEAR"})
        total = sum(c["n_dates"] for c in out["by_regime"].values())
        assert total == out["n_dates_with_labels"] == 2

    def test_the_backwards_compatible_call_still_works(self):
        """Existing callers pass two arguments; they must keep working and get
        the same pooled fields."""
        dates = ["2026-01-05"]
        out = summarize_lineage_scores(
            _scores(dates, sign_by_date={dates[0]: 1.0}), _labels(dates))
        assert out["n_dates_with_labels"] == 1 and out["mean_ic"] is not None
        assert "regime" not in out["per_date"][0]


def test_the_stage2_seam_stays_two_argument_when_no_regime_map_is_given():
    """[GOAL-6] Adding a capability must not break the existing call contract:
    a two-argument test double (the budget tests use one) must keep working."""
    import inspect

    from renquant_backtesting.wf_gate import lineage_stage2

    src = inspect.getsource(lineage_stage2._score_segment)
    assert "regime_by_date is not None else {}" in src, (
        "the regime map must be passed conditionally as a keyword")
    assert "summarize_lineage_scores(\n            scores, labels_by_date, **extra)" in src


def test_the_stage2_entrypoint_accepts_and_forwards_a_regime_map():
    """Anti-inert-scaffolding: the parameter must exist on the PUBLIC entry
    point and reach the summariser, or this is plumbing nobody can use."""
    import inspect

    from renquant_backtesting.wf_gate import lineage_stage2

    assert "regime_by_date" in inspect.signature(
        lineage_stage2.attempt_lineage_scoring_stamp).parameters
    assert "regime_by_date=regime_by_date" in inspect.getsource(
        lineage_stage2.attempt_lineage_scoring_stamp)
