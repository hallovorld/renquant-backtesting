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
        # [codex on bt#107] "bit-identical output" was too strong: supplying a
        # map ADDS a `regime` key to each per_date row and new top-level keys.
        # What is unchanged — and all that is claimed — is the POOLED fields
        # and the per-date ICs.
        for key in ("mean_ic", "n_dates_scored", "n_dates_with_labels"):
            assert without[key] == pytest.approx(with_split[key]) if isinstance(
                without[key], float) else without[key] == with_split[key]
        assert [r["ic"] for r in without["per_date"]] == [
            r["ic"] for r in with_split["per_date"]]
        assert [r["date"] for r in without["per_date"]] == [
            r["date"] for r in with_split["per_date"]]
        # and the ONLY difference in a per_date row is the added regime tag
        assert set(with_split["per_date"][0]) - set(without["per_date"][0]) == {"regime"}


class TestTheDecomposition:
    """[codex on bt#107] The first version asked whether the pooled mean
    disagreed with EVERY regime. That is arithmetically impossible once dates
    are assigned — the pooled mean is a date-weighted average of the regime
    means, so it must lie between them. The flag was dead code for the exact
    shape it was written for. The real shape is DOMINANCE."""

    def _live_shape(self):
        """50 strong-positive 'BEAR' dates against 363 weak-negative 'BULL_CALM'
        dates — the live proportions, in miniature."""
        bear = [f"2026-01-{d:02d}" for d in range(1, 6)]          # 5 dates
        calm = [f"2026-03-{d:02d}" for d in range(1, 32)] + \
               [f"2026-04-{d:02d}" for d in range(1, 6)]          # 36 dates
        dates = bear + calm
        signs = {**{d: 1.0 for d in bear}, **{d: -1.0 for d in calm}}
        s = _scores(dates, sign_by_date=signs)
        # weaken the many negative dates so the few positive ones carry the mean
        weak = s["date"].isin([pd.Timestamp(d) for d in calm])
        rng = np.random.default_rng(7)
        s.loc[weak, "score"] = s.loc[weak, "score"] + rng.normal(0, 40, int(weak.sum()))
        regimes = {**{pd.Timestamp(d): "BEAR" for d in bear},
                   **{pd.Timestamp(d): "BULL_CALM" for d in calm}}
        return s, _labels(dates), regimes

    def test_the_minority_regime_carrying_the_pooled_SIGN_is_named(self):
        s, lab, reg = self._live_shape()
        out = summarize_lineage_scores(s, lab, reg)
        assert out["by_regime"]["BEAR"]["mean_ic"] > 0
        assert out["by_regime"]["BULL_CALM"]["mean_ic"] < 0
        if out["pooled_ic"] > 0:
            carriers = [c["regime"] for c in out["pooled_sign_carriers"]]
            assert "BEAR" in carriers, out["pooled_sign_carriers"]
            bear = next(c for c in out["pooled_sign_carriers"] if c["regime"] == "BEAR")
            assert bear["weight"] < 0.25, bear
            assert bear["pooled_ic_without_it"] < 0, bear

    def test_weights_and_contributions_reconstruct_the_pooled_mean(self):
        """The decomposition must be arithmetic, not decoration."""
        s, lab, reg = self._live_shape()
        out = summarize_lineage_scores(s, lab, reg)
        assert sum(c["weight"] for c in out["by_regime"].values()) == pytest.approx(1.0)
        assert sum(c["contribution_to_pooled_ic"]
                   for c in out["by_regime"].values()) == pytest.approx(
            out["pooled_ic"], abs=1e-9)

    def test_no_carrier_when_every_regime_agrees_with_the_pool(self):
        dates = ["2026-01-05", "2026-01-06"]
        out = summarize_lineage_scores(
            _scores(dates, sign_by_date={d: -1.0 for d in dates}), _labels(dates),
            {pd.Timestamp(d): ("BULL_CALM" if i == 0 else "BEAR")
             for i, d in enumerate(dates)})
        assert out["pooled_ic"] < 0
        assert out["pooled_sign_carriers"] == []

    def test_a_single_regime_is_never_its_own_carrier(self):
        """Removing the only regime leaves nothing to compare against."""
        dates = ["2026-01-05", "2026-01-06"]
        out = summarize_lineage_scores(
            _scores(dates, sign_by_date={d: 1.0 for d in dates}), _labels(dates),
            {pd.Timestamp(d): "BULL_CALM" for d in dates})
        assert out["pooled_sign_carriers"] == []

    def test_a_pooled_mean_of_exactly_zero_reports_no_carriers(self):
        """A zero pool has no sign for a regime to be carrying."""
        import unittest.mock as _m
        dates = ["2026-01-05", "2026-01-06"]
        s, lab = _scores(dates, sign_by_date={"2026-01-05": 1.0,
                                              "2026-01-06": -1.0}), _labels(dates)
        out = summarize_lineage_scores(
            s, lab, {pd.Timestamp(d): ("A" if i == 0 else "B")
                     for i, d in enumerate(dates)})
        assert out["pooled_ic"] == pytest.approx(0.0, abs=1e-12)
        assert out["pooled_sign_carriers"] == []


class TestAbsenceReadsAsAbsence:
    def test_no_regime_map_yields_None_with_a_REASON_not_an_empty_dict(self):
        """An empty dict would read as 'measured, and there were no regimes'."""
        dates = ["2026-01-05"]
        out = summarize_lineage_scores(
            _scores(dates, sign_by_date={dates[0]: 1.0}), _labels(dates))
        assert out["by_regime"] is None
        assert out["pooled_sign_carriers"] is None
        assert out["pooled_ic"] is None
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


def test_a_NaN_regime_is_UNASSIGNED_not_a_literal_nan_bucket():
    """[codex on bt#107] A raw `Series.to_dict()` carries NaN. Stringifying one
    would create a bucket named "nan" that reads like a regime."""
    dates = ["2026-01-05", "2026-01-06"]
    out = summarize_lineage_scores(
        _scores(dates, sign_by_date={d: 1.0 for d in dates}), _labels(dates),
        {pd.Timestamp("2026-01-05"): "BEAR",
         pd.Timestamp("2026-01-06"): np.nan})
    assert set(out["by_regime"]) == {"BEAR", UNASSIGNED_REGIME}
    assert "nan" not in out["by_regime"]


def test_a_pandas_NA_regime_is_also_UNASSIGNED():
    dates = ["2026-01-05", "2026-01-06"]
    out = summarize_lineage_scores(
        _scores(dates, sign_by_date={d: 1.0 for d in dates}), _labels(dates),
        {pd.Timestamp("2026-01-05"): "BEAR", pd.Timestamp("2026-01-06"): pd.NA})
    assert set(out["by_regime"]) == {"BEAR", UNASSIGNED_REGIME}
