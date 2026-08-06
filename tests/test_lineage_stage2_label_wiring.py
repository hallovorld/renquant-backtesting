"""GOAL-6: the Stage-2 lane now scores AGAINST something.

MEASURED 2026-08-04: the stamp scored 124 of 125 windows and carried
`label_summary: null`, because the gate's call site supplied no
`labels_by_date`. A lane that ranks candidates and cannot say whether its
scores relate to outcomes ranks them on nothing.

These tests hold the label builder to the stamp's own contract: it NEVER
raises, and every way it can fail is a DISTINCT stated reason rather than one
catch-all that reads like "the caller did not ask".
"""
from __future__ import annotations

import pandas as pd
import pytest

from renquant_backtesting.wf_gate.lineage_stage2 import labels_by_date_from_panel


def _panel():
    return pd.DataFrame({
        "date": ["2026-01-02", "2026-01-02", "2026-01-05", "2026-01-05"],
        "ticker": ["AAA", "BBB", "AAA", "BBB"],
        "fwd_60d_excess": [0.1, -0.2, 0.3, None],
    })


class TestItBuildsWhatTheSummaryExpects:
    def test_one_series_per_date_indexed_by_ticker(self):
        labels, prov = labels_by_date_from_panel(_panel(), "fwd_60d_excess")
        assert sorted(str(d.date()) for d in labels) == ["2026-01-02", "2026-01-05"]
        assert labels[pd.Timestamp("2026-01-02")].loc["AAA"] == pytest.approx(0.1)
        assert prov["n_dates"] == 2 and prov["n_rows"] == 3

    def test_NULL_labels_are_dropped_not_carried_as_nan(self):
        labels, _ = labels_by_date_from_panel(_panel(), "fwd_60d_excess")
        assert list(labels[pd.Timestamp("2026-01-05")].index) == ["AAA"]

    def test_the_keys_are_TIMESTAMPS_because_the_summary_looks_them_up_that_way(self):
        """`summarize_lineage_scores` does `labels_by_date.get(pd.Timestamp(d))`
        — string keys would silently match nothing and every date would be
        skipped, which is indistinguishable from having no labels at all."""
        labels, _ = labels_by_date_from_panel(_panel(), "fwd_60d_excess")
        assert all(isinstance(k, pd.Timestamp) for k in labels)


class TestEveryFailureIsItsOwnStatedReason:
    """One catch-all reason would read like "the caller did not ask for
    labels", which after this change is false."""

    @pytest.mark.parametrize("panel,label,expected", [
        (None, "y", "no sanity panel was supplied"),
        (pd.DataFrame(), "y", "the sanity panel is empty"),
    ])
    def test_absent_and_empty_are_different_facts(self, panel, label, expected):
        labels, prov = labels_by_date_from_panel(panel, label)
        assert labels == {}
        assert prov["unavailable_because"] == expected

    def test_a_MISSING_COLUMN_names_the_column(self):
        _, prov = labels_by_date_from_panel(_panel(), "not_a_column")
        assert "'not_a_column'" in prov["unavailable_because"]

    def test_a_panel_whose_labels_are_ALL_NULL_says_so(self):
        p = _panel()
        p["fwd_60d_excess"] = None
        _, prov = labels_by_date_from_panel(p, "fwd_60d_excess")
        assert "no non-null" in prov["unavailable_because"]

    def test_it_NEVER_raises_even_on_a_hostile_panel(self):
        """The stamp's whole contract is that it cannot break admission."""
        class Hostile:
            columns = ["date", "ticker", "y"]

            def __len__(self):
                return 3

            def __getitem__(self, _):
                raise RuntimeError("boom")

        labels, prov = labels_by_date_from_panel(Hostile(), "y")
        assert labels == {}
        assert "RuntimeError: boom" in prov["unavailable_because"]

    def test_the_provenance_always_names_the_label_column_and_source(self):
        for panel in (None, pd.DataFrame(), _panel()):
            _, prov = labels_by_date_from_panel(panel, "fwd_60d_excess")
            assert prov["label_col"] == "fwd_60d_excess"
            assert prov["source"] == "sanity_panel"
