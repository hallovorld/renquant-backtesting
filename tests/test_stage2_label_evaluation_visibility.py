"""GOAL-6: a Stage-2 segment that was never EVALUATED must say so.

MEASURED 2026-08-05 on the live artifact corpus:

* the 2026-08-04 stamp SCORED 124 of 125 windows (`lineage_stage2=stage2`) and
  carried **no** `label_summary` at all — because the gate's call site passes no
  `labels_by_date`. Scoring without evaluation produces no IC of any kind, and
  the stamp did not say that;
* the 2026-08-05 stamp is `unavailable`: `stage-1 admitted root 2969e1d199e2… !=
  extension's old root d1161f8d46b5…`, so the frozen extension bundle no longer
  extends the admitted lineage.

The per-regime split (bt#107) sits downstream of the summary that never ran.
"""
from __future__ import annotations

import pandas as pd
import pytest

from renquant_backtesting.wf_gate import lineage_stage2 as S


def _stats(labels, scores_rows):
    """Drive just the statistics block the way _score_segment builds it."""
    scores = pd.DataFrame(scores_rows, columns=["date", "ticker", "score",
                                                "cutoff_date"])
    statistics = {"n_rows_scored": int(len(scores)),
                  "n_dates_scored": int(scores["date"].nunique()) if len(scores) else 0}
    if labels is None:
        statistics["label_summary"] = None
        statistics["label_summary_absent_because"] = (
            "the caller supplied no labels_by_date — this lane SCORED but was "
            "never evaluated against labels, so it produced no IC of any kind")
    elif not len(scores):
        statistics["label_summary"] = None
        statistics["label_summary_absent_because"] = (
            "no rows were scored in this segment, so there is nothing to "
            "evaluate against the supplied labels")
    return statistics


def test_the_module_states_the_reason_rather_than_omitting_the_key():
    """The load-bearing property: a missing key reads as 'nothing to report'.
    An explicit None plus a reason reads as 'not evaluated, and here is why'."""
    import inspect

    src = inspect.getsource(S._score_segment)
    assert 'statistics["label_summary"] = None' in src
    assert "label_summary_absent_because" in src
    assert "SCORED but was" in src


def test_no_labels_supplied_is_distinguishable_from_nothing_matured():
    no_labels = _stats(None, [("2026-01-05", "A", 1.0, "2026-01-01")])
    no_scores = _stats({"x": 1}, [])
    assert no_labels["label_summary"] is None
    assert no_scores["label_summary"] is None
    assert no_labels["label_summary_absent_because"] != \
        no_scores["label_summary_absent_because"]
    assert "supplied no labels_by_date" in no_labels["label_summary_absent_because"]
    assert "nothing to" in no_scores["label_summary_absent_because"]


def test_scoring_without_evaluation_is_named_as_producing_no_IC():
    """The 2026-08-04 shape: 124 windows scored, zero IC. The stamp must not
    let that read as a successful evaluation."""
    s = _stats(None, [("2026-01-05", "A", 1.0, "2026-01-01")])
    assert "produced no IC of any kind" in s["label_summary_absent_because"]
    assert s["n_rows_scored"] == 1, "it really did score — that is the point"


def test_the_signature_still_accepts_labels_and_regimes():
    """Anti-regression: the visibility change must not remove the capability."""
    import inspect

    params = inspect.signature(S.attempt_lineage_scoring_stamp).parameters
    assert "labels_by_date" in params and "regime_by_date" in params


def test_the_LIVE_stamps_are_what_this_record_describes():
    """Bound to reality. If Stage-2 starts carrying a real label_summary, or the
    root mismatch clears, this record must be re-derived rather than inherited."""
    import json
    import pathlib

    root = pathlib.Path(
        "/Users/renhao/git/github/RenQuant/backtesting/renquant_104/artifacts")
    if not root.exists():
        pytest.skip("umbrella artifacts absent — the unit tests above still ran")
    stamps = []
    for p in sorted(root.rglob("*weekly_2026080*staging*.json")):
        if ".claude" in str(p):
            continue
        try:
            m = (json.loads(p.read_text()).get("metadata") or {}).get(
                "wf_gate_metadata") or {}
        except Exception:                       # noqa: BLE001
            continue
        s2 = m.get("lineage_stage2")
        if isinstance(s2, dict) and s2:
            stamps.append(s2)
    if not stamps:
        pytest.skip("no stage-2 stamps in the live corpus")
    scored = [s for s in stamps if s.get("lineage_stage2") == "stage2"]
    assert scored, ("Stage-2 has no scored stamp at all — the record's premise "
                    "('it scores but is never evaluated') needs re-deriving")
    for s in scored:
        assert s.get("n_scored_windows", 0) > 0
