"""The seam between the trainer that STAMPS a cutoff and the gate that READS it.

Three merged changes have to compose for a candidate to be statically evaluable:

1. `renquant-orchestrator#620` — the GBDT trainer stamps `effective_train_cutoff_date`
   at the artifact's TOP LEVEL (`train_gbdt.CUTOFF_FIELD`);
2. `renquant-backtesting#86` — static sanity resolves the training contract wherever it
   is stamped, instead of only at the root;
3. `renquant-backtesting#87` — the eval window can be derived from the artifact's own
   cutoff instead of a fixed 80% of the panel.

Each was tested in its own repo. **The seam between them was not**, and this programme's
expensive defects live exactly there: a producer writing a field a consumer does not read
is how the WF gate spent months answering *"trained_date is wall-clock metadata and
cannot prove OOS label separation"* about artifacts that had no cutoff to find.

These tests assert the composition end to end, from the DOCUMENTED producer field name
through to the OOS contract's verdict — and, equally, that removing the stamp restores
the refusal. A seam test that only passes proves nothing about the seam.
"""

from __future__ import annotations

import pandas as pd
import pytest

from renquant_backtesting.wf_gate.runner import (
    EVAL_WINDOW_MODE_CUTOFF,
    _effective_artifact_cutoff,
    _validate_static_sanity_oos_contract,
    derive_static_eval_start,
)

#: The producer's field name, transcribed from renquant-orchestrator's
#: `train_gbdt.CUTOFF_FIELD`. Hardcoded ON PURPOSE: importing the orchestrator here
#: would make this repo depend on a sibling's internals, which is the boundary
#: violation the twin-implementation registry (orch#623) exists to prevent. If the
#: producer renames it, THIS test must fail — that is the seam breaking, and a silent
#: rename is precisely the failure being guarded.
PRODUCER_CUTOFF_FIELD = "effective_train_cutoff_date"

PANEL = pd.bdate_range("2014-01-02", "2026-05-01")


def _artifact(cutoff: str | None, lookahead: int = 60) -> dict:
    art: dict = {"lookahead_days": lookahead, "trained_date": "2026-05-18"}
    if cutoff is not None:
        art[PRODUCER_CUTOFF_FIELD] = cutoff
    return art


# --- the composition, end to end ---------------------------------------------

def test_a_stamped_artifact_flows_all_the_way_to_a_PASSING_oos_contract():
    """THE SEAM. Producer field -> resolver -> cutoff-derived window -> contract."""
    art = _artifact("2025-06-02")

    cutoff = _effective_artifact_cutoff(art)
    assert cutoff == pd.Timestamp("2025-06-02"), "step 1: the resolver must see it"

    start, meta = derive_static_eval_start(PANEL, artifact=art,
                                           mode=EVAL_WINDOW_MODE_CUTOFF)
    assert start is not None, "step 2: a window must be derivable from that cutoff"
    assert meta["eval_window_safe_last_label"] == "2025-08-25"

    verdict = _validate_static_sanity_oos_contract(art, start)
    assert verdict["passed"] is True, f"step 3: {verdict.get('reason')}"
    assert verdict["safe_last_label_date"] == "2025-08-25"


def test_without_the_stamp_the_gate_still_REFUSES_and_says_why():
    """The negative case. If this ever passes, the contract has stopped being load
    bearing and the test above proves nothing."""
    art = _artifact(None)

    assert _effective_artifact_cutoff(art) is None
    start, meta = derive_static_eval_start(PANEL, artifact=art,
                                           mode=EVAL_WINDOW_MODE_CUTOFF)
    assert start is None
    assert "declares no effective cutoff" in meta["eval_window_cutoff_reason"]

    verdict = _validate_static_sanity_oos_contract(art, pd.Timestamp("2024-01-02"))
    assert verdict["passed"] is False
    assert "trained_date is wall-clock metadata" in verdict["reason"], (
        "the refusal must still name the real cause, not a generic failure")


def test_the_refusal_survives_a_trained_date_that_LOOKS_like_a_cutoff():
    """`trained_date` is wall-clock and proves nothing about label separation. An
    artifact carrying only it must not be admitted by accident."""
    art = {"lookahead_days": 60, "trained_date": "2024-01-02"}
    assert _effective_artifact_cutoff(art) is None


# --- the contract's own boundary ---------------------------------------------

def test_labels_reaching_past_the_panel_are_refused_even_when_stamped():
    """A stamp is not a licence. A cutoff whose forward window ends past the panel has
    no out-of-sample dates at all."""
    art = _artifact("2026-02-27")
    start, meta = derive_static_eval_start(PANEL, artifact=art,
                                           mode=EVAL_WINDOW_MODE_CUTOFF)
    assert start is None
    assert "NO out-of-sample window exists" in meta["eval_window_cutoff_reason"]


def test_the_contract_refuses_an_eval_start_at_or_before_safe_last_label():
    """Boundary is strict: equality must refuse, or the last label date leaks into the
    first evaluated date."""
    art = _artifact("2025-06-02")
    safe = pd.Timestamp("2025-08-25")
    assert _validate_static_sanity_oos_contract(art, safe)["passed"] is False
    assert _validate_static_sanity_oos_contract(
        art, safe + pd.offsets.BDay(1))["passed"] is True


@pytest.mark.parametrize("bad", [None, "", "not-a-date", float("nan"), [], {}])
def test_a_malformed_stamp_does_not_resolve_to_a_usable_cutoff(bad):
    """A producer writing garbage must not yield a cutoff the window is derived from."""
    art = {"lookahead_days": 60, PRODUCER_CUTOFF_FIELD: bad}
    assert _effective_artifact_cutoff(art) is None


@pytest.mark.parametrize("numeric", [-1, 0, 1, 1719792000, 20260602, 2026.0, True])
def test_a_NUMERIC_stamp_is_refused_because_pandas_reads_it_as_an_epoch(numeric):
    """THE FAIL-OPEN this seam test found. `pd.Timestamp(-1)` is 1969-12-31 and
    `pd.Timestamp(0)` is 1970-01-01 --- pandas reads a bare int as nanoseconds since
    the epoch. Before the fix a garbage integer stamp resolved to a valid-looking
    cutoff near 1970, so `safe_last_label` was ~1970 and EVERY eval_start after that
    satisfied the OOS contract. A malformed stamp bought admission instead of a
    refusal, in the one guard whose job is proving label separation.
    """
    art = {"lookahead_days": 60, PRODUCER_CUTOFF_FIELD: numeric}
    assert _effective_artifact_cutoff(art) is None, (
        f"{numeric!r} must not resolve to a cutoff")


def test_the_epoch_hazard_is_real_and_not_hypothetical():
    """Anti-vacuity: prove pandas really does this, so the guard above is not
    protecting against an imagined problem."""
    assert pd.Timestamp(-1).year == 1969
    assert pd.Timestamp(0).year == 1970


@pytest.mark.parametrize("good", ["2025-06-02", "2025-06-02T00:00:00"])
def test_a_well_formed_date_string_still_resolves(good):
    """Negative case: the refusals above come from the TYPE, not from the resolver
    having become unconditionally strict."""
    art = {"lookahead_days": 60, PRODUCER_CUTOFF_FIELD: good}
    assert _effective_artifact_cutoff(art) == pd.Timestamp("2025-06-02")


def test_a_real_timestamp_or_datetime_object_still_resolves():
    """Producers may hand over a date object rather than a string; that is a date
    expressed as a date and must keep working."""
    import datetime as _dt
    for v in (pd.Timestamp("2025-06-02"), _dt.date(2025, 6, 2),
              _dt.datetime(2025, 6, 2)):
        art = {"lookahead_days": 60, PRODUCER_CUTOFF_FIELD: v}
        assert _effective_artifact_cutoff(art) == pd.Timestamp("2025-06-02"), v


def test_an_explicit_zero_lookahead_still_composes():
    """A declared 0 horizon is information; the seam must carry it rather than treating
    it as absent."""
    art = _artifact("2025-06-02", lookahead=0)
    start, meta = derive_static_eval_start(PANEL, artifact=art,
                                           mode=EVAL_WINDOW_MODE_CUTOFF)
    assert start is not None
    assert meta["eval_window_lookahead_days"] == 0
    assert _validate_static_sanity_oos_contract(art, start)["passed"] is True
