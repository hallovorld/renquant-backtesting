"""The default must be byte-identical, and the alternative must be able to help.

Issue #84's remaining half: the static sanity window is cut at a fixed 80% of the
panel's dates with no reference to the artifact it is supposed to be out-of-sample
*for*, so any artifact whose cutoff sits later than that fraction is refused no
matter how much usable panel tail remains.

Two things need pinning. First, that switching the derivation into a function did
not change the historical behaviour — the window decides which dates are scored, so
it decides the IC, so it decides which fold walk-forward selection picks, and a
silent change there would move the gate. Second, that the alternative actually
resolves cases the fixed rule refuses, and that the comparability hazard it
introduces is handled rather than merely mentioned.
"""

from __future__ import annotations

import pandas as pd
import pytest

from renquant_backtesting.wf_gate.runner import (
    EVAL_WINDOW_FIXED_FRACTION,
    EVAL_WINDOW_MODE_CUTOFF,
    EVAL_WINDOW_MODE_ENV,
    EVAL_WINDOW_MODE_FIXED,
    common_eval_start,
    derive_static_eval_start,
    eval_window_mode,
)

PANEL = pd.bdate_range("2014-01-02", "2026-05-01")


def _legacy(dates) -> tuple[object, int]:
    """The expression this replaced, transcribed from runner.py at c3e4739."""
    distinct = sorted(pd.Timestamp(d) for d in pd.Series(list(dates)).unique())
    cut = distinct[int(len(distinct) * 0.8)]
    after = [d for d in distinct if d > cut]
    return (after[0] if after else None), len(after)


# --- the default must not move -----------------------------------------------

@pytest.mark.parametrize("n", [2, 5, 10, 47, 100, 251, 1205, 3217])
def test_default_reproduces_the_historical_expression(n):
    dates = pd.bdate_range("2014-01-02", periods=n)
    want, want_n = _legacy(dates)
    got, meta = derive_static_eval_start(dates)
    assert got == want, f"n={n}"
    assert meta["eval_dates_fixed_fraction"] == want_n
    assert meta["eval_window_mode"] == EVAL_WINDOW_MODE_FIXED


def test_the_default_mode_is_the_historical_one(monkeypatch):
    monkeypatch.delenv(EVAL_WINDOW_MODE_ENV, raising=False)
    assert eval_window_mode() == EVAL_WINDOW_MODE_FIXED
    assert EVAL_WINDOW_FIXED_FRACTION == 0.8


def test_an_unrecognised_mode_falls_back_to_the_historical_one(monkeypatch):
    """A typo in the env var must not silently select a third behaviour."""
    monkeypatch.setenv(EVAL_WINDOW_MODE_ENV, "artifact-cutoff")  # wrong separator
    assert eval_window_mode() == EVAL_WINDOW_MODE_FIXED


def test_the_artifact_is_ignored_by_the_default_mode():
    """Passing an artifact must not change the default answer, or the opt-in is not
    an opt-in."""
    art = {"effective_train_cutoff_date": "2025-06-02", "lookahead_days": 60}
    without, _ = derive_static_eval_start(PANEL)
    with_art, _ = derive_static_eval_start(PANEL, artifact=art)
    assert without == with_art


# --- the alternative resolves what the fixed rule refuses ---------------------

@pytest.mark.parametrize("cutoff,expect_start,expect_dates", [
    ("2024-06-03", "2024-08-27", 439),
    ("2025-06-02", "2025-08-26", 179),
])
def test_cutoff_mode_yields_a_window_where_fixed_refuses(cutoff, expect_start,
                                                          expect_dates):
    art = {"effective_train_cutoff_date": cutoff, "lookahead_days": 60}
    _, meta = derive_static_eval_start(PANEL, artifact=art)
    # the fixed rule's start is BEFORE this artifact's labels end -> refusal
    assert pd.Timestamp(meta["eval_window_safe_last_label"]) >= \
        pd.Timestamp(meta["eval_start_fixed_fraction"])
    got, _ = derive_static_eval_start(PANEL, artifact=art,
                                      mode=EVAL_WINDOW_MODE_CUTOFF)
    assert got is not None
    assert got.date().isoformat() == expect_start
    assert meta["eval_dates_artifact_cutoff"] == expect_dates


def test_both_candidate_values_are_recorded_regardless_of_mode():
    """The A/B is measured on every run without changing behaviour. If this stops
    holding, the metadata stops being evidence about the choice."""
    art = {"effective_train_cutoff_date": "2025-06-02", "lookahead_days": 60}
    for mode in (EVAL_WINDOW_MODE_FIXED, EVAL_WINDOW_MODE_CUTOFF):
        _, meta = derive_static_eval_start(PANEL, artifact=art, mode=mode)
        assert meta["eval_start_fixed_fraction"] == "2023-11-15"
        assert meta["eval_start_artifact_cutoff"] == "2025-08-26"
        assert meta["eval_start_chosen"] == (
            "2023-11-15" if mode == EVAL_WINDOW_MODE_FIXED else "2025-08-26")


def test_labels_covering_the_whole_panel_yield_NO_window_under_either_mode():
    """Correct refusal, not a bug. A cutoff whose forward window ends past the
    panel's last date has no out-of-sample dates at all, and inventing some would
    be leakage."""
    art = {"effective_train_cutoff_date": "2026-02-27", "lookahead_days": 60}
    got, meta = derive_static_eval_start(PANEL, artifact=art,
                                         mode=EVAL_WINDOW_MODE_CUTOFF)
    assert got is None
    assert meta["eval_dates_artifact_cutoff"] == 0
    assert "NO out-of-sample window exists" in meta["eval_window_cutoff_reason"]


def test_an_artifact_with_no_declared_cutoff_gets_no_cutoff_window():
    """Measured on the real production artifacts: they carry only wall-clock
    `trained_date`, no cutoff key the resolver reads. So this mode cannot rescue
    them --- that needs the artifact to stamp its cutoff, which is a producer-side
    change. Recorded here so the limitation is not rediscovered."""
    got, meta = derive_static_eval_start(PANEL, artifact={"trained_date": "2026-05-18"},
                                         mode=EVAL_WINDOW_MODE_CUTOFF)
    assert got is None
    assert "declares no effective cutoff" in meta["eval_window_cutoff_reason"]
    assert "wall-clock" in meta["eval_window_cutoff_reason"]


# --- the comparability hazard the alternative introduces ---------------------

def test_two_arms_with_different_cutoffs_get_ONE_common_window():
    """Different windows per arm would reintroduce the era confound that voided a
    study on this programme. The common window is the LATEST start, so both arms
    stay out-of-sample AND land on the same rows."""
    a = {"effective_train_cutoff_date": "2024-06-03", "lookahead_days": 60}
    b = {"effective_train_cutoff_date": "2025-06-02", "lookahead_days": 60}
    sa, _ = derive_static_eval_start(PANEL, artifact=a, mode=EVAL_WINDOW_MODE_CUTOFF)
    sb, _ = derive_static_eval_start(PANEL, artifact=b, mode=EVAL_WINDOW_MODE_CUTOFF)
    assert sa < sb
    assert common_eval_start([sa, sb]) == sb == pd.Timestamp("2025-08-26")


def test_an_arm_without_a_window_makes_the_comparison_impossible():
    """Not silently the other arm's window: a comparison missing an arm is not a
    comparison, and falling back to the surviving one would score a candidate
    against nothing."""
    a = {"effective_train_cutoff_date": "2024-06-03", "lookahead_days": 60}
    sa, _ = derive_static_eval_start(PANEL, artifact=a, mode=EVAL_WINDOW_MODE_CUTOFF)
    assert common_eval_start([sa, None]) is None
    assert common_eval_start([]) is None


def test_common_eval_start_accepts_strings_and_timestamps():
    assert common_eval_start(["2025-01-02", pd.Timestamp("2024-01-02")]) == \
        pd.Timestamp("2025-01-02")


# --- degenerate inputs must not pass silently --------------------------------

def test_an_empty_panel_yields_None_with_a_reason():
    got, meta = derive_static_eval_start([])
    assert got is None
    assert meta["eval_window_reason"] == "panel has no dates"
    assert meta["eval_window_panel_dates"] == 0


def test_duplicate_dates_are_collapsed_before_the_fraction_is_taken():
    """The fraction is over DISTINCT dates. A panel is one row per (date, ticker),
    so taking it over rows would put the cut wherever the ticker count happens to
    be dense."""
    dates = list(pd.bdate_range("2020-01-01", periods=10)) * 300
    got, meta = derive_static_eval_start(dates)
    assert meta["eval_window_panel_dates"] == 10
    want, _ = _legacy(dates)
    assert got == want
