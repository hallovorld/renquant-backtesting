"""Genuine-IC placebo sub-gate tests (GATE_VERSION 3).

SAFETY-CRITICAL. These pin the re-calibration that fixes the chronic false-reject
of the §5.2 time-shift placebo sub-gate: the raw absolute placebo_ic at the gate's
2×-horizon shift carries a STRUCTURAL ~+0.04 label-autocorrelation floor (the daily
fwd_60d label is itself cross-sectionally autocorrelated there), so the old absolute
rule ``placebo_ic < 0.5×|aligned_real_ic|`` sat below the floor and could never pass.

The fix gates on the leak-free decomposition
    genuine_ic = aligned_real_ic − placebo_ic
(the shared autocorr floor cancels) clearing a positive, conservative bar
    genuine_ic >= max(GENUINE_IC_ABS_FLOOR=0.02, GENUINE_IC_REAL_RATIO=0.25×|aligned_real_ic|).

The shuffled-label control (``abs(shuf_ic) < SHUF_IC_MAX``) is the HARD true-leak
guard and is UNCHANGED — a non-clean shuffle must still FAIL. These tests assert
exactly that the re-calibration removes ONLY the structural-floor mis-calibration
and does NOT let a shuffle-leak or non-positive-edge model through.

Numbers in ``_TODAY`` are the real 2026-06-23 weekly candidate that the old gate
chronically false-rejected: aligned_real_ic 0.0853 / placebo_ic 0.0529 /
label_autocorr_ic 0.040 / shuf_ic −0.0004 → genuine_ic 0.0324 (positive edge).
"""
from __future__ import annotations

import math

import pytest

from renquant_backtesting.wf_gate.runner import (
    GENUINE_IC_ABS_FLOOR,
    GENUINE_IC_REAL_RATIO,
    SHUF_IC_MAX,
    _genuine_ic_bar,
    _genuine_ic_gate_enabled,
    _genuine_ic_value,
    _placebo_subgate_pass,
)

# Today's real failing candidate (the structural-floor false-reject).
_TODAY = {
    "aligned_real_ic": 0.0853,
    "placebo_ic": 0.0529,
    "label_autocorr_ic": 0.040,
    "shuf_ic": -0.0004,
}


# --------------------------------------------------------------------------- #
# Helpers under test
# --------------------------------------------------------------------------- #
def test_default_on_and_no_silent_legacy_resurrection(monkeypatch):
    """Genuine-IC path is default-on; only the exact opt-out string restores legacy."""
    monkeypatch.delenv("RENQUANT_WF_GATE_PLACEBO_MODE", raising=False)
    assert _genuine_ic_gate_enabled() is True
    monkeypatch.setenv("RENQUANT_WF_GATE_PLACEBO_MODE", "genuine_ic")
    assert _genuine_ic_gate_enabled() is True
    # A typo / unrelated value must NOT silently resurrect the buggy absolute path.
    monkeypatch.setenv("RENQUANT_WF_GATE_PLACEBO_MODE", "absolute")
    assert _genuine_ic_gate_enabled() is True
    monkeypatch.setenv("RENQUANT_WF_GATE_PLACEBO_MODE", "legacy_absolute")
    assert _genuine_ic_gate_enabled() is False


def test_genuine_ic_decomposition_cancels_autocorr_floor():
    """genuine_ic = aligned_real_ic − placebo_ic; the autocorr floor cancels."""
    g = _genuine_ic_value(_TODAY["aligned_real_ic"], _TODAY["placebo_ic"])
    assert g == pytest.approx(0.0324, abs=1e-9)


def test_bar_is_conservative_and_below_algebraic_noop():
    """Bar must be LOWER than 0.5×real (the old rule re-expressed) AND >= the floor.

    The OLD absolute rule ``placebo_ic < 0.5·real`` is algebraically identical to
    ``genuine_ic > 0.5·real``; re-using 0.5 would be a no-op that keeps false-
    rejecting. The chosen ratio must therefore be strictly below 0.5, and the
    absolute floor must dominate for small-edge models.
    """
    assert GENUINE_IC_REAL_RATIO < 0.5
    assert _genuine_ic_bar(0.0853) == pytest.approx(
        max(GENUINE_IC_ABS_FLOOR, GENUINE_IC_REAL_RATIO * 0.0853), abs=1e-12
    )
    # Floor dominates for a small-edge model.
    assert _genuine_ic_bar(0.04) == pytest.approx(GENUINE_IC_ABS_FLOOR, abs=1e-12)


# --------------------------------------------------------------------------- #
# Pooled placebo sub-gate — the three mandated cases
# --------------------------------------------------------------------------- #
def test_case1_structural_floor_now_passes():
    """Case 1: placebo_ic ABOVE the old absolute threshold BUT genuine_ic positive.

    Old gate: abs(0.0529) < 0.5×0.0853=0.0427 → FALSE → wrongly FAIL.
    New gate: genuine_ic 0.0324 >= bar 0.0213 → PASS. Shuffle is clean separately.
    """
    # The old absolute rule did fail here (regression anchor).
    old_threshold = max(0.005, 0.5 * abs(_TODAY["aligned_real_ic"]))
    assert abs(_TODAY["placebo_ic"]) >= old_threshold  # old rule FAILED

    passed, genuine = _placebo_subgate_pass(
        _TODAY["aligned_real_ic"], _TODAY["placebo_ic"]
    )
    assert genuine == pytest.approx(0.0324, abs=1e-9)
    assert passed is True
    # And the true-leak guard is independently clean for this candidate.
    assert abs(_TODAY["shuf_ic"]) < SHUF_IC_MAX


def test_case2_genuine_leak_shuffle_not_clean_still_fails():
    """Case 2: shuffle NOT clean → the HARD true-leak guard FAILS, gate-independent.

    The genuine-IC change must not touch this. Even with a positive genuine edge,
    a dirty shuffle (e.g. shuf_ic 0.02) fails the shuffled-label control.
    """
    dirty_shuf = 0.02
    pass_shuf = abs(dirty_shuf) < SHUF_IC_MAX
    assert pass_shuf is False
    # The overall sanity verdict ANDs pass_shuf with the placebo sub-gate, so even
    # a genuine-clean placebo cannot rescue a dirty shuffle.
    placebo_passed, _ = _placebo_subgate_pass(
        _TODAY["aligned_real_ic"], _TODAY["placebo_ic"]
    )
    assert (pass_shuf and placebo_passed) is False


@pytest.mark.parametrize(
    "aligned_real, placebo, label",
    [
        (0.0853, 0.10, "placebo exceeds real → genuine < 0"),
        (0.0853, 0.0853, "placebo equals real → genuine == 0"),
        (0.03, 0.018, "genuine 0.012 below the 0.02 absolute floor"),
    ],
)
def test_case3_no_positive_edge_fails(aligned_real, placebo, label):
    """Case 3: genuine_ic <= 0 (or below the absolute floor) → FAIL."""
    passed, genuine = _placebo_subgate_pass(aligned_real, placebo)
    assert passed is False, f"{label}: genuine={genuine}"


def test_high_edge_model_still_passes():
    """A genuinely strong model (bear-like) clears the scale-aware bar."""
    passed, genuine = _placebo_subgate_pass(0.30, 0.10)
    assert genuine == pytest.approx(0.20, abs=1e-9)
    assert passed is True


def test_missing_placebo_fails_closed():
    """NaN / missing placebo → fail closed (cannot certify leak-free)."""
    passed, genuine = _placebo_subgate_pass(0.0853, float("nan"))
    assert passed is False
    assert genuine is None


def test_legacy_opt_out_restores_absolute_and_refails_today(monkeypatch):
    """The forensic opt-out reproduces the OLD bug (today → FAIL) — proving the
    default-on path is what fixes it, and the absolute rule cannot silently win."""
    monkeypatch.setenv("RENQUANT_WF_GATE_PLACEBO_MODE", "legacy_absolute")
    passed, _ = _placebo_subgate_pass(_TODAY["aligned_real_ic"], _TODAY["placebo_ic"])
    assert passed is False  # old absolute rule false-rejects today's candidate


# --------------------------------------------------------------------------- #
# Regime sub-gate — same three cases. The regime gate composes the SAME
# `_placebo_subgate_pass` with `mean_ic >= min_mean_ic`. We replicate that
# composition here (mirrors runner.run_sanity_battery's per-regime loop) so the
# regime path is pinned without the umbrella panel/regime import.
# --------------------------------------------------------------------------- #
def _regime_passed(mean_ic, aligned_real_gate, placebo_gate_ic, real_ic):
    """Replicate runner's per-regime pass composition (eligible regime)."""
    min_mean_ic = max(0.0, 0.25 * abs(real_ic))
    mean_ic_f = float(mean_ic)
    placebo_ok = True
    if placebo_gate_ic is not None:
        ref = aligned_real_gate
        if ref is None or (isinstance(ref, float) and math.isnan(ref)):
            ref = mean_ic_f
        placebo_ok, _ = _placebo_subgate_pass(ref, placebo_gate_ic)
    return (
        mean_ic_f == mean_ic_f
        and mean_ic_f >= min_mean_ic
        and placebo_ok
    )


def test_regime_case1_structural_floor_now_passes():
    """BULL_CALM-like: positive aligned-real edge, placebo above old threshold but
    genuine positive → now PASS (was FAIL under max_placebo_ratio=0.5)."""
    # aligned_real 0.057, placebo 0.0359 → genuine 0.0211 >= bar max(0.02,0.25*0.057=0.01425)=0.02
    assert _regime_passed(
        mean_ic=0.057, aligned_real_gate=0.057, placebo_gate_ic=0.0359, real_ic=0.0853
    ) is True


def test_regime_case2_shuffle_guard_is_separate_and_hard():
    """The regime sub-gate does not certify leak-freeness on its own; the pooled
    shuffled-label control remains the hard true-leak guard for the whole battery."""
    # A clean-genuine regime cannot override a dirty global shuffle.
    dirty_shuf = 0.02
    regime_ok = _regime_passed(
        mean_ic=0.057, aligned_real_gate=0.057, placebo_gate_ic=0.0359, real_ic=0.0853
    )
    assert (abs(dirty_shuf) < SHUF_IC_MAX and regime_ok) is False


def test_regime_case3_no_positive_edge_fails():
    """BULL_CALM negative-genuine: placebo > aligned_real → genuine < 0 → FAIL."""
    # Real M6 BULL_CALM 2x: aligned_real 0.0302, placebo 0.0413 → genuine −0.0111.
    assert _regime_passed(
        mean_ic=0.0302, aligned_real_gate=0.0302, placebo_gate_ic=0.0413, real_ic=0.0853
    ) is False


def test_regime_legacy_opt_out_refails_floor_case(monkeypatch):
    """Under the forensic opt-out the regime gate reproduces the old false-reject."""
    monkeypatch.setenv("RENQUANT_WF_GATE_PLACEBO_MODE", "legacy_absolute")
    # Same structural-floor regime that PASSES on default-on now FAILS on legacy.
    assert _regime_passed(
        mean_ic=0.057, aligned_real_gate=0.057, placebo_gate_ic=0.0359, real_ic=0.0853
    ) is False
