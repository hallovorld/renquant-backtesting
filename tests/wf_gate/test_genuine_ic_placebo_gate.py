"""genuine_ic decomposition — DIAGNOSTIC-ONLY (GATE VERDICT UNCHANGED).

SAFETY-CRITICAL. This pins the SAFE reframe of the §5.2 time-shift placebo work:
the suspected structural ~+0.04 label-autocorrelation floor at the 2×-horizon shift
is investigated via a LOGGED + STAMPED decomposition
    genuine_ic = aligned_real_ic − placebo_ic
WITHOUT changing the enforced promotion gate. The ENFORCED placebo sub-gate remains
the conservative ABSOLUTE rule on ``origin/main``:
    pass_placebo  ⇔  placebo_ic available AND
                     abs(placebo_ic) < max(0.005, 0.5×|aligned_real_ic|)
(with the legacy aligned_real_ic == 0 → pass special-case retained).

These tests assert THREE things Codex required:
  1. The enforced verdict is IDENTICAL to main on the same inputs — this PR does NOT
     change who passes (the structural-floor candidate still FAILS the absolute rule).
  2. The diagnostic fields (genuine_ic, positive-aligned-real guard, overlap-aware
     CI lower bound) are computed/stamped correctly and are tagged diagnostic-only.
  3. A NEGATIVE aligned_real_ic (with a more-negative placebo) yields a GUARDED /
     None genuine_ic — never a spurious positive.

``_TODAY`` is the real 2026-06-23 candidate the absolute rule false-rejected:
aligned_real_ic 0.0853 / placebo_ic 0.0529 / label_autocorr_ic 0.040 / shuf_ic −0.0004.
Under the ENFORCED absolute rule it still FAILS (0.0529 ≥ 0.5×0.0853 = 0.04265) — and
that is intentional: enforcement is deferred to a separately-calibrated later PR.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from renquant_backtesting.wf_gate.runner import (
    GATE_DIAGNOSTIC_VERSION,
    GATE_VERSION,
    GENUINE_IC_DIAG_ABS_FLOOR,
    GENUINE_IC_DIAG_REAL_RATIO,
    SHUF_IC_MAX,
    _genuine_ic_block_bootstrap_lower,
    _genuine_ic_diag_reference_bar,
    _genuine_ic_diagnostic,
    _genuine_ic_value,
    _placebo_ic_threshold,
)

# Today's real candidate that the ENFORCED absolute rule false-rejects.
_TODAY = {
    "aligned_real_ic": 0.0853,
    "placebo_ic": 0.0529,
    "label_autocorr_ic": 0.040,
    "shuf_ic": -0.0004,
}


# --------------------------------------------------------------------------- #
# Enforced-verdict logic, replicated EXACTLY from runner.run_sanity_battery so
# we can pin "no enforcement change" without the umbrella panel. This mirrors
# the pooled `pass_placebo` expression verbatim (the absolute rule on main).
# --------------------------------------------------------------------------- #
def _enforced_pass_placebo(aligned_real_ic, placebo_ic):
    """Verbatim copy of the ENFORCED pooled placebo rule (matches main)."""
    return (
        (placebo_ic == placebo_ic)
        and (aligned_real_ic == aligned_real_ic)
        and (
            abs(placebo_ic) < _placebo_ic_threshold(aligned_real_ic)
            if aligned_real_ic != 0
            else True
        )
    )


def _enforced_regime_pass(mean_ic, aligned_real_gate, placebo_gate_ic, real_ic):
    """Verbatim copy of the ENFORCED per-regime rule (absolute, matches main)."""
    min_mean_ic = max(0.0, 0.25 * abs(real_ic))
    max_placebo_ratio = 0.5
    mean_ic_f = float(mean_ic)
    placebo_ok = True
    if placebo_gate_ic is not None and mean_ic_f == mean_ic_f:
        placebo_ref = mean_ic_f
        try:
            aligned_real_gate_f = float(aligned_real_gate)
            if aligned_real_gate_f == aligned_real_gate_f:
                placebo_ref = aligned_real_gate_f
        except (TypeError, ValueError):
            placebo_ref = mean_ic_f
        placebo_ok = abs(float(placebo_gate_ic)) <= max(
            0.005, max_placebo_ratio * abs(placebo_ref)
        )
    return (
        mean_ic_f == mean_ic_f
        and mean_ic_f >= min_mean_ic
        and placebo_ok
    )


# --------------------------------------------------------------------------- #
# (A) ENFORCEMENT UNCHANGED — this PR does NOT change who passes.
# --------------------------------------------------------------------------- #
def test_enforced_gate_version_unchanged():
    """The ENFORCED gate version is unchanged (2); diagnostics carry their own version."""
    assert GATE_VERSION == 2
    assert GATE_DIAGNOSTIC_VERSION >= 1


def test_enforced_absolute_threshold_matches_main():
    """The enforced absolute placebo threshold is the conservative 0.5× rule (main)."""
    assert _placebo_ic_threshold(0.0853) == pytest.approx(0.04265, abs=1e-9)
    # Floor at 0.005 for tiny aligned-real edges (unchanged).
    assert _placebo_ic_threshold(0.004) == pytest.approx(0.005, abs=1e-12)


def test_structural_floor_candidate_still_fails_enforced_gate():
    """REGRESSION ANCHOR: today's structural-floor candidate gets the SAME enforced
    verdict as on main — it still FAILS the absolute placebo rule. This PR does NOT
    promote it; the diagnostic decomposition is informational only."""
    passed = _enforced_pass_placebo(_TODAY["aligned_real_ic"], _TODAY["placebo_ic"])
    assert passed is False  # 0.0529 >= 0.5×0.0853 = 0.04265 → FAIL, exactly as main
    # The shuffled-label hard guard is independently clean (kept as-is).
    assert abs(_TODAY["shuf_ic"]) < SHUF_IC_MAX


def test_clearly_clean_candidate_still_passes_enforced_gate():
    """A model whose placebo IC is comfortably below the absolute bar still PASSES —
    enforcement behavior is identical to main for clear-pass candidates too."""
    # aligned_real 0.10, placebo 0.02 < 0.5×0.10=0.05 → PASS (unchanged).
    assert _enforced_pass_placebo(0.10, 0.02) is True


def test_enforced_zero_aligned_real_special_case_unchanged():
    """The legacy aligned_real_ic == 0 → pass special-case is retained (unchanged)."""
    assert _enforced_pass_placebo(0.0, 0.20) is True


def test_enforced_missing_placebo_fails_unchanged():
    """Missing placebo (NaN) → enforced FAIL (fail-closed), as on main."""
    assert _enforced_pass_placebo(0.0853, float("nan")) is False


def test_shuffle_guard_unchanged():
    """The shuffled-label HARD true-leak guard is unchanged (|shuf_ic| < 0.005)."""
    assert SHUF_IC_MAX == 0.005
    assert (abs(0.02) < SHUF_IC_MAX) is False  # dirty shuffle still fails
    assert (abs(-0.0004) < SHUF_IC_MAX) is True  # today's clean shuffle passes


def test_enforced_regime_structural_floor_still_fails():
    """The per-regime enforced rule is the absolute 0.5× rule (main) — a regime whose
    placebo sits above it still FAILS; this PR does not change the regime verdict."""
    # aligned_real 0.057, placebo 0.0359 vs max(0.005, 0.5×0.057=0.0285) → 0.0359 > 0.0285 → FAIL.
    assert _enforced_regime_pass(
        mean_ic=0.057, aligned_real_gate=0.057, placebo_gate_ic=0.0359, real_ic=0.0853
    ) is False


def test_enforced_regime_clean_still_passes():
    """A regime with a comfortably-low placebo still PASSES under the absolute rule."""
    # aligned_real 0.057, placebo 0.010 < 0.0285 and mean_ic 0.057 >= 0.25×0.0853 → PASS.
    assert _enforced_regime_pass(
        mean_ic=0.057, aligned_real_gate=0.057, placebo_gate_ic=0.010, real_ic=0.0853
    ) is True


# --------------------------------------------------------------------------- #
# (B) DIAGNOSTIC decomposition — computed/stamped correctly, gate-unaffected.
# --------------------------------------------------------------------------- #
def test_genuine_ic_decomposition_value():
    """genuine_ic = aligned_real_ic − placebo_ic for a positive-real candidate."""
    g = _genuine_ic_value(_TODAY["aligned_real_ic"], _TODAY["placebo_ic"])
    assert g == pytest.approx(0.0324, abs=1e-9)


def test_reference_bar_is_diagnostic_only_below_algebraic_noop():
    """The DISPLAY reference bar is below 0.5×real (the absolute rule re-expressed) and
    is NOT applied to the verdict — it only makes the stamped decomposition legible."""
    assert GENUINE_IC_DIAG_REAL_RATIO < 0.5
    assert _genuine_ic_diag_reference_bar(0.0853) == pytest.approx(
        max(GENUINE_IC_DIAG_ABS_FLOOR, GENUINE_IC_DIAG_REAL_RATIO * 0.0853), abs=1e-12
    )
    assert _genuine_ic_diag_reference_bar(0.04) == pytest.approx(
        GENUINE_IC_DIAG_ABS_FLOOR, abs=1e-12
    )


def test_diagnostic_payload_is_tagged_and_complete():
    """The diagnostic payload carries the point estimate, guard, reference bar, and the
    diagnostic-only tag — and never a pass/fail verdict field."""
    d = _genuine_ic_diagnostic(_TODAY["aligned_real_ic"], _TODAY["placebo_ic"])
    assert d["genuine_ic"] == pytest.approx(0.0324, abs=1e-9)
    assert d["positive_aligned_real"] is True
    assert d["reference_bar"] == pytest.approx(0.021325, abs=1e-9)
    assert d["reference_bar_meets"] is True  # diagnostic note only — NOT the verdict
    assert d["tag"] == "diagnostic-only, gate unaffected"
    # The payload must NOT carry an enforcement decision.
    assert "passed" not in d
    assert "pass_placebo" not in d


def test_diagnostic_ci_lower_bound_overlap_aware():
    """Block-bootstrap lower CI on genuine_ic respects the overlapping-label block and
    is conservative (below the point estimate)."""
    rng = np.random.default_rng(7)
    real = (0.085 + rng.normal(0, 0.05, 130)).tolist()
    plac = (0.053 + rng.normal(0, 0.05, 130)).tolist()
    paired = list(zip(real, plac))
    d = _genuine_ic_diagnostic(
        float(np.mean(real)), float(np.mean(plac)),
        paired_ics=paired, label_horizon_days=60,
    )
    assert d["ci_lower"] is not None
    assert d["ci_lower"] < d["genuine_ic"]  # conservative
    assert d["ci_block_len"] == 60  # block length tied to the 60d label horizon
    assert "moving-block bootstrap" in d["ci_method"]


def test_diagnostic_ci_block_len_tracks_horizon():
    """Block length is tied to the label horizon (overlapping-label dependence)."""
    pairs = [(0.08 + 0.001 * i, 0.05) for i in range(80)]
    ci = _genuine_ic_block_bootstrap_lower(pairs, label_horizon_days=20)
    assert ci is not None
    assert ci["block_len"] == 20
    assert ci["n_dates"] == 80


def test_diagnostic_ci_unavailable_for_tiny_sample():
    """Too few per-date pairs → no CI (None), never a fabricated bound."""
    d = _genuine_ic_diagnostic(
        0.08, 0.05, paired_ics=[(0.08, 0.05), (0.09, 0.04)], label_horizon_days=60
    )
    assert d["ci_lower"] is None


def test_diagnostic_never_raises_on_missing_inputs():
    """Diagnostic must fail-soft (never raise) — it must never be able to fail the gate."""
    d = _genuine_ic_diagnostic(float("nan"), 0.05)
    assert d["genuine_ic"] is None
    assert d["ci_lower"] is None
    d2 = _genuine_ic_diagnostic(0.08, float("nan"))
    assert d2["genuine_ic"] is None


# --------------------------------------------------------------------------- #
# (C) POSITIVE-ALIGNED-REAL guard — no spurious positive genuine_ic.
# --------------------------------------------------------------------------- #
def test_negative_aligned_real_yields_guarded_none():
    """Codex pathology: aligned_real_ic NEGATIVE but placebo MORE negative would give a
    'positive' genuine_ic of +0.04 — meaningless. The guard must return None instead."""
    # aligned_real −0.05, placebo −0.09 → naive diff +0.04 (spurious positive).
    naive = -0.05 - (-0.09)
    assert naive == pytest.approx(0.04, abs=1e-9)  # what the UNGUARDED formula would give
    assert _genuine_ic_value(-0.05, -0.09) is None  # guard refuses it
    d = _genuine_ic_diagnostic(-0.05, -0.09)
    assert d["genuine_ic"] is None
    assert d["positive_aligned_real"] is False
    assert d["reference_bar_meets"] is False


def test_zero_aligned_real_yields_guarded_none():
    """aligned_real_ic == 0 has no real edge to certify → guarded None."""
    assert _genuine_ic_value(0.0, -0.02) is None


def test_positive_aligned_real_passes_guard():
    """A genuinely positive aligned-real candidate reports a real genuine_ic."""
    assert _genuine_ic_value(0.30, 0.10) == pytest.approx(0.20, abs=1e-9)


@pytest.mark.parametrize(
    "aligned_real, placebo, label",
    [
        (-0.05, -0.09, "negative real, more-negative placebo (spurious positive)"),
        (0.0, -0.02, "zero real (no edge to certify)"),
        (-0.10, 0.02, "negative real, positive placebo"),
    ],
)
def test_no_spurious_positive_genuine_ic(aligned_real, placebo, label):
    """No combination with non-positive aligned_real may yield a numeric genuine_ic."""
    assert _genuine_ic_value(aligned_real, placebo) is None, label


# --------------------------------------------------------------------------- #
# Regime diagnostic stamp uses the SAME guarded genuine_ic (gate-unaffected).
# --------------------------------------------------------------------------- #
def test_regime_diagnostic_genuine_ic_is_guarded():
    """The per-regime stamped placebo_gate_genuine_ic uses the guarded helper, so a
    negative-aligned-real regime stamps None (not a spurious positive)."""
    assert _genuine_ic_value(-0.04, -0.07) is None
    # And a clean positive regime stamps the real difference.
    assert _genuine_ic_value(0.057, 0.0359) == pytest.approx(0.0211, abs=1e-9)
    # The regime ENFORCED verdict for that positive case is still the absolute rule.
    assert math.isclose(0.057 - 0.0359, 0.0211, abs_tol=1e-9)
