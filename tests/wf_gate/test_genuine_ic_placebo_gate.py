"""Gate v3 (S3 placebo difference test) — genuine_ic enforced, legacy diagnostic.

The S3 change replaces the legacy absolute-ceiling placebo rule (gate v2) with
the pre-registered DIFFERENCE test:

    genuine_ic = aligned_real_ic - placebo_ic >= 0.02

The 0.02 margin was frozen 2026-07-02 in the unified 107 master plan S3 row
before any specific candidate evaluation. The 60d label's ~30d embargo overlap
inflates both terms by ~+0.04 (measured, PRs #52/#53, N=477), which cancels in
the difference.

This file pins:
  (1) The ENFORCED verdict is the S3 difference test (>= 0.02).
  (2) The legacy absolute-ceiling rule (gate v2) is stamped as diagnostic for
      evidence continuity — it does not decide pass_placebo.
  (3) The real 2026-06-23 candidate that was false-rejected by the v2 absolute
      ceiling NOW PASSES under S3 (genuine_ic +0.0324 >= 0.02).
  (4) genuine_ic's positive-aligned-real guard, the CI/reference-bar
      diagnostics, and the shuffled-label hard leak guard are unchanged.
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
    PLACEBO_CRITERION,
    PLACEBO_DIFF_MARGIN,
    PLACEBO_GENUINE_IC_MARGIN,
    SHUF_IC_MAX,
    _genuine_ic_block_bootstrap_lower,
    _genuine_ic_diag_reference_bar,
    _genuine_ic_diagnostic,
    _genuine_ic_value,
    _placebo_absolute_rule_pass,
    _placebo_difference_pass,
    _placebo_ic_threshold,
    _pooled_placebo_verdict,
)

# --------------------------------------------------------------------------- #
# Fixtures (aligned_real_ic, placebo_ic) — all at the gate shift.
# --------------------------------------------------------------------------- #
_CLEAN = {"aligned_real_ic": 0.10, "placebo_ic": 0.02}          # genuine +0.080
_LEAKY = {"aligned_real_ic": 0.085, "placebo_ic": 0.083}        # genuine +0.002
_FLOOR_ONLY = {"aligned_real_ic": 0.041, "placebo_ic": 0.040}   # genuine +0.001
_TODAY = {
    "aligned_real_ic": 0.0853,
    "placebo_ic": 0.0529,
    "label_autocorr_ic": 0.040,
    "shuf_ic": -0.0004,
}


def _enforced(fx: dict) -> bool:
    """The ENFORCED pooled placebo verdict (gate v3, S3 difference test)."""
    return bool(
        _pooled_placebo_verdict(fx["aligned_real_ic"], fx["placebo_ic"])["pass_placebo"]
    )


# --------------------------------------------------------------------------- #
# (0) Gate version is v3; margin is pinned.
# --------------------------------------------------------------------------- #
def test_gate_version_is_v3():
    """The ENFORCED rule is the S3 difference test; GATE_VERSION is 3."""
    assert GATE_VERSION == 3
    assert GATE_DIAGNOSTIC_VERSION == 3


def test_margin_is_frozen_at_0_02():
    """FROZEN per unified 107 master plan S3 row: 0.02 vs the measured ~+0.04
    shared embargo floor. Changing this constant requires a NEW design PR."""
    assert PLACEBO_DIFF_MARGIN == 0.02
    assert PLACEBO_GENUINE_IC_MARGIN == PLACEBO_DIFF_MARGIN


def test_criterion_string_is_difference_test():
    """The enforced criterion string reflects the S3 difference test."""
    verdict = _pooled_placebo_verdict(_TODAY["aligned_real_ic"], _TODAY["placebo_ic"])
    assert "genuine_ic" in verdict["placebo_criterion"]
    assert verdict["sanity_placebo_v3_gating"] is True


# --------------------------------------------------------------------------- #
# (a) CLEAN model PASSES.
# --------------------------------------------------------------------------- #
def test_clean_model_passes():
    """Real IC well above placebo + margin -> genuine_ic 0.08 >= 0.02 -> PASS."""
    assert _genuine_ic_value(**_CLEAN) == pytest.approx(0.08, abs=1e-12)
    assert _enforced(_CLEAN) is True


def test_structural_floor_false_reject_now_fixed():
    """The real 2026-06-23 candidate (genuine_ic +0.0324, shuffle-clean) was
    false-rejected by the legacy v2 absolute ceiling. Under S3 it NOW PASSES
    because genuine_ic +0.0324 >= 0.02."""
    g = _genuine_ic_value(_TODAY["aligned_real_ic"], _TODAY["placebo_ic"])
    assert g == pytest.approx(0.0324, abs=1e-9)
    assert _enforced(_TODAY) is True
    assert abs(_TODAY["shuf_ic"]) < SHUF_IC_MAX
    # The legacy v2 absolute rule (diagnostic only) still shows FAIL for this candidate.
    v = _pooled_placebo_verdict(_TODAY["aligned_real_ic"], _TODAY["placebo_ic"])
    assert v["sanity_placebo_absolute_rule_pass"] is False


# --------------------------------------------------------------------------- #
# (b) LEAKY model FAILS.
# --------------------------------------------------------------------------- #
def test_leaky_model_fails():
    """Placebo ~ real (the placebo captures the same signal) -> genuine ~ 0 -> FAIL."""
    assert _genuine_ic_value(**_LEAKY) == pytest.approx(0.002, abs=1e-12)
    assert _enforced(_LEAKY) is False


def test_fully_leaked_model_fails():
    """Placebo exactly equal to real -> genuine_ic 0 -> FAIL."""
    assert _enforced({"aligned_real_ic": 0.09, "placebo_ic": 0.09}) is False


# --------------------------------------------------------------------------- #
# (c) Embargo-floor-only model FAILS.
# --------------------------------------------------------------------------- #
def test_floor_only_model_fails():
    """real ~ placebo ~ +0.04 (the measured embargo floor), genuine ~ 0 -> FAIL:
    the shared floor cancels in the difference."""
    assert _genuine_ic_value(**_FLOOR_ONLY) == pytest.approx(0.001, abs=1e-12)
    assert _enforced(_FLOOR_ONLY) is False


def test_margin_boundary_passes_at_exactly_margin():
    """genuine_ic exactly AT the frozen margin passes (>= not >)."""
    assert _placebo_difference_pass(PLACEBO_DIFF_MARGIN) is True
    assert _placebo_difference_pass(PLACEBO_DIFF_MARGIN - 1e-9) is False


# --------------------------------------------------------------------------- #
# (d) Legacy v2 absolute rule stamped as diagnostic.
# --------------------------------------------------------------------------- #
def test_legacy_v2_absolute_rule_stamped_as_diagnostic():
    """Every pooled verdict stamps the legacy v2 absolute-ceiling verdict as
    diagnostic for evidence continuity. It does NOT decide pass_placebo."""
    v = _pooled_placebo_verdict(_TODAY["aligned_real_ic"], _TODAY["placebo_ic"])
    # S3 difference test: PASSES (genuine_ic 0.0324 >= 0.02) — this decides pass_placebo.
    assert v["pass_placebo"] is True
    assert v["sanity_placebo_v3_gating"] is True
    # Legacy v2 absolute rule (diagnostic only): FAILS (0.0529 >= 0.04265).
    assert v["sanity_placebo_absolute_rule_pass"] is False
    assert v["sanity_placebo_absolute_rule_threshold"] == pytest.approx(
        0.04265, abs=1e-9
    )


def test_difference_test_rejects_what_absolute_rule_passes():
    """A tiny-edge candidate the legacy absolute rule PASSES (placebo under the
    0.005 floor) FAILS the S3 difference test (genuine 0.009 < 0.02)."""
    v = _pooled_placebo_verdict(0.01, 0.001)  # aligned_real 0.01, placebo 0.001
    assert v["sanity_placebo_absolute_rule_pass"] is True    # legacy: PASS
    assert v["pass_placebo"] is False                         # S3: FAIL


def test_negative_aligned_real_fails_closed():
    """Negative aligned_real_ic -> genuine_ic=None -> FAIL (fail-closed).
    The legacy v2 rule had special-case passes for aligned_real==0 or negative
    real with small placebo — those no longer affect the verdict."""
    assert _pooled_placebo_verdict(0.0, -0.02)["pass_placebo"] is False
    assert _pooled_placebo_verdict(-0.10, 0.02)["pass_placebo"] is False


def test_absolute_rule_stamp_matches_verbatim_gate_v2_rule():
    """The stamped diagnostic is the verbatim gate-v2 rule, including its floor and
    the legacy aligned_real == 0 -> pass special case."""
    assert _placebo_ic_threshold(0.0853) == pytest.approx(0.04265, abs=1e-9)
    assert _placebo_ic_threshold(0.004) == pytest.approx(0.005, abs=1e-12)
    assert _placebo_absolute_rule_pass(0.10, 0.02) is True
    assert _placebo_absolute_rule_pass(0.0853, 0.0529) is False
    assert _placebo_absolute_rule_pass(0.0, 0.20) is True          # legacy special case
    assert _placebo_absolute_rule_pass(0.0853, float("nan")) is False
    v = _pooled_placebo_verdict(float("nan"), 0.05)
    assert v["sanity_placebo_absolute_rule_threshold"] is None


# --------------------------------------------------------------------------- #
# Fail-closed behavior of the difference test's genuine_ic guard.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "aligned_real, placebo, label",
    [
        (float("nan"), 0.05, "missing aligned real (NaN)"),
        (0.08, float("nan"), "missing placebo (NaN)"),
        (-0.05, -0.09, "negative real, more-negative placebo (spurious positive diff)"),
        (0.0, -0.02, "zero real (no edge to certify)"),
        (-0.10, 0.02, "negative real, positive placebo"),
    ],
)
def test_genuine_ic_guard_fails_closed(aligned_real, placebo, label):
    """Missing evidence or a non-positive aligned real IC -> genuine_ic None ->
    verdict FAILS (fail-closed)."""
    assert _genuine_ic_value(aligned_real, placebo) is None, label
    v = _pooled_placebo_verdict(aligned_real, placebo)
    assert v["pass_placebo"] is False, label


def test_difference_pass_helper_guards():
    """_placebo_difference_pass never passes None/NaN/non-numeric input."""
    assert _placebo_difference_pass(None) is False
    assert _placebo_difference_pass(float("nan")) is False
    assert _placebo_difference_pass("bogus") is False
    assert _placebo_difference_pass(PLACEBO_DIFF_MARGIN) is True      # >= at margin
    assert _placebo_difference_pass(PLACEBO_DIFF_MARGIN - 1e-9) is False


def test_shuffle_guard_unchanged():
    """The shuffled-label HARD true-leak guard is unchanged (|shuf_ic| < 0.005)."""
    assert SHUF_IC_MAX == 0.005
    assert (abs(0.02) < SHUF_IC_MAX) is False
    assert (abs(-0.0004) < SHUF_IC_MAX) is True


# --------------------------------------------------------------------------- #
# Per-regime: absolute rule kept (multiplicity concern); helpers shared.
# --------------------------------------------------------------------------- #
def test_regime_uses_same_helper_and_margin():
    """The shared genuine_ic/difference-test helpers behave identically whether
    called for the pooled leg or a per-regime reading."""
    floor_inflated_regime = _genuine_ic_value(0.057, 0.0359)
    assert floor_inflated_regime == pytest.approx(0.0211, abs=1e-9)
    assert _placebo_difference_pass(floor_inflated_regime) is True
    floor_only_regime = _genuine_ic_value(0.041, 0.040)
    assert _placebo_difference_pass(floor_only_regime) is False
    assert _placebo_difference_pass(_genuine_ic_value(-0.04, -0.07)) is False


# --------------------------------------------------------------------------- #
# Diagnostic payload (CI / reference bar).
# --------------------------------------------------------------------------- #
def test_diagnostic_payload_tagged():
    """The CI payload stamps evidence and is tagged as gate v3 S3."""
    d = _genuine_ic_diagnostic(_TODAY["aligned_real_ic"], _TODAY["placebo_ic"])
    assert d["genuine_ic"] == pytest.approx(0.0324, abs=1e-9)
    assert d["positive_aligned_real"] is True
    assert "S3" in d["tag"] or "v3" in d["tag"]
    assert "passed" not in d
    assert "pass_placebo" not in d


def test_legacy_reference_bar_still_stamped():
    """The display reference bar max(0.02, 0.25x|real|) is retained for
    historical payload comparability; it is NOT the enforced margin."""
    assert GENUINE_IC_DIAG_REAL_RATIO < 0.5
    assert _genuine_ic_diag_reference_bar(0.0853) == pytest.approx(
        max(GENUINE_IC_DIAG_ABS_FLOOR, GENUINE_IC_DIAG_REAL_RATIO * 0.0853), abs=1e-12
    )
    assert _genuine_ic_diag_reference_bar(0.04) == pytest.approx(
        GENUINE_IC_DIAG_ABS_FLOOR, abs=1e-12
    )
    assert _genuine_ic_diag_reference_bar(0.12) == pytest.approx(0.03, abs=1e-12)
    assert _placebo_difference_pass(0.025) is True


def test_diagnostic_ci_lower_bound_overlap_aware():
    """Block-bootstrap lower CI on genuine_ic respects the overlapping-label block
    and is conservative (below the point estimate)."""
    rng = np.random.default_rng(7)
    real = (0.085 + rng.normal(0, 0.05, 130)).tolist()
    plac = (0.053 + rng.normal(0, 0.05, 130)).tolist()
    paired = list(zip(real, plac))
    d = _genuine_ic_diagnostic(
        float(np.mean(real)), float(np.mean(plac)),
        paired_ics=paired, label_horizon_days=60,
    )
    assert d["ci_lower"] is not None
    assert d["ci_lower"] < d["genuine_ic"]
    assert d["ci_block_len"] == 60
    assert "moving-block bootstrap" in d["ci_method"]


def test_diagnostic_ci_block_len_tracks_horizon():
    """Block length is tied to the label horizon (overlapping-label dependence)."""
    pairs = [(0.08 + 0.001 * i, 0.05) for i in range(80)]
    ci = _genuine_ic_block_bootstrap_lower(pairs, label_horizon_days=20)
    assert ci is not None
    assert ci["block_len"] == 20
    assert ci["n_dates"] == 80


def test_diagnostic_ci_unavailable_for_tiny_sample():
    """Too few per-date pairs -> no CI (None), never a fabricated bound."""
    d = _genuine_ic_diagnostic(
        0.08, 0.05, paired_ics=[(0.08, 0.05), (0.09, 0.04)], label_horizon_days=60
    )
    assert d["ci_lower"] is None


def test_diagnostic_never_raises_on_missing_inputs():
    """Diagnostic payload must fail-soft (never raise)."""
    d = _genuine_ic_diagnostic(float("nan"), 0.05)
    assert d["genuine_ic"] is None
    assert d["ci_lower"] is None
    d2 = _genuine_ic_diagnostic(0.08, float("nan"))
    assert d2["genuine_ic"] is None


# --------------------------------------------------------------------------- #
# Positive-aligned-real guard (unchanged).
# --------------------------------------------------------------------------- #
def test_negative_aligned_real_yields_guarded_none():
    """aligned_real_ic NEGATIVE with a MORE-negative placebo would give a 'positive'
    naive difference of +0.04 — meaningless; the guard returns None (-> FAIL)."""
    naive = -0.05 - (-0.09)
    assert naive == pytest.approx(0.04, abs=1e-9)
    assert _genuine_ic_value(-0.05, -0.09) is None
    d = _genuine_ic_diagnostic(-0.05, -0.09)
    assert d["genuine_ic"] is None
    assert d["positive_aligned_real"] is False
