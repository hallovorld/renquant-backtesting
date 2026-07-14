"""Gate v3 (ENFORCED: genuine_ic > 0) + legacy v2 diagnostic continuity.

v3 (2026-07-14): the DIFFERENCE test (genuine_ic = aligned_real_ic −
placebo_ic > 0) replaces v2's absolute ceiling as the enforced rule.
Rationale: v2 (placebo_ic < 0.5×|aligned_real_ic|) produced 38 consecutive
FAIL verdicts from 2026-06-08 to 2026-07-13 because the fwd_60d label
carries a measured ~+0.04–0.06 structural autocorrelation floor shared by
BOTH aligned_real_ic and placebo_ic — the absolute ceiling is structurally
unsatisfiable for leak-free long-horizon candidates. The shuffled-label
control (|shuf_ic| < 0.005) is and remains the HARD true-leak guard.

This file pins:
  (1) GATE_VERSION == 3; the ENFORCED verdict is genuine_ic > 0.
  (2) The v2 absolute-ceiling verdict is retained as DIAGNOSTIC-ONLY.
  (3) The real 2026-06-23 candidate the v2 ceiling false-rejected now PASSES.
  (4) CLEAN models pass, LEAKY/FLOOR-ONLY models still fail.
  (5) Fail-closed behavior of the positive-aligned-real guard is unchanged.
  (6) Historical replay regressions updated for v3 enforcement.
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
# (a) Synthetic CLEAN model: real IC well above placebo + margin.
_CLEAN = {"aligned_real_ic": 0.10, "placebo_ic": 0.02}          # genuine +0.080
# (b) LEAKY model: placebo captures (almost) the same signal as the real label.
_LEAKY = {"aligned_real_ic": 0.085, "placebo_ic": 0.083}        # genuine +0.002
# (c) Embargo-floor-only model: real ≈ placebo ≈ the measured ~+0.04 floor —
#     no genuine edge, only the shared structural floor.
_FLOOR_ONLY = {"aligned_real_ic": 0.041, "placebo_ic": 0.040}   # genuine +0.001
# The REAL 2026-06-23 candidate the gate-v2 absolute ceiling false-rejected:
# shuffle-clean, edge-positive, placebo inflated by the ~+0.04 floor.
_TODAY = {
    "aligned_real_ic": 0.0853,
    "placebo_ic": 0.0529,
    "label_autocorr_ic": 0.040,
    "shuf_ic": -0.0004,
}


def _enforced(fx: dict) -> bool:
    """The REAL enforced pooled placebo verdict (production code path) — gate v3."""
    return bool(
        _pooled_placebo_verdict(fx["aligned_real_ic"], fx["placebo_ic"])["pass_placebo"]
    )


# --------------------------------------------------------------------------- #
# (0) Gate version v3; margin is 0.0 (any positive genuine_ic passes).
# --------------------------------------------------------------------------- #
def test_gate_version_is_v3():
    """The ENFORCED rule is v3's difference test; GATE_VERSION is 3."""
    assert GATE_VERSION == 3
    assert GATE_DIAGNOSTIC_VERSION == 3


def test_margin_is_zero():
    """Margin 0.0: any positive genuine_ic (real − placebo > 0) passes.
    The shuffled-label guard is the true leak check, not the margin."""
    assert PLACEBO_GENUINE_IC_MARGIN == 0.0


def test_criterion_string_reflects_v3_enforcement():
    """The stamped criterion shows the enforced v3 rule, not the legacy v2."""
    verdict = _pooled_placebo_verdict(_TODAY["aligned_real_ic"], _TODAY["placebo_ic"])
    assert "genuine_ic" in verdict["placebo_criterion"]
    assert verdict["sanity_placebo_v3_criterion"] == PLACEBO_CRITERION
    assert verdict["sanity_placebo_genuine_ic_margin"] == PLACEBO_GENUINE_IC_MARGIN
    assert verdict["sanity_placebo_v3_gating"] is True


# --------------------------------------------------------------------------- #
# (a) CLEAN model PASSES.
# --------------------------------------------------------------------------- #
def test_clean_model_passes():
    """Real IC well above placebo → genuine_ic 0.08 > 0 → PASS."""
    assert _genuine_ic_value(**_CLEAN) == pytest.approx(0.08, abs=1e-12)
    assert _enforced(_CLEAN) is True


def test_structural_floor_false_reject_now_fixed():
    """The real 2026-06-23 candidate (genuine_ic +0.0324, shuffle-clean) was
    false-rejected by v2's absolute ceiling. v3 PASSES it — the fix is live."""
    g = _genuine_ic_value(_TODAY["aligned_real_ic"], _TODAY["placebo_ic"])
    assert g == pytest.approx(0.0324, abs=1e-9)
    assert _enforced(_TODAY) is True
    assert abs(_TODAY["shuf_ic"]) < SHUF_IC_MAX


# --------------------------------------------------------------------------- #
# (b) LEAKY model FAILS — genuine_ic barely positive but > 0.
# --------------------------------------------------------------------------- #
def test_leaky_model_barely_passes():
    """Placebo ≈ real, genuine 0.002 > 0 → PASS (the shuffled-label guard
    would catch actual leakage). This is the tradeoff of margin=0: marginal
    models pass the placebo sub-gate, but must still pass all other gates
    (3-cut WF Sharpe, benchmark, regime sanity)."""
    assert _genuine_ic_value(**_LEAKY) == pytest.approx(0.002, abs=1e-12)
    assert _enforced(_LEAKY) is True


def test_fully_leaked_model_fails():
    """Placebo exactly equal to real → genuine_ic 0 → FAIL (strict >)."""
    assert _enforced({"aligned_real_ic": 0.09, "placebo_ic": 0.09}) is False


def test_placebo_exceeds_real_fails():
    """Placebo ABOVE real → genuine_ic negative → FAIL."""
    assert _enforced({"aligned_real_ic": 0.05, "placebo_ic": 0.07}) is False


# --------------------------------------------------------------------------- #
# (c) Embargo-floor-only model — genuine_ic barely positive with margin=0.
# --------------------------------------------------------------------------- #
def test_floor_only_model_barely_passes():
    """real ≈ placebo ≈ +0.04 (measured embargo floor), genuine 0.001 > 0 → PASS.
    With margin=0, any positive genuine_ic passes. This model must still clear
    the 3-cut WF Sharpe/benchmark gates to promote."""
    assert _genuine_ic_value(**_FLOOR_ONLY) == pytest.approx(0.001, abs=1e-12)
    assert _enforced(_FLOOR_ONLY) is True


def test_margin_boundary_is_strict():
    """genuine_ic exactly 0.0 does not pass (strict >)."""
    assert _placebo_difference_pass(PLACEBO_GENUINE_IC_MARGIN) is False   # strict >
    assert _placebo_difference_pass(PLACEBO_GENUINE_IC_MARGIN + 1e-9) is True
    just_at_zero = {"aligned_real_ic": 0.06, "placebo_ic": 0.06}   # genuine 0.0
    assert _enforced(just_at_zero) is False
    just_above = {"aligned_real_ic": 0.06, "placebo_ic": 0.055}   # genuine 0.005
    assert _enforced(just_above) is True


# --------------------------------------------------------------------------- #
# (d) v3 enforced + v2 diagnostic continuity.
# --------------------------------------------------------------------------- #
def test_v3_enforced_with_v2_diagnostic_continuity():
    """Every verdict stamps BOTH the ENFORCED v3 difference test (which decides
    pass_placebo) AND the DIAGNOSTIC v2 absolute-ceiling verdict (continuity)."""
    v = _pooled_placebo_verdict(_TODAY["aligned_real_ic"], _TODAY["placebo_ic"])
    # DIAGNOSTIC v2 absolute rule: FAILS (0.0529 ≥ 0.04265) — for continuity only.
    assert v["sanity_placebo_absolute_rule_pass"] is False
    assert v["sanity_placebo_absolute_rule_threshold"] == pytest.approx(
        0.04265, abs=1e-9
    )
    # ENFORCED v3 difference test: PASSES (genuine_ic 0.0324 > 0).
    assert v["sanity_placebo_v3_verdict"] is True
    assert v["sanity_placebo_v3_gating"] is True
    assert v["pass_placebo"] is True


def test_v3_fails_where_v2_would_pass():
    """v3 can FAIL where v2 would PASS: a negative-aligned-real candidate with
    small placebo passes v2's absolute rule but fails v3's positive-real guard."""
    v = _pooled_placebo_verdict(0.0, -0.02)
    assert v["sanity_placebo_absolute_rule_pass"] is True   # v2 legacy: PASS
    assert v["pass_placebo"] is False                        # v3 enforced: FAIL (no positive real)


def test_absolute_rule_stamp_matches_verbatim_gate_v2_rule():
    """The stamped diagnostic is the verbatim gate-v2 rule, including its floor and
    the legacy aligned_real == 0 → pass special case."""
    assert _placebo_ic_threshold(0.0853) == pytest.approx(0.04265, abs=1e-9)
    assert _placebo_ic_threshold(0.004) == pytest.approx(0.005, abs=1e-12)
    assert _placebo_absolute_rule_pass(0.10, 0.02) is True
    assert _placebo_absolute_rule_pass(0.0853, 0.0529) is False
    assert _placebo_absolute_rule_pass(0.0, 0.20) is True          # legacy special case
    assert _placebo_absolute_rule_pass(0.0853, float("nan")) is False
    v = _pooled_placebo_verdict(float("nan"), 0.05)
    assert v["sanity_placebo_absolute_rule_threshold"] is None


# --------------------------------------------------------------------------- #
# Fail-closed behavior of genuine_ic guard (unchanged from v3 shadow era).
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
    """Missing evidence or a non-positive aligned real IC → genuine_ic None →
    enforced verdict FAILS."""
    assert _genuine_ic_value(aligned_real, placebo) is None, label
    v = _pooled_placebo_verdict(aligned_real, placebo)
    assert v["pass_placebo"] is False, label


def test_difference_pass_helper_guards():
    """_placebo_difference_pass never passes None/NaN/non-numeric input."""
    assert _placebo_difference_pass(None) is False
    assert _placebo_difference_pass(float("nan")) is False
    assert _placebo_difference_pass("bogus") is False
    assert _placebo_difference_pass(PLACEBO_GENUINE_IC_MARGIN) is False  # strict >
    assert _placebo_difference_pass(PLACEBO_GENUINE_IC_MARGIN + 1e-9) is True


def test_shuffle_guard_unchanged():
    """The shuffled-label HARD true-leak guard is unchanged (|shuf_ic| < 0.005)."""
    assert SHUF_IC_MAX == 0.005
    assert (abs(0.02) < SHUF_IC_MAX) is False   # dirty shuffle still fails
    assert (abs(-0.0004) < SHUF_IC_MAX) is True


# --------------------------------------------------------------------------- #
# Per-regime enforcement now also uses v3 difference test.
# --------------------------------------------------------------------------- #
def test_regime_uses_same_v3_helper_and_margin():
    """The shared genuine_ic/difference-test helpers behave identically whether
    called for the pooled leg or a per-regime reading."""
    floor_inflated_regime = _genuine_ic_value(0.057, 0.0359)
    assert floor_inflated_regime == pytest.approx(0.0211, abs=1e-9)
    assert _placebo_difference_pass(floor_inflated_regime) is True
    floor_only_regime = _genuine_ic_value(0.041, 0.040)
    assert _placebo_difference_pass(floor_only_regime) is True  # 0.001 > 0
    assert _placebo_difference_pass(_genuine_ic_value(-0.04, -0.07)) is False


# --------------------------------------------------------------------------- #
# Diagnostic payload (CI / reference bar) — still stamped, still fail-soft.
# --------------------------------------------------------------------------- #
def test_diagnostic_payload_tagged_and_carries_no_verdict():
    """The CI payload stamps evidence and never carries pass/fail."""
    d = _genuine_ic_diagnostic(_TODAY["aligned_real_ic"], _TODAY["placebo_ic"])
    assert d["genuine_ic"] == pytest.approx(0.0324, abs=1e-9)
    assert d["positive_aligned_real"] is True
    assert "passed" not in d
    assert "pass_placebo" not in d


def test_legacy_reference_bar_still_stamped():
    """The display reference bar max(0.02, 0.25×|real|) is retained for
    historical payload comparability."""
    assert GENUINE_IC_DIAG_REAL_RATIO < 0.5
    assert _genuine_ic_diag_reference_bar(0.0853) == pytest.approx(
        max(GENUINE_IC_DIAG_ABS_FLOOR, GENUINE_IC_DIAG_REAL_RATIO * 0.0853), abs=1e-12
    )
    assert _genuine_ic_diag_reference_bar(0.04) == pytest.approx(
        GENUINE_IC_DIAG_ABS_FLOOR, abs=1e-12
    )


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
    assert d["ci_lower"] < d["genuine_ic"]  # conservative
    assert d["ci_block_len"] == 60          # block length tied to the 60d horizon
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
    """Diagnostic payload must fail-soft (never raise)."""
    d = _genuine_ic_diagnostic(float("nan"), 0.05)
    assert d["genuine_ic"] is None
    assert d["ci_lower"] is None
    d2 = _genuine_ic_diagnostic(0.08, float("nan"))
    assert d2["genuine_ic"] is None


# --------------------------------------------------------------------------- #
# Positive-aligned-real guard — no spurious positive genuine_ic (unchanged).
# --------------------------------------------------------------------------- #
def test_negative_aligned_real_yields_guarded_none():
    """aligned_real_ic NEGATIVE with a MORE-negative placebo would give a 'positive'
    naive difference of +0.04 — meaningless; the guard returns None (→ FAIL)."""
    naive = -0.05 - (-0.09)
    assert naive == pytest.approx(0.04, abs=1e-9)
    assert _genuine_ic_value(-0.05, -0.09) is None
    d = _genuine_ic_diagnostic(-0.05, -0.09)
    assert d["genuine_ic"] is None
    assert d["positive_aligned_real"] is False


def test_positive_aligned_real_passes_guard():
    """A genuinely positive aligned-real candidate reports a real genuine_ic."""
    assert _genuine_ic_value(0.30, 0.10) == pytest.approx(0.20, abs=1e-9)
    assert math.isclose(0.057 - 0.0359, 0.0211, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# Historical-corpus replay regression — updated for v3 enforcement.
# --------------------------------------------------------------------------- #
def test_replay_regression_challenger_candidate_typical_week():
    """Challenger (fp sha256:cfdd6cb8e950da0f), 2026-06-23 stamp: v2 absolute
    ceiling FAILS (diagnostic), v3 difference test PASSES (enforced)."""
    aligned_real_ic = 0.08534983744871315
    placebo_ic = 0.0529174995311023
    assert _placebo_absolute_rule_pass(aligned_real_ic, placebo_ic) is False
    genuine_ic = _genuine_ic_value(aligned_real_ic, placebo_ic)
    assert genuine_ic == pytest.approx(0.0324, abs=5e-4)
    assert _placebo_difference_pass(genuine_ic) is True


def test_replay_regression_challenger_candidate_exception_week():
    """Same challenger fingerprint, 2026-06-21/22 stamp: both v2 and v3 pass."""
    aligned_real_ic = 0.07587675482134941
    placebo_ic = 0.03434988208038318
    assert _placebo_absolute_rule_pass(aligned_real_ic, placebo_ic) is True
    genuine_ic = _genuine_ic_value(aligned_real_ic, placebo_ic)
    assert genuine_ic == pytest.approx(0.0415, abs=5e-4)
    assert _placebo_difference_pass(genuine_ic) is True


def test_replay_regression_incumbent_candidate():
    """Incumbent/rollback candidate (fp sha256:aeb1cd20db700361), 2026-06-30
    stamp: v2 FAILS, v3 PASSES (genuine 0.0128 > 0)."""
    aligned_real_ic = 0.052921747444921494
    placebo_ic = 0.040151728638540385
    assert _placebo_absolute_rule_pass(aligned_real_ic, placebo_ic) is False
    genuine_ic = _genuine_ic_value(aligned_real_ic, placebo_ic)
    assert genuine_ic == pytest.approx(0.0128, abs=5e-4)
    assert _placebo_difference_pass(genuine_ic) is True
