"""Gate v3 (S3) — the placebo leg gates on the pre-registered DIFFERENCE test.

SAFETY-CRITICAL. This pins the ENFORCED §5.2 placebo criterion switched in S3 of
the unified 107 master plan (design lineage: merged #210 freshness-governance
Fix-3, plan S1–S3 row):

    pass_placebo  ⇔  genuine_ic = aligned_real_ic − placebo_ic > 0.02

with the margin FROZEN 2026-07-02 (``PLACEBO_CRITERION`` self-documents every
verdict). WHY: the daily fwd_60d label carries a measured ~+0.04
embargo-leakage / label-autocorrelation floor SHARED by aligned_real_ic and
placebo_ic, so the old absolute ceiling (placebo_ic < 0.5×|aligned_real_ic|,
gate v2) was structurally unsatisfiable for leak-free long-horizon candidates.
The shared floor cancels in the difference.

These tests pin FOUR things:
  (a) a synthetic CLEAN model (real IC well above placebo + margin) PASSES;
  (b) a LEAKY model (placebo ≈ real) FAILS;
  (c) a model sitting exactly at the embargo floor (real ≈ placebo ≈ +0.04,
      genuine ≈ 0) FAILS — the floor alone can no longer FAIL an otherwise-good
      model (the real 2026-06-23 false-reject now passes) NOR PASS a no-edge one;
  (d) the OLD absolute-ceiling criterion's would-be verdict is still STAMPED as
      a diagnostic (evidence fields are never deleted) — and it no longer
      decides in EITHER direction.

The margin is FROZEN: any change to PLACEBO_GENUINE_IC_MARGIN / the criterion
string requires a new design PR, and must consciously rewrite the freeze tests
below.
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
    """The REAL enforced pooled placebo verdict (production code path)."""
    return bool(
        _pooled_placebo_verdict(fx["aligned_real_ic"], fx["placebo_ic"])["pass_placebo"]
    )


# --------------------------------------------------------------------------- #
# (0) FROZEN criterion — margin and self-documentation string are pinned.
# --------------------------------------------------------------------------- #
def test_gate_version_bumped_for_criterion_switch():
    """The ENFORCED rule changed (absolute ceiling → difference test) ⇒ gate v3."""
    assert GATE_VERSION == 3
    assert GATE_DIAGNOSTIC_VERSION == 2


def test_margin_is_frozen_at_0_02():
    """FROZEN 2026-07-02 (unified 107 master plan S1–S3 row: 0.02 vs the measured
    ~+0.04 shared embargo floor). Changing this constant requires a NEW design PR —
    do not 'tune' it here."""
    assert PLACEBO_GENUINE_IC_MARGIN == 0.02


def test_criterion_string_self_documents_the_freeze():
    """Every stamped verdict carries this exact string (placebo_criterion)."""
    assert PLACEBO_CRITERION == "genuine_ic>0.02 (frozen 2026-07-02)"
    verdict = _pooled_placebo_verdict(_TODAY["aligned_real_ic"], _TODAY["placebo_ic"])
    assert verdict["placebo_criterion"] == PLACEBO_CRITERION
    assert verdict["sanity_placebo_genuine_ic_margin"] == PLACEBO_GENUINE_IC_MARGIN


# --------------------------------------------------------------------------- #
# (a) CLEAN model PASSES.
# --------------------------------------------------------------------------- #
def test_clean_model_passes():
    """Real IC well above placebo + margin → genuine_ic 0.08 > 0.02 → PASS."""
    assert _genuine_ic_value(**_CLEAN) == pytest.approx(0.08, abs=1e-12)
    assert _enforced(_CLEAN) is True


def test_structural_floor_false_reject_is_repaired():
    """The floor alone can no longer FAIL an otherwise-good model: the real
    2026-06-23 candidate (genuine_ic +0.0324, shuffle-clean) now PASSES, where the
    gate-v2 absolute ceiling false-rejected it (0.0529 ≥ 0.5×0.0853 = 0.04265)."""
    g = _genuine_ic_value(_TODAY["aligned_real_ic"], _TODAY["placebo_ic"])
    assert g == pytest.approx(0.0324, abs=1e-9)
    assert _enforced(_TODAY) is True
    # Its shuffled-label control was independently clean (hard guard unchanged).
    assert abs(_TODAY["shuf_ic"]) < SHUF_IC_MAX


# --------------------------------------------------------------------------- #
# (b) LEAKY model FAILS.
# --------------------------------------------------------------------------- #
def test_leaky_model_fails():
    """Placebo ≈ real (the placebo captures the same signal) → genuine ≈ 0 → FAIL."""
    assert _genuine_ic_value(**_LEAKY) == pytest.approx(0.002, abs=1e-12)
    assert _enforced(_LEAKY) is False


def test_fully_leaked_model_fails():
    """Placebo exactly equal to real → genuine_ic 0 → FAIL."""
    assert _enforced({"aligned_real_ic": 0.09, "placebo_ic": 0.09}) is False


# --------------------------------------------------------------------------- #
# (c) Embargo-floor-only model FAILS — the floor can't PASS a no-edge model.
# --------------------------------------------------------------------------- #
def test_floor_only_model_fails():
    """real ≈ placebo ≈ +0.04 (the measured embargo floor), genuine ≈ 0 → FAIL:
    the shared floor cancels in the difference, so it cannot manufacture a pass
    for a model whose only 'signal' is the structural floor."""
    assert _genuine_ic_value(**_FLOOR_ONLY) == pytest.approx(0.001, abs=1e-12)
    assert _enforced(_FLOOR_ONLY) is False


def test_margin_boundary_is_strict():
    """genuine_ic exactly AT the frozen margin does not pass (strict >); the
    exact-boundary semantics are pinned on the helper (no float-subtraction noise),
    and the pooled path is checked just below / just above the margin."""
    assert _placebo_difference_pass(PLACEBO_GENUINE_IC_MARGIN) is False   # strict >
    assert _placebo_difference_pass(PLACEBO_GENUINE_IC_MARGIN + 1e-9) is True
    just_below = {"aligned_real_ic": 0.06, "placebo_ic": 0.045}   # genuine 0.015
    assert _enforced(just_below) is False
    just_above = {"aligned_real_ic": 0.06, "placebo_ic": 0.035}   # genuine 0.025
    assert _enforced(just_above) is True


# --------------------------------------------------------------------------- #
# (d) The OLD absolute-ceiling verdict is still STAMPED — as a DIAGNOSTIC.
# --------------------------------------------------------------------------- #
def test_old_criterion_verdict_still_stamped_as_diagnostic():
    """Every pooled verdict stamps the gate-v2 absolute-ceiling would-be verdict and
    threshold (evidence continuity) without letting them decide."""
    v = _pooled_placebo_verdict(_TODAY["aligned_real_ic"], _TODAY["placebo_ic"])
    # Old rule would-FAIL (0.0529 ≥ 0.04265) — stamped…
    assert v["sanity_placebo_absolute_rule_pass"] is False
    assert v["sanity_placebo_absolute_rule_threshold"] == pytest.approx(
        0.04265, abs=1e-9
    )
    # …but the ENFORCED verdict is the difference test → PASS.
    assert v["pass_placebo"] is True


def test_old_criterion_no_longer_decides_in_either_direction():
    """The switch is real in BOTH directions: a tiny-edge candidate the old rule
    would have PASSED (placebo under the 0.005 floor) now FAILS the difference
    test (genuine 0.009 < 0.02) — the new criterion is not 'old rule plus'."""
    v = _pooled_placebo_verdict(0.01, 0.001)  # aligned_real 0.01, placebo 0.001
    assert v["sanity_placebo_absolute_rule_pass"] is True   # old rule: would-PASS
    assert v["pass_placebo"] is False                        # enforced: FAIL


def test_absolute_rule_stamp_matches_verbatim_gate_v2_rule():
    """The stamped diagnostic is the verbatim gate-v2 rule, including its floor and
    the legacy aligned_real == 0 → pass special case."""
    assert _placebo_ic_threshold(0.0853) == pytest.approx(0.04265, abs=1e-9)
    assert _placebo_ic_threshold(0.004) == pytest.approx(0.005, abs=1e-12)
    assert _placebo_absolute_rule_pass(0.10, 0.02) is True
    assert _placebo_absolute_rule_pass(0.0853, 0.0529) is False
    assert _placebo_absolute_rule_pass(0.0, 0.20) is True          # legacy special case
    assert _placebo_absolute_rule_pass(0.0853, float("nan")) is False
    # NaN aligned_real → no threshold to stamp (None), never a fabricated number.
    v = _pooled_placebo_verdict(float("nan"), 0.05)
    assert v["sanity_placebo_absolute_rule_threshold"] is None


# --------------------------------------------------------------------------- #
# Fail-closed behavior of the ENFORCED difference test.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "aligned_real, placebo, label",
    [
        (float("nan"), 0.05, "missing aligned real (NaN)"),
        (0.08, float("nan"), "missing placebo (NaN)"),
        (-0.05, -0.09, "negative real, more-negative placebo (spurious positive diff)"),
        (0.0, -0.02, "zero real (no edge to certify; old special-case now FAILS)"),
        (-0.10, 0.02, "negative real, positive placebo"),
    ],
)
def test_enforced_gate_fails_closed(aligned_real, placebo, label):
    """Missing evidence or a non-positive aligned real IC → genuine_ic None → FAIL."""
    assert _genuine_ic_value(aligned_real, placebo) is None, label
    assert _pooled_placebo_verdict(aligned_real, placebo)["pass_placebo"] is False, label


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
# Per-regime leg uses the SAME frozen criterion (shared helper).
# --------------------------------------------------------------------------- #
def test_regime_leg_same_frozen_criterion():
    """The per-regime placebo leg gates on the same difference test: the real
    structural-floor regime (aligned 0.057 / placebo 0.0359, genuine +0.0211) now
    passes the placebo leg; a floor-only regime (genuine ≈ 0) fails."""
    floor_inflated_regime = _genuine_ic_value(0.057, 0.0359)
    assert floor_inflated_regime == pytest.approx(0.0211, abs=1e-9)
    assert _placebo_difference_pass(floor_inflated_regime) is True
    floor_only_regime = _genuine_ic_value(0.041, 0.040)
    assert _placebo_difference_pass(floor_only_regime) is False
    # Negative-aligned-real regime: guarded None → fail-closed.
    assert _placebo_difference_pass(_genuine_ic_value(-0.04, -0.07)) is False


# --------------------------------------------------------------------------- #
# Diagnostic payload (CI / reference bar) — still stamped, still fail-soft.
# --------------------------------------------------------------------------- #
def test_diagnostic_payload_tagged_and_carries_no_verdict():
    """The CI payload stamps evidence, is tagged, and never carries pass/fail —
    the enforced point estimate is computed independently by _pooled_placebo_verdict."""
    d = _genuine_ic_diagnostic(_TODAY["aligned_real_ic"], _TODAY["placebo_ic"])
    assert d["genuine_ic"] == pytest.approx(0.0324, abs=1e-9)
    assert d["positive_aligned_real"] is True
    assert d["tag"] == (
        "genuine_ic point estimate ENFORCED (gate v3); "
        "CI/reference-bar fields diagnostic-only"
    )
    assert "passed" not in d
    assert "pass_placebo" not in d


def test_legacy_reference_bar_still_stamped_not_enforced():
    """The pre-freeze display reference bar max(0.02, 0.25×|real|) is retained for
    historical payload comparability; it is NOT the enforced margin (which is the
    flat frozen 0.02)."""
    assert GENUINE_IC_DIAG_REAL_RATIO < 0.5
    assert _genuine_ic_diag_reference_bar(0.0853) == pytest.approx(
        max(GENUINE_IC_DIAG_ABS_FLOOR, GENUINE_IC_DIAG_REAL_RATIO * 0.0853), abs=1e-12
    )
    assert _genuine_ic_diag_reference_bar(0.04) == pytest.approx(
        GENUINE_IC_DIAG_ABS_FLOOR, abs=1e-12
    )
    # Divergence proof: at real=0.12 the display bar (0.03) exceeds the enforced
    # margin (0.02) — a genuine_ic of 0.025 PASSES the gate but sits under the bar.
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
    """Diagnostic payload must fail-soft (never raise) — an exception in it can
    never flip the verdict (fail-closed enforcement is computed independently)."""
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
