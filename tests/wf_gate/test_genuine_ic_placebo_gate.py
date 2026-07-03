"""Gate v2 (UNCHANGED, enforced) + gate v3 candidate (SHADOW-ONLY, S3).

SAFETY-CRITICAL. Round 2 of the S3 rollout (Codex 2026-07-02 review): the
first attempt made the pre-registered DIFFERENCE test

    genuine_ic = aligned_real_ic − placebo_ic > 0.02

the ENFORCED §5.2 placebo criterion, but its 0.02 margin was selected while
inspecting the specific candidate it flips (post-outcome gate calibration),
and its own overlap-aware CI stayed diagnostic-only while the noisy point
estimate alone would decide real capital. Both are blockers for a gate that
authorizes capital deployment.

This file now pins:
  (1) The ENFORCED verdict is v2's absolute ceiling
      (placebo_ic < 0.5×|aligned_real_ic|), UNCHANGED from before this PR.
  (2) The v3 DIFFERENCE test (genuine_ic > 0.02, margin still frozen at 0.02
      for the SHADOW evaluation) is computed and STAMPED on every verdict as
      SHADOW-ONLY evidence (``sanity_placebo_v3_shadow_verdict``,
      ``sanity_placebo_v3_gating`` == False) — it does not decide anything
      until a historical-corpus replay and a prospective held-out run
      validate it (see doc/research/2026-07-02-wf-gate-v3-shadow-eval.md).
  (3) The real 2026-06-23 candidate the gate-v2 absolute ceiling
      false-rejected: v3's shadow verdict shows the fix WOULD repair that
      false reject, but the currently ENFORCED verdict is still FAIL — this
      is exactly why v3 needs shadow validation before promotion, not an
      inconsistency.
  (4) genuine_ic's positive-aligned-real guard, the CI/reference-bar shadow
      payload, and the shuffled-label hard leak guard are all unchanged from
      the prior round.
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
    """The REAL enforced pooled placebo verdict (production code path) — gate v2."""
    return bool(
        _pooled_placebo_verdict(fx["aligned_real_ic"], fx["placebo_ic"])["pass_placebo"]
    )


def _shadow(fx: dict) -> bool:
    """The SHADOW-ONLY gate v3 candidate verdict — never decides pass_placebo."""
    return bool(
        _pooled_placebo_verdict(fx["aligned_real_ic"], fx["placebo_ic"])[
            "sanity_placebo_v3_shadow_verdict"
        ]
    )


# --------------------------------------------------------------------------- #
# (0) Gate version UNCHANGED; shadow criterion margin is pinned for the replay.
# --------------------------------------------------------------------------- #
def test_gate_version_unchanged_v3_is_shadow_only():
    """The ENFORCED rule is still v2's absolute ceiling; GATE_VERSION stays 2
    until v3's shadow evaluation validates it."""
    assert GATE_VERSION == 2
    assert GATE_DIAGNOSTIC_VERSION == 2


def test_margin_is_frozen_at_0_02():
    """FROZEN for the SHADOW evaluation (unified 107 master plan S1–S3 row: 0.02
    vs the measured ~+0.04 shared embargo floor). Changing this constant requires
    a NEW design PR — do not 'tune' it here."""
    assert PLACEBO_GENUINE_IC_MARGIN == 0.02


def test_criterion_strings_distinguish_enforced_from_shadow():
    """Every stamped verdict distinguishes the ENFORCED (v2) criterion from the
    SHADOW-ONLY (v3 candidate) criterion — they must never be presented as the
    same string, since only one of them decides pass_placebo."""
    verdict = _pooled_placebo_verdict(_TODAY["aligned_real_ic"], _TODAY["placebo_ic"])
    assert "0.5" in verdict["placebo_criterion"] or "aligned_real_ic" in verdict[
        "placebo_criterion"
    ]
    assert verdict["placebo_criterion"] != PLACEBO_CRITERION
    assert verdict["sanity_placebo_v3_criterion"] == PLACEBO_CRITERION
    assert verdict["sanity_placebo_genuine_ic_margin"] == PLACEBO_GENUINE_IC_MARGIN
    assert verdict["sanity_placebo_v3_gating"] is False


# --------------------------------------------------------------------------- #
# (a) CLEAN model PASSES.
# --------------------------------------------------------------------------- #
def test_clean_model_passes():
    """Real IC well above placebo + margin → genuine_ic 0.08 > 0.02 → PASS."""
    assert _genuine_ic_value(**_CLEAN) == pytest.approx(0.08, abs=1e-12)
    assert _enforced(_CLEAN) is True


def test_structural_floor_false_reject_shadow_shows_the_fix_not_yet_enforced():
    """The real 2026-06-23 candidate (genuine_ic +0.0324, shuffle-clean) is
    exactly the false reject the v3 difference test was designed to repair: its
    SHADOW verdict PASSES, showing the fix works. But the currently ENFORCED
    gate-v2 absolute ceiling still FAILS it (0.0529 ≥ 0.5×0.0853 = 0.04265) —
    this is the expected, correct state while v3 remains shadow-only pending
    historical-corpus validation, not a contradiction."""
    g = _genuine_ic_value(_TODAY["aligned_real_ic"], _TODAY["placebo_ic"])
    assert g == pytest.approx(0.0324, abs=1e-9)
    assert _shadow(_TODAY) is True
    assert _enforced(_TODAY) is False
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
    and the SHADOW pooled path (not the enforced one, which is v2's absolute
    ceiling and has no relationship to this margin) is checked just below/above."""
    assert _placebo_difference_pass(PLACEBO_GENUINE_IC_MARGIN) is False   # strict >
    assert _placebo_difference_pass(PLACEBO_GENUINE_IC_MARGIN + 1e-9) is True
    just_below = {"aligned_real_ic": 0.06, "placebo_ic": 0.045}   # genuine 0.015
    assert _shadow(just_below) is False
    just_above = {"aligned_real_ic": 0.06, "placebo_ic": 0.035}   # genuine 0.025
    assert _shadow(just_above) is True


# --------------------------------------------------------------------------- #
# (d) The v3 candidate DIFFERENCE-test verdict is STAMPED as SHADOW evidence.
# --------------------------------------------------------------------------- #
def test_v3_shadow_verdict_stamped_alongside_enforced_absolute_rule():
    """Every pooled verdict stamps BOTH the ENFORCED gate-v2 absolute-ceiling
    verdict (which decides pass_placebo) and the SHADOW-ONLY gate-v3 difference
    test (which does not) — evidence continuity in both directions."""
    v = _pooled_placebo_verdict(_TODAY["aligned_real_ic"], _TODAY["placebo_ic"])
    # ENFORCED absolute rule: FAILS (0.0529 ≥ 0.04265) — this decides pass_placebo.
    assert v["sanity_placebo_absolute_rule_pass"] is False
    assert v["sanity_placebo_absolute_rule_threshold"] == pytest.approx(
        0.04265, abs=1e-9
    )
    assert v["pass_placebo"] is False
    # SHADOW v3 difference test: PASSES (genuine_ic 0.0324 > 0.02) — stamped, but
    # does not decide anything.
    assert v["sanity_placebo_v3_shadow_verdict"] is True
    assert v["sanity_placebo_v3_gating"] is False


def test_shadow_verdict_diverges_from_enforced_in_either_direction():
    """The two rules are genuinely independent and diverge in BOTH directions: a
    tiny-edge candidate the ENFORCED absolute rule PASSES (placebo under the
    0.005 floor) FAILS the SHADOW difference test (genuine 0.009 < 0.02) — the
    shadow candidate is not 'enforced rule plus'."""
    v = _pooled_placebo_verdict(0.01, 0.001)  # aligned_real 0.01, placebo 0.001
    assert v["sanity_placebo_absolute_rule_pass"] is True    # enforced: PASS
    assert v["pass_placebo"] is True
    assert v["sanity_placebo_v3_shadow_verdict"] is False    # shadow: FAIL


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
# Fail-closed behavior of the SHADOW-ONLY difference test's genuine_ic guard.
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
def test_shadow_genuine_ic_guard_fails_closed(aligned_real, placebo, label):
    """Missing evidence or a non-positive aligned real IC → genuine_ic None →
    SHADOW verdict FAILS. This guard is independent of the ENFORCED gate-v2
    absolute rule, which has its own separate (and, for aligned_real==0 or a
    small |placebo|, sometimes PASSING) special-case behavior — see
    test_v2_zero_and_negative_aligned_real_special_cases below for that."""
    assert _genuine_ic_value(aligned_real, placebo) is None, label
    v = _pooled_placebo_verdict(aligned_real, placebo)
    assert v["sanity_placebo_v3_shadow_verdict"] is False, label


def test_v2_zero_and_negative_aligned_real_special_cases():
    """The ENFORCED gate-v2 absolute rule has its own pre-existing special cases
    that are genuinely independent of the shadow difference test's
    positive-aligned-real guard: aligned_real_ic == 0 is a legacy pass special
    case (_placebo_absolute_rule_pass docstring), and a negative aligned_real
    with a placebo comfortably inside the (still-computed) 0.5x|aligned_real|
    ceiling also passes. Both PASS under the enforced v2 rule even though their
    shadow genuine_ic is None (guarded, shadow-FAILs) — this is real, unchanged
    v2 behavior, not a bug introduced by keeping v2 enforced."""
    assert _pooled_placebo_verdict(0.0, -0.02)["pass_placebo"] is True
    assert _pooled_placebo_verdict(-0.10, 0.02)["pass_placebo"] is True


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
# The per-regime shadow candidate reuses the SAME helper/margin as the pooled
# leg — but per-regime ENFORCEMENT is the absolute rule (gate v2, unchanged);
# this test exercises the shared SHADOW-candidate helper directly, not the
# actual per-regime gating in runner.py (which now uses the absolute rule,
# same as the pooled leg — see runner.py's regime loop).
# --------------------------------------------------------------------------- #
def test_regime_shadow_candidate_uses_same_helper_and_margin():
    """The shared genuine_ic/difference-test helpers behave identically whether
    called for the pooled leg or a per-regime reading: the real
    structural-floor regime (aligned 0.057 / placebo 0.0359, genuine +0.0211)
    shadow-passes; a floor-only regime (genuine ≈ 0) shadow-fails. Per-regime
    ENFORCEMENT itself is the gate-v2 absolute rule, per Codex's 2026-07-02
    review (per-regime looks are an additional multiplicity concern on top of
    the pooled-leg calibration concern, so v3 stays shadow-only there too)."""
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
    """The CI payload stamps evidence, is tagged as SHADOW-ONLY, and never
    carries pass/fail — the enforced verdict (gate v2's absolute ceiling) is
    computed independently by _pooled_placebo_verdict and does not depend on
    this payload at all."""
    d = _genuine_ic_diagnostic(_TODAY["aligned_real_ic"], _TODAY["placebo_ic"])
    assert d["genuine_ic"] == pytest.approx(0.0324, abs=1e-9)
    assert d["positive_aligned_real"] is True
    assert "SHADOW-ONLY" in d["tag"]
    assert "NOT enforced" in d["tag"]
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


# --------------------------------------------------------------------------- #
# Historical-corpus replay regression (2026-07-03, PR #61 round 3+).
#
# Only two independent candidate fingerprints exist in the available
# production wf_gate_metadata corpus (weekly staging/rollback snapshots,
# 2026-06-15..2026-06-30 -- 20 stamped records collapse to 2 distinct
# `candidate_recipe_fingerprint`s once repeated re-stagings of the same
# untrained-since model are deduped). NOTE: the top-level `passed` field in
# these artifacts is the FULL gate's verdict (shuffle leg + placebo leg +
# regime-sanity leg + WF-parity, etc combined) and must NOT be read as the
# placebo sub-gate's own verdict -- recompute via `_placebo_absolute_rule_pass`
# directly, which is what this test (and the replay doc) do.
#
# Ground truth: the challenger candidate's placebo leg disagrees with v3 in
# 8 of its 9 observed (rolling-window) evaluations -- always in the SAME
# direction (v2 placebo-leg FAIL, v3 PASS). The incumbent candidate's placebo
# leg AGREES with v3 (both FAIL). This is n=2 independent underlying models
# -- far too small to validate a frozen threshold, which is the negative
# finding this test pins so it can't silently rot if the gate functions
# change. See docs/progress/2026-07-03-wf-gate-genuine-ic-historical-replay.md.
# --------------------------------------------------------------------------- #
def test_replay_regression_challenger_candidate_typical_week():
    """Challenger (fp sha256:cfdd6cb8e950da0f), 2026-06-23 stamp: v2 placebo
    leg FAILS, v3 shadow rule PASSES -- the typical week (8 of 9 observed)."""
    aligned_real_ic = 0.08534983744871315
    placebo_ic = 0.0529174995311023
    assert _placebo_absolute_rule_pass(aligned_real_ic, placebo_ic) is False
    genuine_ic = _genuine_ic_value(aligned_real_ic, placebo_ic)
    assert genuine_ic == pytest.approx(0.0324, abs=5e-4)
    assert _placebo_difference_pass(genuine_ic) is True


def test_replay_regression_challenger_candidate_exception_week():
    """Same challenger fingerprint, 2026-06-21/22 stamp: the ONE week (of 9)
    where the rolling eval window shifted enough that v2's placebo leg also
    passes -- v2 and v3 agree here. Not evidence the threshold is validated;
    it is one data point in the same tiny sample."""
    aligned_real_ic = 0.07587675482134941
    placebo_ic = 0.03434988208038318
    assert _placebo_absolute_rule_pass(aligned_real_ic, placebo_ic) is True
    genuine_ic = _genuine_ic_value(aligned_real_ic, placebo_ic)
    assert genuine_ic == pytest.approx(0.0415, abs=5e-4)
    assert _placebo_difference_pass(genuine_ic) is True


def test_replay_regression_incumbent_candidate():
    """Incumbent/rollback candidate (fp sha256:aeb1cd20db700361), 2026-06-30
    stamp: v2 placebo leg FAILS, v3 shadow rule ALSO FAILS -- they agree."""
    aligned_real_ic = 0.052921747444921494
    placebo_ic = 0.040151728638540385
    assert _placebo_absolute_rule_pass(aligned_real_ic, placebo_ic) is False
    genuine_ic = _genuine_ic_value(aligned_real_ic, placebo_ic)
    assert genuine_ic == pytest.approx(0.0128, abs=5e-4)
    assert _placebo_difference_pass(genuine_ic) is False
