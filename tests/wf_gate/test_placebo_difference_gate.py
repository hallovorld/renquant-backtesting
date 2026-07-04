"""S3 placebo difference gate — the pass/fail criterion uses genuine_ic.

The 60d label has ~30d embargo overlap that inflates time-shift placebo IC by
~+0.04.  The legacy ratio test (|placebo_ic| < 0.5 x |real_ic|) was
structurally unfair because it didn't account for this floor.  The S3 fix uses
genuine_ic = real_ic - placebo_ic >= PLACEBO_DIFF_MARGIN (0.02) instead.

These tests verify:
- A known-clean model (genuine_ic well above margin) PASSES
- A leaked model (genuine_ic near zero despite high raw IC) FAILS
- Boundary conditions at/below the margin
- Fallback to legacy when genuine_ic is unavailable
- End-to-end from the diagnostic profile assembly
"""
from __future__ import annotations

import math

from renquant_backtesting.wf_gate.runner import (
    PLACEBO_DIFF_MARGIN,
    _assemble_diagnostic_profiles,
    _genuine_ic_value,
    _placebo_difference_pass,
    _placebo_ic_threshold,
    _pooled_placebo_verdict,
)


def test_clean_model_passes():
    """genuine_ic=+0.04 (well above the 0.02 margin) -> PASS."""
    verdict = _pooled_placebo_verdict(
        placebo_aligned_real_ic=0.06, placebo_ic=0.02,
    )
    assert verdict["pass_placebo"] is True
    assert verdict["sanity_placebo_v3_gating"] is True


def test_leaked_model_fails():
    """genuine_ic=+0.005 (model adds almost nothing over placebo) -> FAIL."""
    verdict = _pooled_placebo_verdict(
        placebo_aligned_real_ic=0.06, placebo_ic=0.055,
    )
    assert verdict["pass_placebo"] is False


def test_negative_genuine_ic_fails():
    """genuine_ic=-0.01 (placebo BEATS the model) -> FAIL."""
    verdict = _pooled_placebo_verdict(
        placebo_aligned_real_ic=0.06, placebo_ic=0.07,
    )
    assert verdict["pass_placebo"] is False


def test_exactly_at_margin_passes():
    """genuine_ic exactly at the margin -> PASS (>= not >)."""
    assert _placebo_difference_pass(PLACEBO_DIFF_MARGIN) is True
    assert _placebo_difference_pass(0.02) is True


def test_just_below_margin_fails():
    """genuine_ic just below the margin -> FAIL."""
    verdict = _pooled_placebo_verdict(
        placebo_aligned_real_ic=0.06, placebo_ic=0.041,
    )
    assert verdict["pass_placebo"] is False


def test_negative_aligned_real_fails_closed():
    """Negative aligned_real_ic -> genuine_ic=None -> FAIL (fail-closed)."""
    verdict = _pooled_placebo_verdict(
        placebo_aligned_real_ic=-0.01, placebo_ic=-0.05,
    )
    assert verdict["pass_placebo"] is False
    assert verdict["sanity_placebo_genuine_ic"] is None


def test_nan_inputs_fail_closed():
    """NaN inputs -> FAIL (fail-closed)."""
    verdict = _pooled_placebo_verdict(
        placebo_aligned_real_ic=float("nan"), placebo_ic=0.02,
    )
    assert verdict["pass_placebo"] is False


def test_legacy_absolute_rule_still_stamped():
    """The legacy v2 absolute-ceiling verdict is still stamped as diagnostic."""
    verdict = _pooled_placebo_verdict(
        placebo_aligned_real_ic=0.06, placebo_ic=0.02,
    )
    assert "sanity_placebo_absolute_rule_pass" in verdict
    assert isinstance(verdict["sanity_placebo_absolute_rule_pass"], bool)


def test_margin_constant_value():
    """PLACEBO_DIFF_MARGIN is 0.02 per master plan S3."""
    assert PLACEBO_DIFF_MARGIN == 0.02


def test_difference_from_assembly():
    """End-to-end: _assemble_diagnostic_profiles produces the genuine_ic the gate reads."""
    rows = [
        {"shift_days": 60, "aligned_real_ic": 0.059, "model_placebo_ic": 0.040,
         "label_autocorr_ic": 0.036, "n_dates": 120},
        {"shift_days": 120, "aligned_real_ic": 0.059, "model_placebo_ic": 0.036,
         "label_autocorr_ic": 0.049, "n_dates": 110},
        {"shift_days": 180, "aligned_real_ic": 0.058, "model_placebo_ic": 0.030,
         "label_autocorr_ic": 0.041, "n_dates": 100},
    ]
    _, profile = _assemble_diagnostic_profiles(
        rows, {}, label="fwd_60d_excess", label_horizon=60, shuf_ic=0.0,
    )
    g2 = profile["pooled"]["2x"]["genuine_ic"]
    assert math.isclose(g2, 0.023, abs_tol=1e-9)
    assert g2 >= PLACEBO_DIFF_MARGIN


def test_placebo_difference_pass_function_directly():
    """Unit test the _placebo_difference_pass function."""
    assert _placebo_difference_pass(0.03) is True
    assert _placebo_difference_pass(0.02) is True
    assert _placebo_difference_pass(0.019) is False
    assert _placebo_difference_pass(0.0) is False
    assert _placebo_difference_pass(-0.01) is False
    assert _placebo_difference_pass(None) is False
    assert _placebo_difference_pass(float("nan")) is False
