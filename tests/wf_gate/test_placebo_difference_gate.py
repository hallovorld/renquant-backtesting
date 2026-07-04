"""S3 placebo difference gate — the pass/fail criterion uses genuine_ic.

The 60d label has ~30d embargo overlap that inflates time-shift placebo IC by
~+0.04.  The legacy ratio test (|placebo_ic| < 0.5 × |real_ic|) was
structurally unfair because it didn't account for this floor.  The S3 fix uses
genuine_ic = real_ic − placebo_ic ≥ PLACEBO_DIFF_MARGIN (0.02) instead.

These tests verify:
- A known-clean model (genuine_ic well above margin) PASSES
- A leaked model (genuine_ic near zero despite high raw IC) FAILS
- Fallback to legacy when Layer-1a profile is unavailable
"""
from __future__ import annotations

import math

from renquant_backtesting.wf_gate.runner import (
    PLACEBO_DIFF_MARGIN,
    _assemble_diagnostic_profiles,
)


def _profile_with_genuine_ic(genuine_ic_2x: float):
    """Build a minimal model_placebo_profile with a given genuine_ic at the 2x gate shift."""
    return {
        "label_col": "fwd_60d_excess",
        "shuf_ic": 0.0,
        "gate_shift_multiple": "2x",
        "pooled": {
            "1x": {"aligned_real_ic": 0.06, "placebo_ic": 0.04, "genuine_ic": 0.02,
                    "label_autocorr_ic": 0.03, "n_dates": 100},
            "2x": {"aligned_real_ic": 0.06, "placebo_ic": 0.06 - genuine_ic_2x,
                    "genuine_ic": genuine_ic_2x,
                    "label_autocorr_ic": 0.05, "n_dates": 90},
            "3x": {"aligned_real_ic": 0.06, "placebo_ic": 0.03, "genuine_ic": 0.03,
                    "label_autocorr_ic": 0.04, "n_dates": 80},
        },
        "per_regime": {},
        "method": "test",
    }


def _eval_pass_placebo(model_placebo_profile, placebo_ic, placebo_aligned_real_ic):
    """Reproduce the gate's pass_placebo logic for unit testing."""
    _genuine_ic_2x = (
        ((model_placebo_profile or {}).get("pooled", {}).get("2x", {}) or {})
        .get("genuine_ic")
    )
    _has_genuine = (
        isinstance(_genuine_ic_2x, (int, float))
        and _genuine_ic_2x == _genuine_ic_2x  # NaN check
    )

    if _has_genuine:
        return float(_genuine_ic_2x) >= PLACEBO_DIFF_MARGIN
    # Legacy fallback
    from renquant_backtesting.wf_gate.runner import _placebo_ic_threshold
    return (
        (placebo_ic == placebo_ic)
        and (placebo_aligned_real_ic == placebo_aligned_real_ic)
        and (
            abs(placebo_ic) < _placebo_ic_threshold(placebo_aligned_real_ic)
            if placebo_aligned_real_ic != 0 else
            True
        )
    )


def test_clean_model_passes():
    """genuine_ic=+0.04 (well above the 0.02 margin) → PASS."""
    profile = _profile_with_genuine_ic(0.04)
    assert _eval_pass_placebo(profile, placebo_ic=0.02, placebo_aligned_real_ic=0.06)


def test_leaked_model_fails():
    """genuine_ic=+0.005 (model adds almost nothing over placebo) → FAIL."""
    profile = _profile_with_genuine_ic(0.005)
    assert not _eval_pass_placebo(profile, placebo_ic=0.055, placebo_aligned_real_ic=0.06)


def test_negative_genuine_ic_fails():
    """genuine_ic=−0.01 (placebo BEATS the model) → FAIL."""
    profile = _profile_with_genuine_ic(-0.01)
    assert not _eval_pass_placebo(profile, placebo_ic=0.07, placebo_aligned_real_ic=0.06)


def test_exactly_at_margin_passes():
    """genuine_ic exactly at the margin → PASS (≥ not >)."""
    profile = _profile_with_genuine_ic(PLACEBO_DIFF_MARGIN)
    assert _eval_pass_placebo(profile, placebo_ic=0.04, placebo_aligned_real_ic=0.06)


def test_just_below_margin_fails():
    """genuine_ic just below the margin → FAIL."""
    profile = _profile_with_genuine_ic(PLACEBO_DIFF_MARGIN - 0.001)
    assert not _eval_pass_placebo(profile, placebo_ic=0.041, placebo_aligned_real_ic=0.06)


def test_legacy_fallback_when_profile_unavailable():
    """When Layer-1a profile is None, falls back to the legacy ratio test."""
    # Legacy should PASS: placebo_ic=0.02 < 0.5 × 0.06 = 0.03
    assert _eval_pass_placebo(None, placebo_ic=0.02, placebo_aligned_real_ic=0.06)
    # Legacy should FAIL: placebo_ic=0.04 > 0.5 × 0.06 = 0.03
    assert not _eval_pass_placebo(None, placebo_ic=0.04, placebo_aligned_real_ic=0.06)


def test_legacy_fallback_when_genuine_ic_is_none():
    """Profile present but genuine_ic missing at 2x → legacy fallback."""
    profile = _profile_with_genuine_ic(0.04)
    profile["pooled"]["2x"]["genuine_ic"] = None
    # Legacy: placebo_ic=0.02 < 0.5 × 0.06 = 0.03 → PASS
    assert _eval_pass_placebo(profile, placebo_ic=0.02, placebo_aligned_real_ic=0.06)


def test_margin_constant_value():
    """PLACEBO_DIFF_MARGIN is 0.02 per master plan §S3."""
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
    # genuine_ic=0.023 ≥ PLACEBO_DIFF_MARGIN=0.02 → PASS
    assert _eval_pass_placebo(profile, placebo_ic=0.036, placebo_aligned_real_ic=0.059)
