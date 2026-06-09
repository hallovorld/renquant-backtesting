"""RFC #259 Layer 1a (P1) — diagnostic profile assembly tests.

Targets the pure assembly (`_assemble_diagnostic_profiles`): the
genuine_ic = aligned_real_ic − placebo_ic arithmetic, the {1x,2x,3x} shift
nesting, pooled + per-regime, and graceful handling of missing rows. The
profiles are diagnostic-only and must never affect the gate verdict, so the
test asserts shape/values, not pass/fail.
"""
from __future__ import annotations

import math

from renquant_backtesting.wf_gate.runner import _assemble_diagnostic_profiles


def _rows(h):
    """shift_diagnostics-shaped rows at {1x,2x,3x}×h for a fwd_h label."""
    return [
        {"shift_days": h, "aligned_real_ic": 0.059, "model_placebo_ic": 0.040,
         "label_autocorr_ic": 0.036, "n_dates": 120},
        {"shift_days": 2 * h, "aligned_real_ic": 0.059, "model_placebo_ic": 0.036,
         "label_autocorr_ic": 0.049, "n_dates": 110},
        {"shift_days": 3 * h, "aligned_real_ic": 0.058, "model_placebo_ic": 0.030,
         "label_autocorr_ic": 0.041, "n_dates": 100},
    ]


def test_genuine_ic_is_real_minus_placebo():
    autocorr, placebo = _assemble_diagnostic_profiles(
        _rows(60), {}, label="fwd_60d_excess", label_horizon=60, shuf_ic=-0.0005,
    )
    # gate shift = 2x: genuine = 0.059 - 0.036 = 0.023 (the RFC's headline number)
    g2 = placebo["pooled"]["2x"]["genuine_ic"]
    assert math.isclose(g2, 0.023, abs_tol=1e-9)
    assert math.isclose(placebo["pooled"]["1x"]["genuine_ic"], 0.019, abs_tol=1e-9)
    assert math.isclose(placebo["pooled"]["3x"]["genuine_ic"], 0.028, abs_tol=1e-9)
    assert placebo["shuf_ic"] == -0.0005
    assert placebo["gate_shift_multiple"] == "2x"


def test_label_autocorr_profile_picks_right_shifts():
    autocorr, _ = _assemble_diagnostic_profiles(
        _rows(60), {}, label="fwd_60d_excess", label_horizon=60, shuf_ic=0.0,
    )
    assert autocorr["horizon_days"] == 60
    assert autocorr["shift_multiples"] == {"1x": 60, "2x": 120, "3x": 180}
    assert autocorr["pooled"]["2x"] == 0.049  # the +0.049 lag-120 confound
    assert autocorr["pooled"]["1x"] == 0.036


def test_per_regime_nested():
    per_regime = {"BULL_CALM": _rows(60), "BEAR": _rows(60)}
    autocorr, placebo = _assemble_diagnostic_profiles(
        per_regime_rows=per_regime, pooled_rows=_rows(60),
        label="fwd_60d_excess", label_horizon=60, shuf_ic=0.0,
    )
    assert set(placebo["per_regime"]) == {"BULL_CALM", "BEAR"}
    assert math.isclose(placebo["per_regime"]["BULL_CALM"]["2x"]["genuine_ic"], 0.023, abs_tol=1e-9)
    assert autocorr["per_regime"]["BEAR"]["2x"] == 0.049


def test_missing_shift_row_yields_none_not_crash():
    partial = [{"shift_days": 60, "aligned_real_ic": 0.05, "model_placebo_ic": 0.03,
                "label_autocorr_ic": 0.02, "n_dates": 40}]  # only 1x present
    autocorr, placebo = _assemble_diagnostic_profiles(
        partial, {}, label="fwd_60d_excess", label_horizon=60, shuf_ic=0.0,
    )
    assert math.isclose(placebo["pooled"]["1x"]["genuine_ic"], 0.02, abs_tol=1e-9)
    assert placebo["pooled"]["2x"]["genuine_ic"] is None  # missing → None, no crash
    assert autocorr["pooled"]["3x"] is None


def test_none_ic_yields_none_genuine():
    rows = [{"shift_days": 120, "aligned_real_ic": None, "model_placebo_ic": 0.03,
             "label_autocorr_ic": None, "n_dates": 0}]
    _, placebo = _assemble_diagnostic_profiles(
        rows, {}, label="fwd_60d_excess", label_horizon=60, shuf_ic=0.0,
    )
    assert placebo["pooled"]["2x"]["genuine_ic"] is None


def test_horizon_fallback_to_60():
    autocorr, _ = _assemble_diagnostic_profiles(
        _rows(60), {}, label="x", label_horizon=None, shuf_ic=0.0,
    )
    assert autocorr["shift_multiples"] == {"1x": 60, "2x": 120, "3x": 180}
