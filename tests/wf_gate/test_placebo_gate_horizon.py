"""Regression tests for `_placebo_gate_horizon` + the gate-shift selection.

Pins the 2026-06-02 fix: §5.2 placebo gate metric is shift = 2 × label_horizon,
not a hard-coded 60 days. The hard-coded threshold mis-fires on long-horizon
momentum strategies whose alpha legitimately persists ~60-90 days
(Kelly-Gu-Xiu 2020 RFS Table 7). See diagnostic memo at
``doc/research/2026-06-02-placebo-gate-overstrict-for-long-horizon.md`` in
the umbrella for the GBDT decay-profile evidence that motivated this fix.
"""
from __future__ import annotations

import inspect

import pytest

from renquant_backtesting.wf_gate import runner
from renquant_backtesting.wf_gate.runner import _placebo_gate_horizon


class TestPlaceboGateHorizonParsing:
    """Parse the forecast horizon (days) from a label column name."""

    def test_fwd_60d_excess(self):
        assert _placebo_gate_horizon("fwd_60d_excess") == 60

    def test_fwd_20d_excess(self):
        assert _placebo_gate_horizon("fwd_20d_excess") == 20

    def test_fwd_5d_excess(self):
        assert _placebo_gate_horizon("fwd_5d_excess") == 5

    def test_fwd_120d_excess(self):
        # gate_shift will be 240; the gate iteration adds it to the grid.
        assert _placebo_gate_horizon("fwd_120d_excess") == 120

    def test_fwd_30d_no_suffix(self):
        # The regex anchor allows `fwd_30d` even without `_excess`.
        assert _placebo_gate_horizon("fwd_30d") == 30

    def test_unknown_label_returns_none(self):
        assert _placebo_gate_horizon("custom_label") is None

    def test_empty_string_returns_none(self):
        assert _placebo_gate_horizon("") is None

    def test_none_label_returns_none(self):
        assert _placebo_gate_horizon(None) is None  # type: ignore[arg-type]

    def test_negative_or_zero_horizon_returns_none(self):
        # The regex requires at least one digit, so "fwd_0d_excess" would
        # parse to 0; the helper rejects non-positive horizons.
        assert _placebo_gate_horizon("fwd_0d_excess") is None


class TestPlaceboGateHorizonEdgeCases:
    """Defensive shape checks — the parser must not crash on weird input."""

    @pytest.mark.parametrize("label", [
        "fwd_60d_quantile_excess",
        "fwd_60d_decile_excess",
        "fwd_60d_class",
    ])
    def test_accepts_anything_following_fwd_Nd(self, label):
        assert _placebo_gate_horizon(label) == 60

    def test_rejects_fwdsixtyd(self):
        # The horizon must be a numeric digit run.
        assert _placebo_gate_horizon("fwd_sixtyd_excess") is None

    def test_picks_first_match_if_multiple(self):
        # Regex `search` returns the leftmost match.
        assert _placebo_gate_horizon("fwd_60d_then_fwd_120d") == 60


class TestGateShiftSelection:
    """Documents the gate-shift selection rule: 2 × label_horizon, fallback 60."""

    def test_fwd_60d_picks_120(self):
        h = _placebo_gate_horizon("fwd_60d_excess")
        assert h is not None
        assert 2 * h == 120

    def test_fwd_20d_picks_40(self):
        h = _placebo_gate_horizon("fwd_20d_excess")
        assert 2 * (h or 0) == 40

    def test_fwd_120d_picks_240(self):
        # 240 isn't in the default grid (5,10,20,40,60,80,120,180,252); the
        # gate iteration code inserts it.
        h = _placebo_gate_horizon("fwd_120d_excess")
        assert 2 * (h or 0) == 240

    def test_unknown_label_falls_back_to_60(self):
        # The actual fallback rule lives in the caller (see runner.py): when
        # `_placebo_gate_horizon` returns None, gate_shift = 60. This test
        # asserts that None signal so callers know to use the legacy
        # 60-day metric.
        assert _placebo_gate_horizon("custom_label") is None


def test_wf_gate_metadata_keeps_placebo_gate_fields():
    """Final artifact metadata must expose the selected gate shift."""
    source = inspect.getsource(runner)
    assert '"sanity_label_horizon_days": sanity_result.get(' in source
    assert '"sanity_placebo_gate_shift_days": (' in source


def test_lazy_helper_prefers_packaged_qp_contracts():
    helper = runner._load_qp_helper("qp_contracts")
    assert helper.__name__ == "renquant_backtesting.wf_gate.qp_contracts"
    assert helper.validate_qp_contract_config({}).summary() == "QP disabled"
