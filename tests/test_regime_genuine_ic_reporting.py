"""orch#805: the gate already knew, four levels deep, and nobody read it.

MEASURED on the 2026-07-06 staging stamp: the model's whole edge is in BEAR
(genuine_ic +0.335, n=50, placebo +0.016) while BULL_CALM — 363 of 452 dates,
and 136 of the strategy's 154 buys — is NEGATIVE (-0.029). The pooled +0.0089
that every promote/reject decision was read off is a regime-mix artifact.

This is a REPORTING surface. If any of these tests ever has to assert that it
changed a verdict, the change was wrong.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from renquant_backtesting.wf_gate.runner import (
    format_regime_genuine_ic,
    regime_genuine_ic_summary,
)

# The literal profile shape stamped by _build_diagnostic_profiles, values as
# read out of panel-ltr.alpha158_fund.weekly_20260706T230931Z.staging.json.
REAL_PROFILE = {
    "pooled": {"2x": {"aligned_real_ic": 0.0659050614827684,
                      "placebo_ic": 0.05704956961095635,
                      "genuine_ic": 0.008855491871812046,
                      "label_autocorr_ic": 0.04818312175781547,
                      "n_dates": 452}},
    "per_regime": {
        "BEAR": {"1x": {"genuine_ic": 0.245},
                 "2x": {"aligned_real_ic": 0.3508898711944478,
                        "placebo_ic": 0.016249617164625113,
                        "genuine_ic": 0.33464025402982267,
                        "label_autocorr_ic": 0.02962710044748641,
                        "n_dates": 50}},
        "BULL_CALM": {"2x": {"aligned_real_ic": 0.029972087331594996,
                             "placebo_ic": 0.059451228730756894,
                             "genuine_ic": -0.0294791413991619,
                             "label_autocorr_ic": 0.05092839950583167,
                             "n_dates": 363}},
        "BULL_VOLATILE": {"2x": {"genuine_ic": -0.08004687342463489,
                                 "n_dates": 11}},
        "CHOPPY": {"2x": {"genuine_ic": -0.0409959442809048, "n_dates": 28}},
    },
}


class TestSummary:
    def test_it_reads_the_2x_shift_the_ENFORCED_leg_uses(self):
        """2x is the placebo leg's own shift (2 x horizon). A summary computed
        off a different shift would describe a different experiment than the
        verdict it sits next to."""
        s = regime_genuine_ic_summary(REAL_PROFILE)
        assert s["BEAR"]["genuine_ic"] == pytest.approx(0.33464025402982267)
        assert s["BEAR"]["n_dates"] == 50
        assert s["BULL_CALM"]["genuine_ic"] == pytest.approx(-0.0294791413991619)
        # the 1x cell present on BEAR must NOT be the one reported
        assert s["BEAR"]["genuine_ic"] != pytest.approx(0.245)

    def test_it_carries_the_context_needed_to_interpret_the_number(self):
        """genuine_ic alone cannot be read: BULL_CALM's placebo is mostly label
        autocorrelation (+0.0509 of +0.0595), which is the whole argument that
        the ratio rule is the wrong instrument."""
        s = regime_genuine_ic_summary(REAL_PROFILE)["BULL_CALM"]
        assert s["placebo_ic"] == pytest.approx(0.059451228730756894)
        assert s["label_autocorr_ic"] == pytest.approx(0.05092839950583167)
        assert s["aligned_real_ic"] == pytest.approx(0.029972087331594996)

    def test_a_missing_profile_yields_an_empty_summary_not_a_crash(self):
        for bad in (None, {}, {"per_regime": None}, {"per_regime": {}}):
            assert regime_genuine_ic_summary(bad) == {}

    def test_a_regime_with_no_2x_cell_is_OMITTED_not_zero_filled(self):
        """An absent measurement must read as absent. A zero would read as
        'measured, and it is zero'."""
        s = regime_genuine_ic_summary({"per_regime": {"BEAR": {"1x": {"genuine_ic": 0.2}}}})
        assert s == {}


class TestLogLine:
    def test_the_WORST_regime_is_printed_first(self):
        line = format_regime_genuine_ic(regime_genuine_ic_summary(REAL_PROFILE))
        order = [line.index(r) for r in ("BULL_VOLATILE", "CHOPPY", "BULL_CALM", "BEAR")]
        assert order == sorted(order), line

    def test_it_names_the_negative_regimes_with_their_sample_size(self):
        line = format_regime_genuine_ic(regime_genuine_ic_summary(REAL_PROFILE))
        assert "BULL_CALM=-0.0295(n=363)" in line
        assert "BEAR=+0.3346(n=50)" in line

    def test_an_unavailable_profile_says_UNAVAILABLE_not_nothing(self):
        assert "UNAVAILABLE" in format_regime_genuine_ic({})

    def test_a_non_numeric_genuine_ic_degrades_to_n_slash_a(self):
        line = format_regime_genuine_ic({"BEAR": {"genuine_ic": None, "n_dates": 5}})
        assert "BEAR=n/a" in line


def test_reporting_only_it_is_referenced_by_NO_pass_fail_leg():
    """The load-bearing constraint. If `sanity_regime_genuine_ic` or either
    helper ever appears inside the verdict computation, this fails."""
    src = (Path(__file__).resolve().parent.parent / "src" / "renquant_backtesting"
           / "wf_gate" / "runner.py").read_text()
    start = src.index("def _compute_overall_pass(")
    body = src[start:src.index("def _sanity_result_passed(")]
    for forbidden in ("sanity_regime_genuine_ic", "regime_genuine_ic_summary",
                      "format_regime_genuine_ic"):
        assert forbidden not in body, f"{forbidden} must not decide anything"
