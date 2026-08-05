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


# [codex on bt#105] Scanning only _compute_overall_pass misses the OTHER
# verdict-producing spans — above all `run_sanity_battery`, where
# `pass_all = pass_shuf and pass_placebo and pass_regime` is formed. A future
# reference there would change behaviour without touching the scanned slice, so
# the advertised guard would not have been guarding.
_REPORTING_SYMBOLS = ("sanity_regime_genuine_ic", "regime_genuine_ic_summary",
                      "format_regime_genuine_ic")

# Every place a pass/fail is COMPUTED. Each is asserted to exist, so a rename
# that silently empties this list fails instead of vacuously passing.
_VERDICT_FUNCTIONS = (
    "_compute_overall_pass",
    "_sanity_result_passed",
    "_placebo_difference_pass",
    "_placebo_absolute_rule_pass",
    "_pooled_placebo_verdict",
    "run_sanity_battery",
)


def _runner_source() -> str:
    return (Path(__file__).resolve().parent.parent / "src" / "renquant_backtesting"
            / "wf_gate" / "runner.py").read_text()


def _function_body(src: str, name: str) -> str:
    """Source of `def name(` up to the next TOP-LEVEL `def `/`class `.

    Nested defs and decorated helpers inside the function are indented, so they
    do not match `\ndef `/`\nclass ` and stay inside the returned body — which
    is what the guard wants: a reference hidden in a closure still counts.
    """
    marker = f"\ndef {name}("
    start = src.index(marker) + 1          # first char of `def`, not the newline
    rest = src[start:]
    tail = rest[1:]                        # skip our own `def` when looking ahead
    ends = [tail.index(m) for m in ("\ndef ", "\nclass ") if m in tail]
    return rest[:min(ends) + 1] if ends else rest


def assert_decides_nothing(body: str, func: str) -> None:
    """THE guard. Both the live check and its own regression call THIS, so the
    regression cannot pass while the check has stopped checking. [codex on bt#106]"""
    for forbidden in _REPORTING_SYMBOLS:
        assert forbidden not in body, (
            f"{forbidden} appears inside {func} — a reporting surface must not "
            f"decide anything")


@pytest.mark.parametrize("func", _VERDICT_FUNCTIONS)
def test_reporting_only_no_verdict_function_references_it(func):
    """The load-bearing constraint, across EVERY pass/fail-producing span."""
    src = _runner_source()
    assert f"\ndef {func}(" in src, (
        f"{func} no longer exists — this guard is now scanning nothing; "
        f"re-derive the list of verdict-producing functions")
    assert_decides_nothing(_function_body(src, func), func)


def test_the_guard_REJECTS_a_planted_reference():
    """Anti-vacuity for the guard itself. It is not enough to show the planted
    string is present — that is tautological and would still pass if the live
    check stopped inspecting bodies. This runs the SAME assertion path and
    requires it to RAISE. [codex on bt#106]"""
    src = _runner_source()
    body = _function_body(src, "run_sanity_battery")
    assert "pass_all" in body and len(body) > 500, (
        "the extractor did not return run_sanity_battery's real body")

    planted = body.replace("pass_all", "pass_all and sanity_regime_genuine_ic", 1)
    assert planted != body, "the plant did not apply — the regression is vacuous"
    with pytest.raises(AssertionError, match="must not decide anything"):
        assert_decides_nothing(planted, "run_sanity_battery")

    # ... and the unplanted body must still pass through the same call, or the
    # guard would be rejecting everything and proving nothing.
    assert_decides_nothing(body, "run_sanity_battery")


@pytest.mark.parametrize("func", _VERDICT_FUNCTIONS)
def test_every_extracted_body_is_REAL_source_not_an_empty_string(func):
    """A body extractor that silently returns '' would make every guard above
    pass vacuously — the exact failure mode this file exists to prevent."""
    body = _function_body(_runner_source(), func)
    assert body.startswith(f"def {func}("), (func, body[:60])
    assert len(body) > 120, (func, len(body))
    # a body that is only a signature would pass a substring check vacuously
    assert "return" in body or "assert" in body, func


def test_the_summary_IS_reachable_from_the_stamping_path():
    """The mirror image: reporting-only must not mean unreachable. If nothing
    stamps it, the whole PR is inert scaffolding."""
    src = _runner_source()
    assert '"sanity_regime_genuine_ic": regime_genuine_ic_summary(' in src
    assert "format_regime_genuine_ic(wf_meta.get(" in src
