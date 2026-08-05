"""orch#805: the gate already knew, four levels deep, and nobody read it.

MEASURED on the 2026-07-06 staging stamp: the model's whole edge is in BEAR
(genuine_ic +0.335, n=50, placebo +0.016) while BULL_CALM — 363 of 452 dates,
and 136 of the strategy's 154 buys — is NEGATIVE (-0.029). The pooled +0.0089
that every promote/reject decision was read off is a regime-mix artifact.

This is a REPORTING surface. If any of these tests ever has to assert that it
changed a verdict, the change was wrong.
"""
from __future__ import annotations

import ast
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


# [codex on bt#105] Scanning only _compute_overall_pass missed the OTHER
# verdict-producing spans — above all `run_sanity_battery`, where
# `pass_all = pass_shuf and pass_placebo and pass_regime` is formed, and `main`,
# where `overall_pass` is assembled and handed to `sys.exit`.
#
# SCOPE, deliberately narrow [codex on bt#106]: this is a DIRECT-REFERENCE
# invariant. It catches the mistake that actually happens — someone wires the
# reporting summary into a verdict — and it does NOT chase indirection through
# aliases, wrappers, methods, `for`/`with` binds or walrus. An earlier revision
# grew a 372-line AST dataflow analyser to close those; that is disproportionate
# maintenance and false-positive risk for a reporting-only change, and it belongs
# in a separately scoped tool if it is ever wanted. The wording here claims only
# what it checks.
_REPORTING_SYMBOLS = ("sanity_regime_genuine_ic", "regime_genuine_ic_summary",
                      "format_regime_genuine_ic")

# Every function where a pass/fail is COMPUTED. Each is asserted to exist, so a
# rename empties the list loudly instead of passing vacuously.
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
    """Source of `def name(` up to the next top-level `def `/`class `.

    Trailing decorators and blank lines belonging to the NEXT definition are
    dropped, so exactly one function comes back.
    """
    marker = f"\ndef {name}("
    start = src.index(marker) + 1          # first char of `def`, not the newline
    rest = src[start:]
    tail = rest[1:]                        # skip our own `def` when looking ahead
    ends = [tail.index(m) for m in ("\ndef ", "\nclass ") if m in tail]
    body = rest[:min(ends) + 1] if ends else rest
    lines = body.splitlines(keepends=True)
    while lines and (not lines[-1].strip() or lines[-1].startswith("@")):
        lines.pop()
    return "".join(lines)


def assert_decides_nothing(text: str, where: str) -> None:
    """THE guard. The live checks and their regressions all call THIS, so a
    regression cannot pass while the check has stopped checking."""
    for forbidden in _REPORTING_SYMBOLS:
        assert forbidden not in text, (
            f"{forbidden} appears in {where} — a reporting surface must not "
            f"decide anything")


def _verdict_expressions(src: str) -> list[tuple[str, str]]:
    """(label, source text) for what PRODUCES the verdict in `main`.

    `main` cannot be scanned wholesale: it legitimately mentions the reporting
    symbols, because it is where they are STAMPED. The subject is the expression
    the verdict is computed from — the value of `overall_pass = …` and the
    argument of the `sys.exit` that carries it.
    """
    tree = ast.parse(src)
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    found: list[tuple[str, str]] = []
    for node in ast.walk(main):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "overall_pass"
                for t in node.targets):
            found.append(("overall_pass = <expr>", ast.unparse(node.value)))
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "exit"):
            for arg in node.args:
                text = ast.unparse(arg)
                if "overall_pass" in text:
                    found.append(("sys.exit(<verdict>)", text))
    return found


@pytest.mark.parametrize("func", _VERDICT_FUNCTIONS)
def test_no_verdict_function_references_a_reporting_symbol(func):
    src = _runner_source()
    assert f"\ndef {func}(" in src, (
        f"{func} no longer exists — this guard is now scanning nothing; "
        f"re-derive the list of verdict-producing functions")
    assert_decides_nothing(_function_body(src, func), func)


def test_mains_verdict_EXPRESSIONS_are_clean():
    """The span the first list omitted, checked at expression granularity."""
    exprs = _verdict_expressions(_runner_source())
    labels = [lbl for lbl, _ in exprs]
    assert any("overall_pass =" in lbl for lbl in labels), labels
    assert any("sys.exit" in lbl for lbl in labels), labels
    for label, text in exprs:
        assert_decides_nothing(text, f"main {label}")


def test_the_guard_FIRES_on_a_planted_reference():
    """Anti-vacuity: it is not enough to show the plant is present — that would
    still pass if the checks stopped inspecting. This runs the SAME assertion
    path and requires it to RAISE, in a verdict function and in main's verdict
    expression."""
    src = _runner_source()
    body = _function_body(src, "run_sanity_battery")
    assert "pass_all" in body and len(body) > 500, "extractor returned no real body"
    with pytest.raises(AssertionError, match="must not decide anything"):
        assert_decides_nothing(
            body.replace("pass_all", "pass_all and sanity_regime_genuine_ic", 1),
            "run_sanity_battery")

    planted = src.replace(
        "    overall_pass = _compute_overall_pass(",
        "    overall_pass = sanity_regime_genuine_ic and _compute_overall_pass(", 1)
    assert planted != src, "the plant did not apply — the regression is vacuous"
    with pytest.raises(AssertionError, match="must not decide anything"):
        for label, text in _verdict_expressions(planted):
            assert_decides_nothing(text, f"main {label}")


def test_the_guard_TOLERATES_the_legitimate_stamping_statement():
    """Anti-false-positive: `wf_meta = {...}` holds BOTH `"passed": overall_pass`
    and the reporting summary, in one correct statement. Scanning `main`
    wholesale would reject it — which is why the subject is the expression."""
    src = _runner_source()
    assert '"passed": overall_pass' in src
    assert '"sanity_regime_genuine_ic": regime_genuine_ic_summary(' in src
    assert _verdict_expressions(src), "no verdict expression found at all"


@pytest.mark.parametrize("func", _VERDICT_FUNCTIONS)
def test_every_extracted_body_is_REAL_source(func):
    """A extractor that silently returned '' would make every guard above pass
    vacuously."""
    body = _function_body(_runner_source(), func)
    assert body.startswith(f"def {func}("), (func, body[:60])
    assert len(body) > 120 and ("return" in body or "assert" in body), func


def test_the_extractor_stops_before_the_next_functions_DECORATOR():
    """`runner.py` has no decorators today, so this pins behaviour rather than
    the current file."""
    src = "\ndef first():\n    return 1\n\n\n@dec\ndef second():\n    return 2\n"
    body = _function_body(src, "first")
    assert body.rstrip().endswith("return 1"), repr(body)
    assert "@dec" not in body and "second" not in body


def test_the_OUT_OF_SCOPE_indirection_is_written_down_not_forgotten():
    """Aliases, wrappers, methods, `for`/`with` binds and walrus are NOT
    checked. That is a decision — recorded so widening the claim later has to
    confront the list rather than quietly inherit it."""
    doc = " ".join((Path(__file__).resolve().parent.parent / "doc" / "progress"
                    / "2026-08-05-regime-genuine-ic-in-verdict.md").read_text().split())
    assert "OUT OF SCOPE" in doc
    for shape in ("alias", "wrapper", "method", "walrus"):
        assert shape in doc, shape


def test_the_summary_IS_reachable_from_the_stamping_path():
    """Reporting-only must not quietly become unreachable."""
    src = _runner_source()
    assert '"sanity_regime_genuine_ic": regime_genuine_ic_summary(' in src
    assert "format_regime_genuine_ic(wf_meta.get(" in src
