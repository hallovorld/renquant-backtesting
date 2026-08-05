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

# `main` is BOTH where the verdict is assembled (`overall_pass = ...`, then
# `sys.exit(0 if overall_pass else 1)`) and where the reporting summary is
# STAMPED — so a blanket "this symbol must not appear" scan cannot apply there.
# [codex on bt#106] The guard for `main` is therefore narrower and exact: no
# statement that touches `overall_pass` may reference a reporting symbol.
_VERDICT_ASSEMBLY_FUNCTION = "main"


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
    body = rest[:min(ends) + 1] if ends else rest
    # [codex on bt#106] The next function's DECORATOR lines sit above its `def`
    # and would otherwise be swallowed into this body. Walk back over trailing
    # top-level decorators and blank lines so exactly one function is returned.
    lines = body.splitlines(keepends=True)
    while lines and (not lines[-1].strip() or lines[-1].startswith("@")):
        lines.pop()
    return "".join(lines)


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


def _names_in(node) -> set[str]:
    """Every identifier appearing anywhere under `node`."""
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            out.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            out.add(sub.attr)
    return out


def _plain_names(node) -> set[str]:
    """Identifiers bound as NAMES under `node` (attributes excluded — `.get` is
    not a dependency and treating it as one smears the slice over the module)."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _verdict_dependencies(src: str, exprs: list) -> set[str]:
    """BACKWARD slice of the verdict, over ASSIGNMENTS only.

    [codex on bt#106] Two earlier attempts were wrong in opposite directions.
    Forward taint smeared over 441 names because `wf_meta = {...}` legitimately
    holds a reporting symbol. Then merging called functions' bodies into the
    slice by NAME re-inflated it to 609, because a local inside a big helper
    that happens to share a name with a `main` local (`md`, `wf_meta`) drags
    main's assignment in — a name-keyed slice cannot tell those apart, and
    real def-use analysis is not worth building inside a test.

    So the slice follows ASSIGNMENTS (main's scope plus module level) and stops
    at call boundaries. Functions reached this way are checked SEPARATELY, by
    `_functions_referencing_reporting`, which asks the only question that
    matters about them: does this function hand back a reporting symbol?
    """
    tree = ast.parse(src)
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    assigns: dict[str, list] = {}
    scopes = [main] + [n for n in tree.body if not isinstance(
        n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    for scope in scopes:
        for node in ast.walk(scope):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    # REBINDINGS only. `md["wf_gate_metadata"] = wf_meta`
                    # mutates a container that is DOWNSTREAM of the verdict;
                    # treating it as "md depends on wf_meta" walks the slice
                    # forward into the stamping code and re-introduces the
                    # false positive this guard exists to avoid.
                    if isinstance(target, ast.Name):
                        assigns.setdefault(target.id, []).append(node.value)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        for sub in target.elts:
                            if isinstance(sub, ast.Name):
                                assigns.setdefault(sub.id, []).append(node.value)

    deps: set[str] = set()
    for _, expr in exprs:
        deps |= _plain_names(expr)
    for _ in range(50):                       # fixpoint
        grew = False
        for name in list(deps):
            for value in assigns.get(name, ()):
                fresh = _plain_names(value) - deps
                if fresh:
                    deps |= fresh
                    grew = True
        if not grew:
            break
    else:                                     # pragma: no cover - defensive
        raise AssertionError("dependency slice did not converge")
    return deps


def _functions_referencing_reporting(src: str, names: set[str]) -> set[str]:
    """Of `names`, the module-level functions that mention a reporting symbol.

    Closes the WRAPPED bypass (`def _h(): return sanity_regime_genuine_ic`,
    then `overall_pass = _h() and ...`) without merging the callee's locals
    into the caller's slice. Follows calls transitively, so a two-hop wrapper
    is caught too.
    """
    tree = ast.parse(src)
    # CLASSES too. [codex on bt#106] `class _C: def verdict(self): return
    # sanity_regime_genuine_ic` then `_c = _C(); overall_pass = _c.verdict()
    # and ...` is a normal refactoring shape, not a syntax corner, and the
    # slice already resolves `_c -> _C`. Mapping the class NAME to its whole
    # ClassDef scans every method body, so the wrapper check sees it.
    functions = {n.name: n for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef))}
    guilty: set[str] = set()
    seen: set[str] = set()
    frontier = [n for n in names if n in functions]
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        body_names = _plain_names(functions[name])
        if body_names & set(_REPORTING_SYMBOLS):
            guilty.add(name)
        frontier.extend(n for n in body_names if n in functions and n not in seen)
    return guilty


def _verdict_expressions(src: str) -> list[tuple[str, ast.AST]]:
    """(label, expression) for everything that PRODUCES the verdict in `main`.

    [codex on bt#106] A line-based scan was wrong twice over: `wf_meta = {...}`
    is ONE multi-line statement that legitimately contains both
    `"passed": overall_pass` and `"sanity_regime_genuine_ic":
    regime_genuine_ic_summary(...)`, so a statement-level rule would fail on
    correct code while a line-level rule missed the statement entirely. The
    right subject is neither the line nor the statement: it is the EXPRESSION
    the verdict is computed from.
    """
    tree = ast.parse(src)
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    found: list[tuple[str, ast.AST]] = []
    for node in ast.walk(main):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "overall_pass":
                    found.append(("overall_pass = <expr>", node.value))
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "exit"):
            for arg in node.args:
                # Only the exit that carries the VERDICT. `main` also has plain
                # error exits (`sys.exit(1)` on a refused precondition); those
                # are not the verdict and pulling them in would drag unrelated
                # locals into the slice.
                if "overall_pass" in _plain_names(arg):
                    found.append(("sys.exit(<verdict>)", arg))
    return found


def _assert_verdict_untainted(src: str, exprs: list) -> None:
    """THE main-guard assertion: nothing the verdict DEPENDS ON, transitively,
    may be a reporting symbol. Shared by the live check and every regression
    below, so a regression cannot pass while the check has stopped checking."""
    deps = _verdict_dependencies(src, exprs)
    direct = sorted(deps & set(_REPORTING_SYMBOLS))
    assert not direct, (
        f"{direct} is in the verdict's dependency slice — a reporting surface "
        f"must not decide anything (transitively)")
    wrapped = sorted(_functions_referencing_reporting(src, deps))
    assert not wrapped, (
        f"the verdict calls {wrapped}, which reference a reporting symbol — a "
        f"reporting surface must not decide anything (through a wrapper)")


def test_the_verdict_ASSEMBLY_in_main_is_also_clean():
    """The span the first guard list omitted. `main` legitimately mentions the
    reporting symbols — it is where they are STAMPED — so the rule is on the
    verdict EXPRESSION: what `overall_pass` is computed from, and what
    `sys.exit` is handed, may not reference a reporting symbol."""
    src = _runner_source()
    exprs = _verdict_expressions(src)
    labels = [lbl for lbl, _ in exprs]
    assert any("overall_pass =" in lbl for lbl in labels), labels
    assert any("sys.exit" in lbl for lbl in labels), labels
    _assert_verdict_untainted(src, exprs)


def test_the_main_guard_REJECTS_a_planted_reference_in_the_VERDICT_EXPRESSION():
    """Plant into the expression the verdict is computed from and require the
    same assertion path to RAISE — including via the multi-line `wf_meta`
    statement shape that defeated the line-based version."""
    src = _runner_source()
    planted = src.replace(
        "    overall_pass = _compute_overall_pass(",
        "    overall_pass = sanity_regime_genuine_ic and _compute_overall_pass(", 1)
    assert planted != src, "the plant did not apply — the regression is vacuous"
    with pytest.raises(AssertionError, match="must not decide anything"):
        _assert_verdict_untainted(planted, _verdict_expressions(planted))


def test_the_main_guard_TOLERATES_the_legitimate_wf_meta_statement():
    """Anti-false-positive: `wf_meta = {...}` holds BOTH `"passed": overall_pass`
    and the reporting summary in one statement. That is correct code and the
    guard must not reject it — which is why the subject is the expression, not
    the statement."""
    src = _runner_source()
    assert '"passed": overall_pass' in src
    assert '"sanity_regime_genuine_ic": regime_genuine_ic_summary(' in src
    exprs = _verdict_expressions(src)          # must not raise
    assert exprs, "the extractor found no verdict expression at all"


def test_the_extractor_stops_before_the_next_functions_DECORATOR():
    """[codex on bt#106] `runner.py` has no decorators today, so this pins the
    behaviour rather than the current file."""
    src = "\ndef first():\n    return 1\n\n\n@dec\ndef second():\n    return 2\n"
    body = _function_body(src, "first")
    assert body.rstrip().endswith("return 1"), repr(body)
    assert "@dec" not in body and "second" not in body


def test_the_main_guard_CATCHES_an_ALIASED_reporting_symbol():
    """[codex on bt#106] `helper = sanity_regime_genuine_ic` then
    `overall_pass = helper and ...` — the verdict subtree names only `helper`."""
    src = _runner_source().replace(
        "    overall_pass = _compute_overall_pass(",
        "    _vh = sanity_regime_genuine_ic\n"
        "    overall_pass = _vh and _compute_overall_pass(", 1)
    with pytest.raises(AssertionError, match="must not decide anything"):
        _assert_verdict_untainted(src, _verdict_expressions(src))


def test_the_main_guard_CATCHES_a_WRAPPED_reporting_symbol():
    """`def _h(): return sanity_regime_genuine_ic` then
    `overall_pass = _h() and ...` — the subtree names only `_h`."""
    src = _runner_source()
    src = src.replace("\ndef main(",
                      "\ndef _reporting_verdict_helper():\n"
                      "    return sanity_regime_genuine_ic\n\n\ndef main(", 1)
    src = src.replace(
        "    overall_pass = _compute_overall_pass(",
        "    overall_pass = _reporting_verdict_helper() and _compute_overall_pass(", 1)
    with pytest.raises(AssertionError, match="must not decide anything"):
        _assert_verdict_untainted(src, _verdict_expressions(src))


def test_the_main_guard_CATCHES_a_reporting_symbol_behind_an_OBJECT_METHOD():
    """[codex on bt#106] A normal refactoring shape, not a syntax corner:
    `class _C: def verdict(self): return <reporting>` then
    `_c = _C(); overall_pass = _c.verdict() and ...`."""
    src = _runner_source()
    src = src.replace("\ndef main(",
                      "\nclass _VerdictCarrier:\n"
                      "    def verdict(self):\n"
                      "        return sanity_regime_genuine_ic\n\n\ndef main(", 1)
    src = src.replace(
        "    overall_pass = _compute_overall_pass(",
        "    _carrier = _VerdictCarrier()\n"
        "    overall_pass = _carrier.verdict() and _compute_overall_pass(", 1)
    with pytest.raises(AssertionError, match="must not decide anything"):
        _assert_verdict_untainted(src, _verdict_expressions(src))


def test_the_ACCEPTED_CORNERS_are_written_down_not_forgotten():
    """`for`, `with ... as` and a separate-statement walrus rebind are outside
    the documented scope (REBINDING assignments). Codex judged them acceptable
    for a test-level guard; that judgement is recorded here so it is a decision
    rather than an oversight, and so widening the claim later has to confront
    it."""
    doc = (Path(__file__).resolve().parent.parent / "doc" / "progress"
           / "2026-08-05-regime-genuine-ic-in-verdict.md").read_text()
    flat = " ".join(doc.split())
    assert "ACCEPTED CORNERS" in flat
    for corner in ("for", "with", "walrus"):
        assert corner in flat


def test_the_slice_is_a_real_slice_not_the_whole_module():
    """Anti-false-positive AND anti-vacuity. A slice that swallowed the module
    would reject every future edit and get switched off; an empty one would
    prove nothing. It must contain the verdict's real inputs and NOT the
    reporting surface."""
    src = _runner_source()
    deps = _verdict_dependencies(src, _verdict_expressions(src))
    for real_input in ("overall_pass", "_compute_overall_pass", "wf_result",
                       "sanity_result", "parity_result"):
        assert real_input in deps, real_input
    for symbol in _REPORTING_SYMBOLS:
        assert symbol not in deps, symbol
    total = len({n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)})
    assert len(deps) < total / 2, (len(deps), total)
