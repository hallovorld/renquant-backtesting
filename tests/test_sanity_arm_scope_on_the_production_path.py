"""Which booster the sanity arm actually scores with, on the production path.

backtesting#89 landed a CORRECTION saying *"the gate DOES score the candidate's booster
— in the sanity arm, on its own validation partition."* **That is false on the
production path**, and this module pins the chain that shows it.

`PanelScorer.load(artifact_path)` — the candidate — exists, but only inside the branch
that stamps `sanity_eval_scope: "static_artifact"`. The merged census in this same PR
measured that branch running **0 times in 29 production artifacts**. Citing a line inside
an unreachable branch as proof that the candidate is scored is twin-registry R7's exact
shape: *a branch no caller reaches is dead code wearing a docstring.*

These assertions are structural (AST + source), not behavioural, because running the real
gate needs the artifact corpus. They fail if the dispatch changes.
"""

from __future__ import annotations

import ast
import pathlib

RUNNER = (pathlib.Path(__file__).resolve().parent.parent
          / "src/renquant_backtesting/wf_gate/runner.py")
SRC = RUNNER.read_text(encoding="utf-8")
TREE = ast.parse(SRC)


def _fn(name: str) -> ast.FunctionDef:
    for n in ast.walk(TREE):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise AssertionError(f"{name} not found in runner.py")


def test_the_production_path_declares_eval_scope_walkforward_manifest():
    """`walkforward.enabled` -> `eval_scope: "walkforward_manifest"`, and
    `candidate_artifact_used: False` alongside it."""
    assert '"eval_scope": "walkforward_manifest",' in SRC
    assert '"candidate_artifact_used": False,' in SRC


def test_run_sanity_battery_branches_on_that_scope():
    """The dispatch that decides which booster gets used."""
    fn = _fn("run_sanity_battery")
    body = ast.get_source_segment(SRC, fn) or ""
    assert 'artifact_usage.get("eval_scope") == "walkforward_manifest"' in body
    assert "if manifest_scope:" in body


def test_the_candidate_is_loaded_ONLY_under_the_static_artifact_scope():
    """The load of the CANDIDATE's booster and the `static_artifact` stamp are in the
    same branch — the one the census measured running zero times.

    If a future change moves `PanelScorer.load(artifact_path)` onto the manifest path,
    this fails, and that would be the fix this finding asks for.
    """
    fn = _fn("run_sanity_battery")
    seg = ast.get_source_segment(SRC, fn) or ""
    assert "PanelScorer.load(artifact_path)" in seg, "candidate load moved out entirely"
    # the candidate load must sit AFTER the manifest branch's early returns, in the else
    i_manifest = seg.index("if manifest_scope:")
    i_candidate = seg.index("PanelScorer.load(artifact_path)")
    assert i_candidate > i_manifest
    assert '"sanity_eval_scope": "static_artifact"' in seg


def test_the_manifest_branch_scores_from_the_MANIFEST_uris():
    """The production branch loads its scorer from a manifest URI, not from the
    candidate's path — this is the substance of the finding."""
    assert "PanelScorer.load(uri_path)" in SRC
    # and that helper stamps the production scope
    i_uri = SRC.index("PanelScorer.load(uri_path)")
    tail = SRC[i_uri:i_uri + 2000]
    assert '"sanity_eval_scope": "walkforward_manifest"' in tail


def test_the_census_still_shows_the_candidate_branch_never_ran():
    """Anti-drift on the measurement the conclusion rests on."""
    import csv
    p = (pathlib.Path(__file__).resolve().parent.parent
         / "docs/research/evidence/2026-07-31-sanity-scope/sanity_scope_census.csv")
    rows = list(csv.DictReader(p.open()))
    assert len(rows) == 29
    assert [r for r in rows if r["sanity_eval_scope"] == "static_artifact"] == []
    assert all(r["sanity_eval_scope"] == "walkforward_manifest" for r in rows)
