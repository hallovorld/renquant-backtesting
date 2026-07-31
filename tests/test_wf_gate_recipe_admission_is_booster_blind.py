"""GOAL-6 — the WF gate's admission scope is BOOSTER-BLIND, executed rather than read.

WHY THIS FILE EXISTS
--------------------
On the production path `strategy_config.walkforward.enabled` is true, and
`runner.inspect_artifact_usage` returns `candidate_artifact_used: False`
**unconditionally** there — a current artifact cannot be replayed into old sim
windows without look-ahead, so the gate validates the retraining *recipe* instead.

Admission scope is then decided by

    validation_scope_ok = candidate_artifact_used or recipe_validated
                        = False                  or recipe_validated

so on the production path admission rides **entirely** on `recipe_validated`.

And `recipe_projection` — the thing the fingerprint is taken over — contains
`kind`, `feature_cols`, `feature_norm_kind`, the feature-source contract KEYS,
`label_col`, `lookahead_days` and the semantic learner params.  **It contains no
learned parameter.**  Two artifacts trained from the same recipe on different data,
or to different quality, therefore carry the *same* fingerprint.

THE CLAIM, TWICE CORRECTED — READ THIS BEFORE CITING THE FILE
------------------------------------------------------------
1. I first wrote *"no gate anywhere in this path scores the candidate's own
   weights."*
2. I withdrew it: `run_sanity_battery` **does** contain a branch that runs
   `PanelScorer.load(artifact_path)` and `scorer.score(X)` (runner.py:2772-2779),
   so a candidate-scoring path exists.
3. **That branch does not execute on the production configuration.** It is the
   branch that stamps `sanity_eval_scope: "static_artifact"`. Measured across
   **29 of 29** production artifacts carrying `wf_gate_metadata`:
   `sanity_eval_scope == "walkforward_manifest"` — the candidate-scoring branch
   ran **zero** times
   `[VERIFIED — 本次实测 2026-08-01, docs/research/evidence/2026-08-01-sanity-scope/]`.

So the accurate statement is neither of the first two: **the gate HAS a
candidate-scoring path and, as configured in production, never takes it.** Reading
the source gave the opposite answer to executing it, which is why the census below
is the primary artifact and the source reading is secondary.

These are CHARACTERISATION tests.  They pin the behaviour that is in force today so
it cannot change silently.  **They do not endorse it** — the gap they describe is
what blocks GOAL-6's evaluation path and GOAL-4's ensemble admission alike.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from renquant_backtesting.wf_gate.recipe_match import (
    manifest_recipe_usage,
    recipe_fingerprint,
    recipe_projection,
)


def _artifact(booster: str, *, features=("f1", "f2", "f3"), kind="panel_ltr_xgboost",
              lookahead=60, params=None) -> dict:
    """A minimal artifact whose recipe fields are explicit and whose learned
    payload (`booster`) is the ONLY thing varied between the two candidates."""
    return {
        "kind": kind,
        "feature_cols": list(features),
        "feature_norm_kind": ["zscore"] * len(features),
        "feature_source_contract": {"raw": "prose that may be edited", "panel": "…"},
        "label_col": "fwd_60d_excess",
        "lookahead_days": lookahead,
        "params": params or {"eta": 0.05, "max_depth": 4, "seed": 7},
        # everything below is the LEARNED payload — not in recipe_projection
        "booster": booster,
        "trained_date": "2026-07-30",
        "n_trees": 400,
    }


def _write(tmp: Path, name: str, payload: dict) -> Path:
    p = tmp / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _manifest(tmp: Path, artifact_paths, name="walkforward_manifest.json") -> Path:
    p = tmp / name
    p.write_text(json.dumps(
        {"retrains": [{"artifact_uri": str(a)} for a in artifact_paths]}),
        encoding="utf-8")
    return p


# ------------------------------------------------------- the projection ------
def test_recipe_projection_contains_no_learned_parameter():
    """The fingerprint's inputs, enumerated. If a learned field ever enters this
    set the tests below stop describing the system and this one fails first."""
    proj = recipe_projection(_artifact("anything"))
    assert set(proj) == {
        "kind", "feature_cols", "feature_norm_kind",
        "feature_source_contract_keys", "label_col", "lookahead_days", "params",
    }
    flat = json.dumps(proj)
    assert "booster" not in flat
    assert "n_trees" not in flat
    assert "trained_date" not in flat


def test_two_boosters_from_one_recipe_share_a_fingerprint():
    good = _artifact("BOOSTER-TRAINED-ON-REAL-LABELS")
    junk = _artifact("BOOSTER-TRAINED-ON-SHUFFLED-LABELS")
    assert good["booster"] != junk["booster"]
    assert recipe_fingerprint(good) == recipe_fingerprint(junk)


def test_prose_edits_to_the_source_contract_do_not_move_the_fingerprint():
    """Behavioural anchor 2026-05-27 — keys are hashed, values are not."""
    a = _artifact("b")
    b = _artifact("b")
    b["feature_source_contract"] = {"raw": "COMPLETELY DIFFERENT PROSE", "panel": "x"}
    assert recipe_fingerprint(a) == recipe_fingerprint(b)


# ------------------------------------------- admission on the WF path --------
def test_a_different_booster_is_admitted_on_another_boosters_evidence(tmp_path):
    """The headline. The manifest contains artifact A; the CANDIDATE is B."""
    a = _write(tmp_path, "a.json", _artifact("BOOSTER-A"))
    b = _write(tmp_path, "b.json", _artifact("BOOSTER-B"))
    man = _manifest(tmp_path, [a])

    usage = manifest_recipe_usage(man, b, strategy_dir=tmp_path)
    assert usage["recipe_validated"] is True

    # …and this is what the runner then computes, verbatim:
    candidate_artifact_used = False          # hardcoded on the walkforward path
    validation_scope_ok = candidate_artifact_used or bool(usage["recipe_validated"])
    assert validation_scope_ok is True


@pytest.mark.parametrize("mutate", [
    lambda a: a.update(feature_cols=["f1", "f2"]),          # dropped a feature
    lambda a: a.update(kind="hf_patchtst"),                 # different model kind
    lambda a: a.update(lookahead_days=20),                  # different label horizon
    lambda a: a.update(params={"eta": 0.30, "max_depth": 4, "seed": 7}),
])
def test_a_genuinely_different_RECIPE_is_refused(tmp_path, mutate):
    """ANTI-VACUITY. Without this, a `recipe_validated` that is hardcoded True
    would pass the headline test and the file would prove nothing."""
    a = _write(tmp_path, "a.json", _artifact("BOOSTER-A"))
    cand = _artifact("BOOSTER-B")
    mutate(cand)
    b = _write(tmp_path, "b.json", cand)
    man = _manifest(tmp_path, [a])

    usage = manifest_recipe_usage(man, b, strategy_dir=tmp_path)
    assert usage["recipe_validated"] is False
    assert not (False or bool(usage["recipe_validated"]))


def test_one_mismatching_sample_refuses_the_whole_manifest(tmp_path):
    """`all_match` — a single divergent manifest row must sink admission."""
    same = _write(tmp_path, "same.json", _artifact("BOOSTER-A"))
    other = _artifact("BOOSTER-C")
    other["feature_cols"] = ["f1"]
    diff = _write(tmp_path, "diff.json", other)
    man = _manifest(tmp_path, [same, diff])
    usage = manifest_recipe_usage(man, same, strategy_dir=tmp_path)
    assert usage["recipe_validated"] is False


def test_a_missing_manifest_artifact_is_a_refusal_not_a_skip(tmp_path):
    a = _write(tmp_path, "a.json", _artifact("BOOSTER-A"))
    man = _manifest(tmp_path, [tmp_path / "does-not-exist.json"])
    usage = manifest_recipe_usage(man, a, strategy_dir=tmp_path)
    assert usage["recipe_validated"] is False


def test_an_absent_manifest_is_a_refusal_not_a_skip(tmp_path):
    a = _write(tmp_path, "a.json", _artifact("BOOSTER-A"))
    usage = manifest_recipe_usage(tmp_path / "nope.json", a, strategy_dir=tmp_path)
    assert usage["recipe_validated"] is False


def test_an_empty_manifest_is_a_refusal_not_a_skip(tmp_path):
    a = _write(tmp_path, "a.json", _artifact("BOOSTER-A"))
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"retrains": []}), encoding="utf-8")
    usage = manifest_recipe_usage(man, a, strategy_dir=tmp_path)
    assert usage["recipe_validated"] is False


def test_the_reported_fingerprint_is_the_candidates_own(tmp_path):
    """So a reader of the gate's metadata can reproduce the comparison."""
    payload = _artifact("BOOSTER-A")
    a = _write(tmp_path, "a.json", payload)
    man = _manifest(tmp_path, [a])
    usage = manifest_recipe_usage(man, a, strategy_dir=tmp_path)
    assert usage["candidate_recipe_fingerprint"] == recipe_fingerprint(payload)
    assert usage["manifest_rows_checked"] == 1


# ================= WHICH BRANCH ACTUALLY RAN, from real artifacts ============
# codex on #89 asked for an EXECUTION test rather than a re-reading of the scope
# expression. This reads the branch marker out of every production artifact: the
# candidate-scoring branch stamps "static_artifact", the manifest branch stamps
# "walkforward_manifest". The census records which one the gate actually took.

import pathlib
import csv as _csv
import json as _json
import pytest as _pytest

_EVID = (pathlib.Path(__file__).resolve().parent.parent
         / "docs/research/evidence/2026-07-31-sanity-scope")


def _census():
    with (_EVID / "sanity_scope_census.csv").open() as fh:
        return list(_csv.DictReader(fh))


def test_the_candidate_scoring_branch_ran_ZERO_times_in_production():
    """29 artifacts, 29 scoped, `static_artifact` zero times.

    RETRACTED 2026-07-31: an earlier version of this test asserted **14** rows and
    stated that the hand-built 29-row census had "asserted an observation nobody made"
    fifteen times. **That was false, and it was an accusation of fabricated evidence.**

    The cause was in the tool, not the corpus. `_scopes` read `wf_gate_metadata` at the
    TOP LEVEL of each artifact. Measured against the corpus: **all 29 artifacts carry
    `metadata.wf_gate_metadata`**, and 14 of them ALSO carry a legacy top-level copy.
    The tool was reading the legacy key, found it on 14, and called the other 15
    invented. A checker looking in the wrong place does not discover that its subjects
    are missing -- it discovers that it is looking in the wrong place.

    The original 29-row census was RIGHT. The finding is unchanged and now rests on 29
    measured observations rather than 12.
    """
    rows = _census()
    assert len(rows) == 29
    assert all(r["sanity_eval_scope"] == "walkforward_manifest" for r in rows), \
        [r["artifact"] for r in rows if r["sanity_eval_scope"] != "walkforward_manifest"]
    assert [r for r in rows if r["sanity_eval_scope"] == "static_artifact"] == []


def test_every_row_records_WHICH_KEY_the_scope_came_from():
    """The duplication that caused the false accusation is now visible in the evidence.

    Reading the canonical key silently would fix the count and hide the reason. Each row
    names its source, so a reader can see that all 29 answered on
    `metadata.wf_gate_metadata` -- and would see immediately if any row fell back to the
    legacy top-level copy.
    """
    rows = _census()
    assert all(r["scope_source"] == "metadata.wf_gate_metadata" for r in rows), \
        {r["artifact"]: r["scope_source"] for r in rows
         if r["scope_source"] != "metadata.wf_gate_metadata"}


def test_the_two_copies_of_the_gate_block_are_not_assumed_consistent():
    """Where both exist they DISAGREE on 2 of 14 -- so which key you read matters.

    The legacy top-level copy carries no `sanity_eval_scope` on
    `panel-ltr.alpha158_fund.previous.json` and
    `panel-ltr.alpha158_fund.weekly_rollback_2026-07-06.json`, while the canonical block
    records `walkforward_manifest` for both. This pins that asymmetry so "just read
    either one" cannot creep back in.
    """
    import json as _j
    root = pathlib.Path(_j.loads((_EVID / "census_manifest.json")
                                 .read_text(encoding="utf-8"))["collection_root"])
    base = pathlib.Path.home() / "git" / "github" / "RenQuant" / root
    if not base.is_dir():
        _pytest.skip(f"artifact corpus not present at {base}")
    both = disagree = 0
    for f in sorted(base.glob("panel-ltr.alpha158_fund*.json")):
        d = _j.loads(f.read_text(encoding="utf-8"))
        top = d.get("wf_gate_metadata")
        canon = (d.get("metadata") or {}).get("wf_gate_metadata")
        if isinstance(top, dict) and isinstance(canon, dict):
            both += 1
            if top.get("sanity_eval_scope") != canon.get("sanity_eval_scope"):
                disagree += 1
    assert both == 14, both
    assert disagree == 2, disagree


def test_the_candidate_scoring_branch_nevertheless_EXISTS():
    """Both halves are needed. Without this the file would restate claim (1) and
    without the census it would restate claim (2)."""
    runner = (pathlib.Path(__file__).resolve().parent.parent / "src"
              / "renquant_backtesting" / "wf_gate" / "runner.py").read_text(encoding="utf-8")
    assert "PanelScorer.load(artifact_path)" in runner
    assert 'sanity_eval_scope": "static_artifact"' in runner


def test_the_census_states_how_it_was_collected():
    man = _json.loads((_EVID / "census_manifest.json").read_text(encoding="utf-8"))
    for k in ("collection_root", "inclusion_query", "inclusion_rule", "n_included"):
        assert k in man, k
    assert man["n_included"] == len(_census())


def test_every_census_row_carries_a_path_and_a_digest():
    """The path is relative to the collection root the manifest names -- an absolute
    path would pin the census to one machine and make `--verify` read as "everything
    is missing" anywhere else."""
    for r in _census():
        assert r["artifact_path"] and not r["artifact_path"].startswith("/")
        assert r["artifact_path"].endswith(".json")
        assert len(r["content_sha256"]) == 64


def test_the_census_was_produced_by_the_COMMITTED_command():
    """Codex's ask: a reproducible census command that reads the declared source
    artifacts and verifies their digests. The manifest's schema and inclusion rule are
    the tool's own constants, so a hand-edited CSV that drifts from the tool is
    detectable here rather than by inspection."""
    import importlib.util
    import sys as _sys
    root = pathlib.Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_census_tool", root / "tools" / "sanity_scope_census.py")
    tool = importlib.util.module_from_spec(spec)
    _sys.modules[spec.name] = tool
    spec.loader.exec_module(tool)
    man = _json.loads((_EVID / "census_manifest.json").read_text(encoding="utf-8"))
    assert man["schema"] == tool.SCHEMA
    assert man["inclusion_query"] == tool.INCLUSION_QUERY
    assert man["inclusion_rule"] == tool.INCLUSION_RULE
    assert man["n_included"] == len(_census())
    with (_EVID / tool.CSV_NAME).open() as fh:
        assert _csv.DictReader(fh).fieldnames == tool.FIELDS
