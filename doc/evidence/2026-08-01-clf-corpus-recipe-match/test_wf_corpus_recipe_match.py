"""Does clf's recipe have any walk-forward corpus? Zero of 85 — by the gate's criterion.

The GOAL-6 anchor says it does not. On 2026-07-31 I "corrected" that anchor using the GBDT
corpus's 43 folds and had to withdraw it — I had attached one lane's corpus to another by
DIRECTORY NAME. This settles it with `run_wf_gate._recipe_projection` instead.

The near-miss is the reason this is a tool: omitting `params` from the projection flips the
answer from 0/85 to 82/85. clf and prod agree on kind, all 172 feature_cols,
feature_norm_kind, label_col and lookahead_days; the entire difference is
`objective: binary:logistic` vs `rank:pairwise`.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

OPS = pathlib.Path(__file__).resolve().parent.parent / "ops" / "renquant104"
sys.path.insert(0, str(OPS))

import wf_corpus_recipe_match as M  # noqa: E402


def _art(tmp, name, **over):
    d = {"kind": "panel_ltr_xgboost", "feature_cols": ["a", "b"],
         "feature_norm_kind": ["z", "z"], "label_col": "fwd_60d_excess",
         "lookahead_days": 60, "params": {"objective": "rank:pairwise", "eta": 0.05}}
    d.update(over)
    p = tmp / name
    p.write_text(json.dumps(d))
    return p


# ------------------------------------------------------------- the near-miss --
def test_PARAMS_ALONE_decides_the_match():
    """Drop `params` and the answer inverts. It is the one nested-dict field of the six,
    and it is the only thing separating clf from prod."""
    a = {"kind": "k", "feature_cols": ["x"], "feature_norm_kind": ["z"],
         "label_col": "L", "lookahead_days": 60,
         "params": {"objective": "binary:logistic"}}
    b = dict(a, params={"objective": "rank:pairwise"})
    assert M.fingerprint(a) != M.fingerprint(b)
    assert M.first_difference(a, b) == ["params"]


def test_projection_includes_params():
    assert "params" in M.PROJECTION_FIELDS
    assert set(M.PROJECTION_FIELDS) == {
        "kind", "feature_cols", "feature_norm_kind", "label_col", "lookahead_days",
        "params"}


def test_the_derived_gate_field_is_declared_unchecked():
    """`feature_source_contract_keys` is computed inside the gate, not an artifact field.
    Silently omitting it would overstate this tool's coverage."""
    assert M.NOT_CHECKED == ("feature_source_contract_keys",)


# ------------------------------------------------------------------ matching --
def test_an_identical_recipe_matches(tmp_path):
    c = _art(tmp_path, "cand.json")
    (tmp_path / "corpus").mkdir()
    _art(tmp_path / "corpus", "f1.json")
    rep = M.survey(c, str(tmp_path / "corpus" / "*.json"))
    assert rep["n_folds"] == 1 and rep["n_matching"] == 1
    assert M.main(["--candidate", str(c),
                   "--corpus-glob", str(tmp_path / "corpus" / "*.json")]) == 0


def test_a_differing_OBJECTIVE_does_not_match_and_names_the_field(tmp_path):
    c = _art(tmp_path, "cand.json", params={"objective": "binary:logistic", "eta": 0.05})
    (tmp_path / "corpus").mkdir()
    _art(tmp_path / "corpus", "f1.json")
    rep = M.survey(c, str(tmp_path / "corpus" / "*.json"))
    assert rep["n_matching"] == 0
    assert rep["differing_field_sets"] == {"params": 1}
    assert M.main(["--candidate", str(c),
                   "--corpus-glob", str(tmp_path / "corpus" / "*.json")]) == 1


def test_a_companion_artifact_without_features_is_not_counted_as_a_NON_match(tmp_path):
    """Calibration companions live in these directories. Counting them as non-matching
    folds would inflate the denominator with things that were never scorers."""
    c = _art(tmp_path, "cand.json")
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "cal.json").write_text(
        json.dumps({"kind": "global_panel_calibration"}))
    _art(tmp_path / "corpus", "f1.json")
    rep = M.survey(c, str(tmp_path / "corpus" / "*.json"))
    assert rep["n_folds"] == 1


def test_an_UNREADABLE_fold_is_counted_separately_not_as_a_non_match(tmp_path):
    c = _art(tmp_path, "cand.json")
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "bad.json").write_text("{not json")
    _art(tmp_path / "corpus", "f1.json")
    rep = M.survey(c, str(tmp_path / "corpus" / "*.json"))
    assert rep["n_unreadable"] == 1 and rep["n_folds"] == 1 and rep["n_matching"] == 1


# ----------------------------------------------------------------- fail-closed --
def test_NO_folds_SKIPS_with_3_rather_than_reporting_no_corpus(tmp_path):
    """'No fold matched' and 'no fold was read' are different facts, and only the first
    is evidence about the candidate."""
    c = _art(tmp_path, "cand.json")
    assert M.survey(c, str(tmp_path / "nope" / "*.json"))["status"] == "no_folds"
    assert M.main(["--candidate", str(c),
                   "--corpus-glob", str(tmp_path / "nope" / "*.json")]) == 3


def test_an_unreadable_CANDIDATE_is_a_usage_error(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{nope")
    assert M.main(["--candidate", str(p), "--corpus-glob", str(tmp_path / "*.json")]) == 2


def test_the_note_records_that_DIRECTORY_NAME_is_not_used(tmp_path):
    c = _art(tmp_path, "cand.json")
    (tmp_path / "corpus").mkdir()
    _art(tmp_path / "corpus", "f1.json")
    rep = M.survey(c, str(tmp_path / "corpus" / "*.json"))
    assert "Directory name is NOT used" in rep["note"]
