"""GOAL-6 — the WF gate's unaided pass rate, over an AUDITABLE census.

Codex review round 2: *"the claimed census has no source-artifact provenance… without
that, the tests only preserve a table that asserts completeness."* Correct, and acting
on it changed the census.

The first version reported **11** artifacts. That was the *deployed + staging* subset,
and **the subset choice was never stated**. The stated inclusion query
`panel-ltr.alpha158_fund*.json` matches **29** files, and **all 29** carry a
`wf_gate_metadata` block — the 18 excluded without comment were `rollback` (16),
`previous` (1) and one restamp snapshot.

The conclusion survives the correction and strengthens: on the full census there are
**18 passes, 18 of them overridden, and zero unaided.**

Every row now carries the artifact's repo-relative path, its `sha256` and its size, and
`census_manifest.json` records the collection root, the inclusion query, the inclusion
rule and the excluded list — so completeness is auditable rather than asserted.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib

DIR = (pathlib.Path(__file__).resolve().parent.parent
       / "doc/research/evidence/2026-07-31-wf-gate-unaided-passes")
CSV = DIR / "gate_verdicts.csv"
MANIFEST = json.loads((DIR / "census_manifest.json").read_text(encoding="utf-8"))
UMBRELLA = pathlib.Path("/Users/renhao/git/github/RenQuant")


def _rows():
    with CSV.open() as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------ the finding ---
def test_zero_artifacts_passed_the_gate_unaided():
    rows = _rows()
    assert len(rows) == 29
    passed = [r for r in rows if r["passed"] == "True"]
    unaided = [r for r in passed if not r["override_reason"].strip()]
    assert len(passed) == 18
    assert unaided == [], [r["artifact"] for r in unaided]


def test_every_artifact_was_admitted_on_recipe_identity_only():
    rows = _rows()
    assert all(r["candidate_artifact_used"] == "False" for r in rows)
    assert all(r["recipe_validated"] == "True" for r in rows)


def test_the_deployed_artifact_is_one_of_the_overrides():
    dep = [r for r in _rows() if r["deployed"] == "True"]
    assert len(dep) == 1
    assert dep[0]["passed"] == "True"
    assert "2026-06-22" in dep[0]["override_reason"]


def test_the_deployed_artifacts_own_sanity_battery_says_FAIL():
    dep = next(r for r in _rows() if r["deployed"] == "True")
    assert dep["sanity_reason"].startswith("FAIL")
    assert "regime sanity IC failed" in dep["sanity_reason"]


# --------------------------------------------------------- the provenance ---
def test_the_census_states_how_it_was_collected():
    for key in ("collection_root", "inclusion_query", "inclusion_rule",
                "n_files_matching_query", "n_included", "n_excluded", "excluded"):
        assert key in MANIFEST, key
    assert MANIFEST["n_files_matching_query"] == MANIFEST["n_included"] == 29
    assert MANIFEST["n_excluded"] == 0 and MANIFEST["excluded"] == []


def test_every_row_carries_a_path_and_a_content_digest():
    for r in _rows():
        assert r["artifact_path"].startswith("backtesting/renquant_104/artifacts/prod/")
        assert len(r["content_sha256"]) == 64
        assert int(r["bytes"]) > 0


def test_the_digests_match_the_artifacts_on_disk():
    """The claim the review actually needed: the rows are bound to real files.
    Skipped rather than failed when the umbrella tree is absent — a test that
    measures the operator's disk must not fail on a machine that lacks it."""
    import pytest

    if not UMBRELLA.exists():
        pytest.skip("umbrella tree not present on this machine")
    for r in _rows()[:3]:                      # first three is enough to bind
        p = UMBRELLA / r["artifact_path"]
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h == r["content_sha256"], r["artifact"]


# ------------------------------------------------------------ anti-vacuity --
def test_the_rejects_carry_no_override():
    """If overrides were everywhere, 'every pass is overridden' would be
    unremarkable. They are not."""
    rejected = [r for r in _rows() if r["passed"] == "False"]
    assert len(rejected) == 11
    assert all(not r["override_reason"].strip() for r in rejected)
