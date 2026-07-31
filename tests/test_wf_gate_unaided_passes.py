"""GOAL-6 — the WF gate's unaided pass rate, pinned so the finding cannot rot.

Measured 2026-07-31 over every `panel-ltr.alpha158_fund` artifact carrying
`wf_gate_metadata`: 11 artifacts, 2 with `passed=True`, and BOTH of those carry an
operator override. **Zero unaided passes.**

Read from a frozen CSV, never from the live tree: that tree is a production surface
and a test reaching into it would couple this suite to one operator's disk and
re-measure a moving target.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / ("doc/research/evidence/2026-07-31-wf-gate-unaided-passes/"
              "gate_verdicts.csv")


def _rows():
    with CSV.open() as fh:
        return list(csv.DictReader(fh))


def test_zero_artifacts_passed_the_gate_unaided():
    rows = _rows()
    assert len(rows) == 11
    passed = [r for r in rows if r["passed"] == "True"]
    unaided = [r for r in passed if not r["override_reason"].strip()]
    assert len(passed) == 2
    assert unaided == [], unaided          # THE finding


def test_the_deployed_artifact_is_one_of_the_overrides():
    dep = [r for r in _rows() if r["deployed"] == "True"]
    assert len(dep) == 1
    assert dep[0]["passed"] == "True"
    assert "2026-06-22" in dep[0]["override_reason"]
    assert dep[0]["diagnostic_only"] == "True"


def test_the_deployed_artifacts_own_sanity_battery_says_FAIL():
    """Corrects an earlier claim of mine that it 'passes'. It passes ONE enforced
    placebo sub-criterion; its overall sanity verdict is FAIL on regime sanity IC."""
    dep = next(r for r in _rows() if r["deployed"] == "True")
    assert dep["sanity_reason"].startswith("FAIL")
    assert "regime sanity IC failed" in dep["sanity_reason"]


def test_every_artifact_was_admitted_on_recipe_identity_only():
    """The #83 shape, present on all 11 — the gate never scored a candidate's own
    booster. Kept beside the override finding because they compound."""
    rows = _rows()
    assert all(r["candidate_artifact_used"] == "False" for r in rows)
    assert all(r["recipe_validated"] == "True" for r in rows)


def test_the_nine_rejects_carry_no_override():
    """Anti-vacuity: if overrides were everywhere, 'both passes are overrides'
    would be unremarkable. They are not."""
    rejected = [r for r in _rows() if r["passed"] == "False"]
    assert len(rejected) == 9
    assert all(not r["override_reason"].strip() for r in rejected)
