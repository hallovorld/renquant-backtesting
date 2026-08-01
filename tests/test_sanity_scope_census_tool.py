"""The census instrument, exercised against CONTROLLED fixtures.

Reviewed `[codex on #89]`: *"add a reproducible census command that reads the declared
source artifacts, verifies their digests and inclusion count, and make the test exercise
it against controlled fixtures."*

Two halves, and the second is the one that matters. `--emit` reading the real artifact
store is an observation of one machine at one moment; it cannot be a test, because the
store legitimately changes and a machine without it would go green for the wrong reason.
What CAN be tested is that the instrument reports each way the CSV and the disk can
disagree — a vanished path, changed bytes, a scope that moved, an artifact the census
never listed.

Each fixture below is a way the committed census could be silently wrong.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "sanity_scope_census", ROOT / "tools" / "sanity_scope_census.py")
CEN = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = CEN
_spec.loader.exec_module(CEN)


def _artifact(root, name, *, sanity_scope="walkforward_manifest",
              eval_scope="walkforward_manifest", stamped=True, extra=None):
    root.mkdir(parents=True, exist_ok=True)
    payload = {"kind": "panel_ltr_xgboost", "booster_raw_json": extra or "{}"}
    if stamped:
        payload["wf_gate_metadata"] = {"sanity_eval_scope": sanity_scope,
                                       "eval_scope": eval_scope}
    p = root / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _emit(tmp_path, root, out):
    CEN.main(["--emit", "--root", str(root), "--out", str(out)])
    return list(CEN.csv.DictReader((out / CEN.CSV_NAME).open()))


def test_it_reads_the_scope_out_of_each_artifact_not_out_of_a_filename(tmp_path):
    root, out = tmp_path / "prod", tmp_path / "evid"
    _artifact(root, "panel-ltr.alpha158_fund.json")
    _artifact(root, "panel-ltr.alpha158_fund.b.json", sanity_scope="static_artifact")
    rows = _emit(tmp_path, root, out)
    got = {r["artifact"]: r["sanity_eval_scope"] for r in rows}
    assert got == {"panel-ltr.alpha158_fund.json": "walkforward_manifest",
                   "panel-ltr.alpha158_fund.b.json": "static_artifact"}


def test_an_UNSTAMPED_artifact_is_excluded_rather_than_counted_as_blank(tmp_path):
    """An artifact the gate never ran is not evidence about which branch executed.
    Recording it with an empty scope would put a non-observation in the denominator."""
    root, out = tmp_path / "prod", tmp_path / "evid"
    _artifact(root, "panel-ltr.alpha158_fund.json")
    _artifact(root, "panel-ltr.alpha158_fund.never-gated.json", stamped=False)
    rows = _emit(tmp_path, root, out)
    assert [r["artifact"] for r in rows] == ["panel-ltr.alpha158_fund.json"]
    man = json.loads((out / CEN.MANIFEST_NAME).read_text())
    assert man["n_included"] == 1
    assert man["inclusion_rule"] and man["collection_root"] == str(root)


def test_the_digest_is_of_the_BYTES_and_verify_accepts_them_unchanged(tmp_path):
    root, out = tmp_path / "prod", tmp_path / "evid"
    p = _artifact(root, "panel-ltr.alpha158_fund.json")
    rows = _emit(tmp_path, root, out)
    assert rows[0]["content_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert CEN.verify(root, out / CEN.CSV_NAME)["ok"] is True


def test_verify_CATCHES_bytes_that_changed_under_the_census(tmp_path):
    """The whole point of committing a digest. A retrain rewriting an artifact in place
    is invisible to a census that lists filenames."""
    root, out = tmp_path / "prod", tmp_path / "evid"
    _artifact(root, "panel-ltr.alpha158_fund.json")
    _emit(tmp_path, root, out)
    _artifact(root, "panel-ltr.alpha158_fund.json", extra='{"retrained":1}')
    result = CEN.verify(root, out / CEN.CSV_NAME)
    assert result["digest_changed"] == ["panel-ltr.alpha158_fund.json"]
    assert result["ok"] is False


def test_verify_CATCHES_an_artifact_that_vanished(tmp_path):
    root, out = tmp_path / "prod", tmp_path / "evid"
    p = _artifact(root, "panel-ltr.alpha158_fund.json")
    _emit(tmp_path, root, out)
    p.unlink()
    result = CEN.verify(root, out / CEN.CSV_NAME)
    assert result["missing"] == ["panel-ltr.alpha158_fund.json"]
    assert result["ok"] is False


def test_verify_CATCHES_an_artifact_the_census_never_listed(tmp_path):
    """Inclusion count, in the direction that matters: a census taken before a new
    artifact landed reads as complete unless something compares it to the store."""
    root, out = tmp_path / "prod", tmp_path / "evid"
    _artifact(root, "panel-ltr.alpha158_fund.json")
    _emit(tmp_path, root, out)
    _artifact(root, "panel-ltr.alpha158_fund.new.json", sanity_scope="static_artifact")
    result = CEN.verify(root, out / CEN.CSV_NAME)
    assert result["uncensused_on_disk"] == ["panel-ltr.alpha158_fund.new.json"]


def test_verify_CATCHES_a_scope_that_moved_without_the_digest_moving(tmp_path):
    """Anti-vacuity for the digest check itself: `verify` must read the SCOPE back out
    of the artifact, not trust the CSV's copy of it. Constructed by editing the CSV,
    which is exactly how a transcription error looks."""
    root, out = tmp_path / "prod", tmp_path / "evid"
    _artifact(root, "panel-ltr.alpha158_fund.json")
    _emit(tmp_path, root, out)
    csv_path = out / CEN.CSV_NAME
    csv_path.write_text(csv_path.read_text().replace("walkforward_manifest",
                                                     "static_artifact"))
    result = CEN.verify(root, csv_path)
    assert result["scope_drift"] == ["panel-ltr.alpha158_fund.json"]
    assert result["ok"] is False


def test_a_nonexistent_root_does_not_read_as_a_clean_census(tmp_path):
    """A host without the artifact store must not report a verified census. Every
    committed row is missing, which is the honest answer."""
    out = tmp_path / "evid"
    _artifact(tmp_path / "prod", "panel-ltr.alpha158_fund.json")
    _emit(tmp_path, tmp_path / "prod", out)
    result = CEN.verify(tmp_path / "gone", out / CEN.CSV_NAME)
    assert result["ok"] is False
