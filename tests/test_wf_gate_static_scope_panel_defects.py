"""Three bounded defects that made static scope evaluate the WRONG panel (#84).

Static scope is the only path that scores a candidate's own booster --- measured:
68 metadata keys, 16 differ between a 296-ticker and a 292-ticker artifact, versus
4 in manifest scope where all four are echoed paths. But it read the training
contract at the ROOT while renquant-orchestrator#620 stamps it under `metadata`
(forced there because a root key is UNCLASSIFIED in
renquant_common.model_fingerprint, renquant-common#38). The lookup missed and the
gate silently fell back to the 292-ticker rawlabel corpus --- so it could not see a
universe extension even when it ran.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "wf_runner",
    Path(__file__).resolve().parent.parent
    / "src" / "renquant_backtesting" / "wf_gate" / "runner.py")
mod = importlib.util.module_from_spec(SPEC)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
try:
    SPEC.loader.exec_module(mod)
finally:
    sys.path.pop(0)


# --- defect 1: the contract is found wherever it is stamped ----------------

def test_metadata_nested_contract_is_found():
    """THE REGRESSION. #620 stamps under `metadata`; a root-only read missed it."""
    art = {"metadata": {"training_contract": {"dataset": "data/candidate_296.parquet"}}}
    assert mod.training_contract_dataset(art) == "data/candidate_296.parquet"


def test_root_contract_still_found_and_wins_over_nested():
    art = {"training_contract": {"dataset": "root.parquet"},
           "metadata": {"training_contract": {"dataset": "nested.parquet"}}}
    assert mod.training_contract_dataset(art) == "root.parquet"


def test_bare_dataset_key_is_the_last_resort():
    assert mod.training_contract_dataset({"dataset": "bare.parquet"}) == "bare.parquet"


def test_absent_contract_returns_None_not_a_default():
    """None must mean 'nothing declared', so the caller cannot mistake a missing
    contract for a validated one."""
    for art in ({}, {"metadata": {}}, {"training_contract": {}},
                {"metadata": {"training_contract": {}}}, {"training_contract": None}):
        assert mod.training_contract_dataset(art) is None, art


def test_the_old_root_only_read_would_have_failed_this():
    """Guards against the pin being vacuous: the pre-fix expression returns None
    on the shape #620 actually produces."""
    art = {"metadata": {"training_contract": {"dataset": "x.parquet"}}}
    legacy = (art.get("training_contract") or {}).get("dataset") or art.get("dataset")
    assert legacy is None
    assert mod.training_contract_dataset(art) == "x.parquet"


# --- defect 2: the fallback corpus's INVALID receipt is consulted ----------

def _panel(tmp_path: Path, name: str, cols: list[str]) -> Path:
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=40, freq="B"),
                       "ticker": ["AAA"] * 40})
    for c in cols:
        df[c] = 0.5
    p = tmp_path / name
    df.to_parquet(p)
    return p


def test_invalid_receipt_on_the_fallback_corpus_refuses(tmp_path, monkeypatch):
    """A run must not silently evaluate on a corpus another component disowned."""
    data = tmp_path / "data"
    data.mkdir()
    raw = _panel(data, "alpha158_291_fundamental_dataset_rawlabel.parquet",
                 ["fwd_60d_excess", "f1"])
    (data / (raw.name + ".INVALID.json")).write_text(
        json.dumps({"reason": "coverage != source panel, panel-only=432"}))
    monkeypatch.setattr(mod, "REPO", tmp_path)
    with pytest.raises(FileNotFoundError, match="INVALID receipt"):
        mod._load_sanity_panel(["f1"], "fwd_60d_excess", dataset_path=None)


def test_no_receipt_means_the_fallback_still_works(tmp_path, monkeypatch):
    """The refusal must be caused by the receipt, not by the test's fixture."""
    data = tmp_path / "data"
    data.mkdir()
    _panel(data, "alpha158_291_fundamental_dataset_rawlabel.parquet",
           ["fwd_60d_excess", "f1"])
    monkeypatch.setattr(mod, "REPO", tmp_path)
    panel, meta = mod._load_sanity_panel(["f1"], "fwd_60d_excess", dataset_path=None)
    assert len(panel) > 0


def test_the_alternate_receipt_spelling_is_also_caught(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    raw = _panel(data, "alpha158_291_fundamental_dataset_rawlabel.parquet",
                 ["fwd_60d_excess", "f1"])
    raw.with_suffix(".INVALID.json").write_text("{}")
    monkeypatch.setattr(mod, "REPO", tmp_path)
    with pytest.raises(FileNotFoundError, match="INVALID receipt"):
        mod._load_sanity_panel(["f1"], "fwd_60d_excess", dataset_path=None)


# --- defect 3: a static run records WHICH panel it scored ------------------

def test_static_branch_merges_panel_meta_before_its_own_keys():
    """Structural, and stated as such: the merge sits inside a long function that
    a unit test cannot reach without a full sim. The assertion is on the exact
    shape --- panel_meta spread FIRST so explicit keys still win --- so reordering
    or dropping it fails."""
    src = (Path(__file__).resolve().parent.parent / "src" / "renquant_backtesting"
           / "wf_gate" / "runner.py").read_text()
    needle = '"sanity_eval_scope": "static_artifact"'
    # The string occurs more than once (a contract description echoes it), so check
    # EVERY occurrence rather than the first --- my initial version used index()
    # and failed on an unrelated match, which is the same read-the-wrong-object
    # mistake this whole issue is about.
    hits = [i for i in range(len(src)) if src.startswith(needle, i)]
    assert hits, needle
    assert any("**(panel_meta or {})" in src[max(0, i - 200):i] for i in hits), (
        "no static sanity_meta literal merges panel_meta, so a static run does "
        "not record which panel it scored")
