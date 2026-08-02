"""Fail-closed tests for the #94 lineage admissibility module (slice 1)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from renquant_backtesting.wf_gate import lineage_admissibility as L


def _mk_lineage(tmp_path: Path, windows: list[dict], recipe_id: str = "recipe-abc",
                tamper_root: bool = False, drop_file_for: str | None = None,
                wrong_sha_for: str | None = None) -> Path:
    folds = []
    shas = []
    for w in windows:
        cutoff = w["cutoff_date"]
        d = tmp_path / "fold_artifacts" / cutoff
        d.mkdir(parents=True, exist_ok=True)
        p = d / "snap.json"
        p.write_text(json.dumps(w, sort_keys=True))
        true_sha = hashlib.sha256(p.read_bytes()).hexdigest()
        entry_sha = "00" * 32 if wrong_sha_for == cutoff else true_sha
        if drop_file_for == cutoff:
            p.unlink()
        folds.append({"cutoff_date": cutoff, "artifact_sha256": entry_sha,
                      "artifact_path": f"fold_artifacts/{cutoff}/snap.json"})
        # the ROOT is always over the TRUE shas — a per-entry lie must surface as a
        # per-window digest refusal, not collapse into a root mismatch
        shas.append(true_sha)
    root = L.lineage_root_sha(recipe_id, shas)
    if tamper_root:
        root = "11" * 32
    man = tmp_path / "lineage_manifest.json"
    man.write_text(json.dumps({"recipe_src_sha256": recipe_id,
                               "lineage_root_sha": root, "folds": folds}))
    return man


def _win(cutoff: str, etc: str) -> dict:
    return {"cutoff_date": cutoff, "cutoff_embargo_days": 60,
            "effective_train_cutoff_date": etc}


def test_admissible_window_records_the_embargo_margin():
    v = L.check_window(_win("2024-01-15", "2023-10-16"),
                       pd.Timestamp("2024-01-16"), 60)
    assert v.admissible and v.embargo_margin_bdays >= 1
    stamp = v.as_stamp()
    assert stamp["admissibility"] == "admissible"
    assert stamp["first_oos_date"] == "2024-01-16"


def test_causal_violation_is_refused_with_the_arithmetic_in_the_reason():
    # etc + 60 BDays lands ON/after the first OOS date -> refused
    v = L.check_window(_win("2024-01-15", "2023-12-01"),
                       pd.Timestamp("2024-01-16"), 60)
    assert not v.admissible
    assert "causal violation" in v.reason and "2023-12-01" in v.reason


def test_missing_provenance_fields_are_refused_not_defaulted():
    v = L.check_window({"cutoff_date": "2024-01-15"}, pd.Timestamp("2024-01-16"), 60)
    assert not v.admissible and "missing self-carried provenance" in v.reason


def test_lineage_end_to_end_admissible(tmp_path):
    wins = [_win(f"2024-0{i}-15", "2023-06-01") for i in range(1, 10)]
    man = _mk_lineage(tmp_path, wins)
    out = L.evaluate_lineage(
        man, recipe_id_key="recipe_src_sha256", label_horizon_bdays=60,
        first_oos_dates={w["cutoff_date"]: pd.Timestamp(w["cutoff_date"]) + pd.offsets.BDay(1)
                         for w in wins})
    assert out["lineage_verdict"] == "admissible" and out["n_admissible"] == 9
    assert out["lineage_root_sha_recomputed"] == out["lineage_root_sha_claimed"]


def test_tampered_root_refuses_the_whole_lineage(tmp_path):
    wins = [_win(f"2024-0{i}-15", "2023-06-01") for i in range(1, 10)]
    man = _mk_lineage(tmp_path, wins, tamper_root=True)
    out = L.evaluate_lineage(
        man, recipe_id_key="recipe_src_sha256", label_horizon_bdays=60,
        first_oos_dates={w["cutoff_date"]: pd.Timestamp(w["cutoff_date"]) + pd.offsets.BDay(1)
                         for w in wins})
    assert out["lineage_verdict"] == "refused"
    assert "lineage_root_sha mismatch" in out["reason"]


def test_digest_mismatch_and_missing_file_are_stamped_refusals(tmp_path):
    wins = [_win(f"2024-0{i}-15", "2023-06-01") for i in range(1, 10)]
    man = _mk_lineage(tmp_path, wins, wrong_sha_for="2024-01-15",
                      drop_file_for="2024-02-15")
    out = L.evaluate_lineage(
        man, recipe_id_key="recipe_src_sha256", label_horizon_bdays=60,
        first_oos_dates={w["cutoff_date"]: pd.Timestamp(w["cutoff_date"]) + pd.offsets.BDay(1)
                         for w in wins})
    by = {w["cutoff_date"]: w for w in out["windows"]}
    assert by["2024-01-15"]["admissibility"] == "refused"
    assert "digest mismatch" in by["2024-01-15"]["reason"]
    assert by["2024-02-15"]["admissibility"] == "refused"
    assert "missing" in by["2024-02-15"]["reason"]
    # 7 remain admissible < default minimum 8 -> whole lineage refused, loudly
    assert out["n_admissible"] == 7
    assert out["lineage_verdict"] == "refused"
    assert "admissible windows < minimum" in out["reason"]


def test_too_few_windows_refuses_even_when_all_admissible(tmp_path):
    wins = [_win("2024-01-15", "2023-06-01")]
    man = _mk_lineage(tmp_path, wins)
    out = L.evaluate_lineage(
        man, recipe_id_key="recipe_src_sha256", label_horizon_bdays=60,
        first_oos_dates={"2024-01-15": pd.Timestamp("2024-01-16")})
    assert out["lineage_verdict"] == "refused" and out["n_admissible"] == 1


def test_the_real_clf_lineage_bundle_if_present_evaluates_admissible():
    """Integration against the model repo's committed lineage (model#181). Skips
    loudly when the sibling checkout/branch is absent — never a silent pass."""
    import pytest
    man = Path("/Users/renhao/git/github/renquant-model-wt-clfrebuild/doc/research/"
               "data/2026-08-01-clf-wf-lineage-bundle/clf_lineage_manifest.json")
    if not man.is_file():
        pytest.skip("clf lineage bundle not present on this machine")
    lin = json.loads(man.read_text())
    # first OOS date per window = the artifact's own recorded oos_window[0] — used
    # here as the CALLER's grid (the runner will supply the corpus grid instead).
    first = {}
    for f in lin["folds"]:
        art = json.loads((man.parent / f["artifact_path"]).read_text())
        first[f["cutoff_date"]] = pd.Timestamp(art["oos_window"][0])
    out = L.evaluate_lineage(man, recipe_id_key="recipe_src_sha256",
                             label_horizon_bdays=60, first_oos_dates=first)
    assert out["lineage_verdict"] == "admissible"
    assert out["n_admissible"] == 43
    margins = [w["embargo_margin_bdays"] for w in out["windows"]]
    assert min(margins) >= 1
