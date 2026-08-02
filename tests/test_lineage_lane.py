"""Slice-3 lane tests: never-raises contract, in-memory gbdt lineage, honest stamps."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from renquant_backtesting.wf_gate import lineage_lane as LL


def _mk_manifest(tmp_path: Path, n=9, etc_offset_ok=True) -> Path:
    retrains = []
    for i in range(n):
        cut = pd.Timestamp("2024-01-01") + pd.offsets.BDay(21 * i)
        d = tmp_path / "artifacts" / f"w{i}"
        d.mkdir(parents=True)
        etc = cut - pd.offsets.BDay(85 if etc_offset_ok else 10)
        art = {"cutoff_date": str(cut.date()), "cutoff_embargo_days": 60,
               "effective_train_cutoff_date": str(etc.date())}
        (d / "panel.json").write_text(json.dumps(art))
        retrains.append({"cutoff_date": str(cut.date()),
                         "artifact_uri": f"artifacts/w{i}/panel.json"})
    man = tmp_path / "wf_manifest.json"
    man.write_text(json.dumps({"retrains": retrains}))
    return man


def test_happy_path_stamps_candidate_lineage_used(tmp_path):
    man = _mk_manifest(tmp_path)
    out = LL.attempt_lineage_stamp(
        artifact_usage={"eval_scope": "walkforward_manifest",
                        "manifest_path": str(man),
                        "candidate_recipe_fingerprint": "sha256:cfdd6cb8"},
        strategy_dir=tmp_path, label_horizon_bdays=60)
    assert out["lineage_lane"] == "stage1"
    assert out["candidate_lineage_used"] is True
    assert out["candidate_artifact_used"] is False       # documented, honest
    assert out["n_admissible"] == 9 and out["lineage_admissibility"] == "admissible"
    # root recomputes from the on-disk artifacts
    shas = []
    for i in range(9):
        p = tmp_path / "artifacts" / f"w{i}" / "panel.json"
        shas.append(hashlib.sha256(p.read_bytes()).hexdigest())
    payload = "sha256:cfdd6cb8" + "\n" + "\n".join(shas) + "\n"
    assert out["lineage_root_sha"] == hashlib.sha256(payload.encode()).hexdigest()


def test_causal_violations_refuse_the_lineage(tmp_path):
    man = _mk_manifest(tmp_path, etc_offset_ok=False)   # etc too close to cutoff
    out = LL.attempt_lineage_stamp(
        artifact_usage={"eval_scope": "walkforward_manifest",
                        "manifest_path": str(man),
                        "candidate_recipe_fingerprint": "sha256:cfdd6cb8"},
        strategy_dir=tmp_path, label_horizon_bdays=60)
    assert out["lineage_lane"] == "stage1"
    assert out["n_admissible"] == 0
    assert out["lineage_admissibility"] == "refused"
    assert all("causal violation" in w["reason"] for w in out["windows"])


def test_every_failure_is_a_stamped_unavailable_never_a_raise(tmp_path):
    # wrong scope
    a = LL.attempt_lineage_stamp(artifact_usage={"eval_scope": "static_artifact"},
                                 strategy_dir=tmp_path, label_horizon_bdays=60)
    assert a == {"lineage_lane": "unavailable",
                 "reason": "eval scope is not walkforward_manifest"}
    # no manifest path
    b = LL.attempt_lineage_stamp(artifact_usage={"eval_scope": "walkforward_manifest"},
                                 strategy_dir=tmp_path, label_horizon_bdays=60)
    assert b["lineage_lane"] == "unavailable" and "manifest_path" in b["reason"]
    # missing manifest file
    c = LL.attempt_lineage_stamp(
        artifact_usage={"eval_scope": "walkforward_manifest",
                        "manifest_path": str(tmp_path / "absent.json"),
                        "candidate_recipe_fingerprint": "x"},
        strategy_dir=tmp_path, label_horizon_bdays=60)
    assert c["lineage_lane"] == "unavailable"
    # missing window artifact
    man = _mk_manifest(tmp_path)
    (tmp_path / "artifacts" / "w3" / "panel.json").unlink()
    d = LL.attempt_lineage_stamp(
        artifact_usage={"eval_scope": "walkforward_manifest",
                        "manifest_path": str(man),
                        "candidate_recipe_fingerprint": "x"},
        strategy_dir=tmp_path, label_horizon_bdays=60)
    assert d["lineage_lane"] == "unavailable" and "missing" in d["reason"]


def test_the_real_gbdt_manifest_if_present_builds_an_admissible_lineage():
    """Loud-skip integration against the live strategy surface (read-only)."""
    import pytest
    sd = Path("/Users/renhao/git/github/RenQuant/backtesting/renquant_104")
    man = sd / "artifacts/sim/walkforward_manifest_gbdt_prod_recipe_v2.calibrated.json"
    if not man.is_file():
        pytest.skip("live gbdt WF manifest absent on this machine")
    out = LL.attempt_lineage_stamp(
        artifact_usage={"eval_scope": "walkforward_manifest",
                        "manifest_path": str(man),
                        "candidate_recipe_fingerprint": "sha256:cfdd6cb8e950da0f"},
        strategy_dir=sd, label_horizon_bdays=60)
    assert out["lineage_lane"] == "stage1", out.get("reason")
    assert out["n_windows"] == 43
    assert out["lineage_admissibility"] == "admissible"
    assert out["n_admissible"] == 43


def test_recipe_stamp_is_byte_unchanged_when_the_lane_is_unavailable(tmp_path):
    """The Stage-1 contract: attaching the lane block must not perturb anything
    the recipe path stamps. We verify at the unit level: attempt_lineage_stamp
    on a broken input returns ONLY the sibling block and mutates nothing."""
    usage = {"eval_scope": "walkforward_manifest", "manifest_path": None,
             "candidate_recipe_fingerprint": "x", "reason": "r",
             "manifest_rows_checked": 5}
    before = json.dumps(usage, sort_keys=True)
    block = LL.attempt_lineage_stamp(artifact_usage=usage, strategy_dir=tmp_path,
                                     label_horizon_bdays=60)
    assert json.dumps(usage, sort_keys=True) == before   # input not mutated
    assert block["lineage_lane"] == "unavailable"


def test_reordered_or_duplicated_ladder_and_cutoff_mismatch_are_unavailable(tmp_path):
    """Review round 1 item 2: structural violations must stamp unavailable."""
    # reordered ladder
    man = _mk_manifest(tmp_path)
    m = json.loads(man.read_text())
    m["retrains"] = list(reversed(m["retrains"]))
    man.write_text(json.dumps(m))
    out = LL.attempt_lineage_stamp(
        artifact_usage={"eval_scope": "walkforward_manifest",
                        "manifest_path": str(man),
                        "candidate_recipe_fingerprint": "x"},
        strategy_dir=tmp_path, label_horizon_bdays=60)
    assert out["lineage_lane"] == "unavailable" and "not chronologically ordered" in out["reason"]
    # duplicate cutoffs
    m["retrains"] = [m["retrains"][0], m["retrains"][0]]
    man.write_text(json.dumps(m))
    out2 = LL.attempt_lineage_stamp(
        artifact_usage={"eval_scope": "walkforward_manifest",
                        "manifest_path": str(man),
                        "candidate_recipe_fingerprint": "x"},
        strategy_dir=tmp_path, label_horizon_bdays=60)
    assert out2["lineage_lane"] == "unavailable" and "duplicates" in out2["reason"]
    # artifact cutoff != manifest cutoff (wrong artifact behind the window)
    tmp2 = tmp_path / "second"
    man2 = _mk_manifest(tmp2)
    art_p = tmp2 / "artifacts" / "w0" / "panel.json"
    art = json.loads(art_p.read_text())
    art["cutoff_date"] = "1999-01-01"
    art_p.write_text(json.dumps(art))
    out3 = LL.attempt_lineage_stamp(
        artifact_usage={"eval_scope": "walkforward_manifest",
                        "manifest_path": str(man2),
                        "candidate_recipe_fingerprint": "x"},
        strategy_dir=tmp2, label_horizon_bdays=60)
    assert out3["lineage_lane"] == "unavailable" and "wrong artifact" in out3["reason"]


def test_runner_attaches_the_lane_only_as_a_sibling_output_key():
    """Review round 1 item 1, pinned at source level: the runner must NEVER write
    into artifact_usage (which flows through the WF and sanity paths); the lane
    block appears exclusively as its own key in the output assembly. A full
    end-to-end byte-diff needs a gate-run harness tests do not have — this guard
    plus the input-never-mutated unit test above are the enforceable halves."""
    src = (Path(__file__).resolve().parent.parent /
           "src/renquant_backtesting/wf_gate/runner.py").read_text()
    assert 'artifact_usage["lineage_stage1"]' not in src
    assert "artifact_usage = dict(artifact_usage)" not in src
    assert '"lineage_stage1":      lineage_stage1' in src
