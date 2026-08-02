"""Fail-closed tests for the #94 lineage admissibility module (slice 1)."""
from __future__ import annotations

import hashlib
import json
import re
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


# `tests/fixtures/`, NOT `tests/data/`: `.gitignore` line 12 is an unanchored `data/`,
# which matches ANY directory of that name. My first push put the fixture there, `git
# add -A` silently dropped it, the suite was green locally and CI failed with
# FileNotFoundError -- the exact "runs on one machine only" defect this test exists to
# end, committed by the test itself. The ignore rule is left alone: it is what keeps
# production parquet out of the repo, and narrowing it to win an argument with a
# fixture path would be the worse trade.
GRID = Path(__file__).resolve().parent / "fixtures" / "clf_lineage_window_grid.json"


def test_the_real_43_window_lineage_is_admissible_ON_A_REPO_CONTAINED_GRID():
    """The real-lineage result, checked by a test that ACTUALLY RUNS in CI.

    REVIEW ROUND 2, and it is round 1's own finding one repo downstream. The round-1
    integration test read the bundle from
    `/Users/renhao/git/github/renquant-model-wt-clfrebuild/...` and skipped when absent.
    That path is a transient WORKTREE: the test ran on exactly one machine, in a
    directory that disappears when the worktree is removed, and skipped silently
    everywhere else — while "43/43 admissible" was quoted as a result of this suite.
    A skipped test underwriting a published number is the shape this programme has now
    caught six times (`tests-that-measure-the-operators-disk`).

    Cross-repo integration genuinely cannot be repo-contained here: `evaluate_lineage`
    hashes 43 fold artifacts that live in renquant-model. So the responsibility splits,
    and each half runs where its inputs are:

      * renquant-model#181 owns DIGEST + ROOT verification — it has the artifacts, and
        it now has an in-repo verifier that recomputes all 43 digests and the
        `lineage_root_sha` from committed bytes;
      * this repo owns the ADMISSIBILITY CONTRACT, which needs only the per-window
        provenance and the caller grid — both committed here, both small.

    The grid is still corpus-derived (per cutoff, `min(date)` over the committed score
    corpus), so it remains the independent source round 1 asked for: no window is
    judged against its own declared bounds.
    """
    g = json.loads(GRID.read_text())
    assert g["schema"] == "clf-lineage-window-grid-v1"
    assert g["n_windows"] == len(g["windows"]) == 43
    refused = []
    margins = []
    for w in g["windows"]:
        v = L.check_window(w, pd.Timestamp(w["first_oos_date_from_corpus"]),
                           w["cutoff_embargo_days"])
        if not v.admissible:
            refused.append((w["cutoff_date"], v.reason))
        else:
            margins.append(v.embargo_margin_bdays)
    assert refused == [], refused
    assert len(margins) == 43 and min(margins) >= 1


def test_the_grid_says_WHERE_IT_CAME_FROM():
    """A snapshot without provenance is a number someone typed. The fixture carries the
    source repo/PR, the bundle path, the `lineage_root_sha` and the score corpus's
    sha256 — so a later reader can tell a changed bundle from a changed fixture, and the
    live cross-check below has something to compare against."""
    g = json.loads(GRID.read_text())
    # `source_pr` is asserted to EXIST, not to equal 181. Pinning the number made this
    # go red on the #182 regeneration while nothing was wrong -- a proxy for the claim
    # rather than the claim, and it adds no safety over `lineage_root_sha`, which the
    # drift guard below checks against upstream directly.
    assert g["source_repo"] == "renquant-model"
    assert isinstance(g["source_pr"], int) and g["source_pr"] > 0
    assert re.fullmatch(r"[0-9a-f]{64}", g["lineage_root_sha"])
    assert re.fullmatch(r"[0-9a-f]{64}", g["score_corpus_sha256"])
    assert g["grid_derivation"] == "per cutoff, min(date) over clf_wf_scores.parquet"


def test_the_grid_is_NOT_the_artifacts_own_windows():
    """ANTI-VACUITY, and the whole point of round 1's finding.

    The fixture records each artifact's self-declared `oos_window` alongside the
    corpus-derived date precisely so this test can prove they are two different objects
    that happen to agree. If the grid were silently re-derived from the declared windows
    the suite would keep passing and the self-attestation would be back — so the
    admissibility check above must be fed the corpus column, and the declared column
    must never be read by it.
    """
    g = json.loads(GRID.read_text())
    for w in g["windows"]:
        assert "artifact_declared_oos_window" in w and "first_oos_date_from_corpus" in w
    src = Path(__file__).read_text()
    body = src[src.index("def test_the_real_43_window_lineage_is_admissible"):
               src.index("def test_the_grid_says_WHERE_IT_CAME_FROM")]
    assert "artifact_declared_oos_window" not in body, \
        "the admissibility check is reading the artifacts' own windows again"


def test_the_grid_matches_the_UPSTREAM_REF_when_this_machine_has_the_repo():
    """Drift guard, and the only test here allowed to skip.

    A committed snapshot goes stale the moment upstream regenerates, and it did:
    `renquant-model#182` found that all 43 fold artifacts had been persisted with
    `feature_norm_kind` stringified, regenerated them, and moved
    `lineage_root_sha` from `e9eefe8137…` to `1da510478e…`. This guard caught it, which
    is the only reason the pin below is current.

    READ FROM `origin/main`, NOT FROM THE WORKING TREE. The first version resolved the
    bundle by looking for the file on disk, so what it compared against depended on which
    branch a sibling checkout happened to be sitting on — it found the bundle in a
    throwaway worktree during the incident and would have found nothing at all had that
    worktree been removed. A guard whose subject changes with someone else's `git
    checkout` is not measuring upstream; it is measuring the machine. `git show
    origin/main:…` is branch-independent, which is what "upstream" has to mean here.
    """
    import io
    import subprocess

    import pytest
    repo = Path(__file__).resolve().parents[2] / "renquant-model"
    if not (repo / ".git").exists():
        pytest.skip("no renquant-model checkout on this machine")
    sub = "doc/research/data/2026-08-01-clf-wf-lineage-bundle"

    def _show(rel: str) -> bytes:
        r = subprocess.run(["git", "-C", str(repo), "show", f"origin/main:{sub}/{rel}"],
                           capture_output=True)
        return r.stdout if r.returncode == 0 else b""

    raw_corpus, raw_man = _show("clf_wf_scores.parquet"), _show("clf_lineage_manifest.json")
    if not raw_corpus or not raw_man:
        pytest.skip("renquant-model has no origin/main carrying the bundle (unfetched?)")

    g = json.loads(GRID.read_text())
    corpus = pd.read_parquet(io.BytesIO(raw_corpus), columns=["cutoff", "date"])
    corpus["cutoff"] = pd.to_datetime(corpus["cutoff"])
    corpus["date"] = pd.to_datetime(corpus["date"])
    live = {str(c.date()): str(d.date())
            for c, d in corpus.groupby("cutoff")["date"].min().items()}
    snap = {w["cutoff_date"]: w["first_oos_date_from_corpus"] for w in g["windows"]}
    assert snap == live, "the committed grid has drifted from the upstream corpus"
    man = json.loads(raw_man)
    assert g["lineage_root_sha"] == man["lineage_root_sha"], (
        "the grid was cut from a different lineage than upstream's current one: "
        f"fixture {g['lineage_root_sha'][:12]}… vs origin/main {man['lineage_root_sha'][:12]}…")
    assert {w["cutoff_date"]: w["artifact_sha256"] for w in g["windows"]} == \
        {f["cutoff_date"]: f["artifact_sha256"] for f in man["folds"]}, \
        "per-fold digests disagree with upstream even though the root matched"


def test_caller_grid_governs_over_artifact_declared_windows(tmp_path):
    """Regression (review round 1): when the caller's grid and an artifact's own
    declared window conflict, the CALLER'S date decides — an artifact whose
    self-declared window would pass cannot rescue itself from a grid date that
    violates the causal contract."""
    win = _win("2024-06-03", "2024-03-01")
    # the artifact ALSO self-declares a comfortable window (ignored by design)
    win["oos_window"] = ["2024-06-04", "2024-06-24"]
    man = _mk_lineage(tmp_path, [win])
    # caller grid says this window's first OOS date is much EARLIER — a causal
    # violation under etc=2024-03-01 + 60 BDays (~2024-05-24)
    out = L.evaluate_lineage(
        man, recipe_id_key="recipe_src_sha256", label_horizon_bdays=60,
        first_oos_dates={"2024-06-03": pd.Timestamp("2024-04-01")},
        min_admissible_windows=1)
    by = out["windows"][0]
    assert by["admissibility"] == "refused"
    assert "causal violation" in by["reason"]
    assert out["lineage_verdict"] == "refused"
