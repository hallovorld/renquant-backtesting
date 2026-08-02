"""Slice-2 skeleton tests: fail-closed scoring bookkeeping with injected scorers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from renquant_backtesting.wf_gate import lineage_admissibility as LA
from renquant_backtesting.wf_gate import lineage_scoring as LS


def _bundle(tmp_path: Path, cutoffs: list[str]) -> tuple[Path, dict]:
    folds, shas = [], []
    for c in cutoffs:
        d = tmp_path / "fold_artifacts" / c
        d.mkdir(parents=True)
        p = d / "snap.json"
        p.write_text(json.dumps({"cutoff_date": c, "cutoff_embargo_days": 60,
                                 "effective_train_cutoff_date": "2023-06-01",
                                 "feature_cols": ["f1"]}))
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        folds.append({"cutoff_date": c, "artifact_sha256": sha,
                      "artifact_path": f"fold_artifacts/{c}/snap.json"})
        shas.append(sha)
    man = tmp_path / "lineage_manifest.json"
    man.write_text(json.dumps({
        "recipe_src_sha256": "recipe-xyz",
        "lineage_root_sha": LA.lineage_root_sha("recipe-xyz", shas),
        "folds": folds}))
    grid = {c: [pd.Timestamp(c) + pd.offsets.BDay(1)] for c in cutoffs}
    adm = LA.evaluate_lineage(man, recipe_id_key="recipe_src_sha256",
                              label_horizon_bdays=60,
                              first_oos_dates={c: g[0] for c, g in grid.items()},
                              min_admissible_windows=1)
    return man, {"adm": adm, "grid": grid}


def _panel(cutoffs: list[str]) -> pd.DataFrame:
    rows = []
    for c in cutoffs:
        d = pd.Timestamp(c) + pd.offsets.BDay(1)
        for t in ("AAA", "BBB", "CCC"):
            rows.append({"date": d, "ticker": t, "f1": 1.0})
    return pd.DataFrame(rows)


def _ok_factory(artifact, path):
    # factory contract: sub is TICKER-INDEXED with feature columns
    return lambda sub: pd.Series(range(len(sub)), index=sub.index, dtype=float)


def test_scores_every_admissible_window_and_pools(tmp_path):
    cutoffs = ["2024-01-15", "2024-02-15"]
    man, ctx = _bundle(tmp_path, cutoffs)
    out = LS.score_lineage(lineage_manifest=man, admissibility=ctx["adm"],
                           panel=_panel(cutoffs), oos_dates_by_cutoff=ctx["grid"],
                           min_admissible_windows=1, scorer_factory=_ok_factory)
    assert out["lineage_scoring_verdict"] == "scored"
    assert out["n_scored_windows"] == 2
    assert len(out["scores"]) == 6
    assert set(out["scores"]["cutoff_date"]) == set(cutoffs)


def test_admissible_then_failing_window_is_a_stamped_scoring_error(tmp_path):
    cutoffs = ["2024-01-15", "2024-02-15"]
    man, ctx = _bundle(tmp_path, cutoffs)

    def _flaky(artifact, path):
        if artifact["cutoff_date"] == "2024-02-15":
            raise RuntimeError("booster exploded")
        return _ok_factory(artifact, path)

    out = LS.score_lineage(lineage_manifest=man, admissibility=ctx["adm"],
                           panel=_panel(cutoffs), oos_dates_by_cutoff=ctx["grid"],
                           min_admissible_windows=2, scorer_factory=_flaky)
    by = {w["cutoff_date"]: w for w in out["windows"]}
    assert by["2024-02-15"]["scoring"] == "scoring_error"
    assert "booster exploded" in by["2024-02-15"]["scoring_reason"]
    # 1 scored < minimum 2 -> the LINEAGE refuses, loudly
    assert out["lineage_scoring_verdict"] == "refused"
    assert "count against the lineage" in out["reason"]


def test_inadmissible_windows_are_never_scored(tmp_path):
    cutoffs = ["2024-01-15"]
    man, ctx = _bundle(tmp_path, cutoffs)
    # caller grid violating the causal contract -> window inadmissible upstream
    adm = LA.evaluate_lineage(man, recipe_id_key="recipe_src_sha256",
                              label_horizon_bdays=60,
                              first_oos_dates={"2024-01-15": pd.Timestamp("2023-07-01")},
                              min_admissible_windows=1)
    calls = []

    def _spy(artifact, path):
        calls.append(artifact["cutoff_date"])
        return _ok_factory(artifact, path)

    out = LS.score_lineage(lineage_manifest=man, admissibility=adm,
                           panel=_panel(cutoffs),
                           oos_dates_by_cutoff=ctx["grid"],
                           min_admissible_windows=1, scorer_factory=_spy)
    assert calls == []                       # never loaded, never scored
    assert out["windows"][0]["scoring"] == "skipped_inadmissible"
    assert out["lineage_scoring_verdict"] == "refused"


def test_summarize_lineage_scores_uses_only_caller_labels(tmp_path):
    import numpy as np
    cutoffs = ["2024-01-15"]
    man, ctx = _bundle(tmp_path, cutoffs)
    panel = _panel(cutoffs)
    # widen the cross-section so the >= 20-name floor is satisfiable
    rows = []
    d = pd.Timestamp("2024-01-15") + pd.offsets.BDay(1)
    for i in range(25):
        rows.append({"date": d, "ticker": f"T{i:02d}", "f1": 1.0})
    panel = pd.DataFrame(rows)
    out = LS.score_lineage(lineage_manifest=man, admissibility=ctx["adm"],
                           panel=panel, oos_dates_by_cutoff=ctx["grid"],
                           min_admissible_windows=1,
                           scorer_factory=_ok_factory)
    labels = {d: pd.Series(np.arange(25, dtype=float),
                           index=[f"T{i:02d}" for i in range(25)])}
    summ = LS.summarize_lineage_scores(out["scores"], labels)
    assert summ["n_dates_with_labels"] == 1
    assert summ["mean_ic"] == 1.0            # monotone fake scorer vs monotone labels
    # a date with NO caller label contributes nothing (never invented)
    summ2 = LS.summarize_lineage_scores(out["scores"], {})
    assert summ2["n_dates_with_labels"] == 0 and summ2["mean_ic"] is None


def test_GOLDEN_default_factory_reproduces_the_committed_corpus_end_to_end():
    """The whole slice-2 pipeline against the REAL repaired lineage (model#182):
    admissibility (corpus grid) → default (recipe-transform) scoring → the pooled
    scores must reproduce the committed corpus < 1e-6 on a sampled window. Loud
    skip where the model checkout/panel are absent."""
    import io
    import subprocess
    import tempfile
    import pytest
    # Anchor to renquant-model origin/main (review round: a mutable sibling worktree
    # can vanish and silently skip the regression guard for the #182 corruption).
    model_repo = Path(__file__).resolve().parents[2] / "renquant-model"
    BUNDLE_REF = "origin/main:doc/research/data/2026-08-01-clf-wf-lineage-bundle"
    def _show(rel: str) -> bytes:
        return subprocess.run(
            ["git", "-C", str(model_repo), "show", f"{BUNDLE_REF}/{rel}"],
            capture_output=True, check=True).stdout
    panel_path = Path("/Users/renhao/git/github/RenQuant/data/alpha158_291_fundamental_dataset.parquet")
    if not panel_path.is_file():
        pytest.skip("panel absent on this machine (the ONLY permitted skip)")
    # bundle bytes come from the STABLE ref — materialized to a temp dir so the
    # engine under test reads exactly what model main carries.
    tmp = Path(tempfile.mkdtemp(prefix="lineage_golden_"))
    lin = json.loads(_show("clf_lineage_manifest.json").decode())
    (tmp / "clf_lineage_manifest.json").write_bytes(_show("clf_lineage_manifest.json"))
    for f in lin["folds"]:
        d = tmp / Path(f["artifact_path"]).parent
        d.mkdir(parents=True, exist_ok=True)
        (tmp / f["artifact_path"]).write_bytes(_show(f["artifact_path"]))
    man = tmp / "clf_lineage_manifest.json"
    corpus = pd.read_parquet(io.BytesIO(_show("clf_wf_scores.parquet")))
    corpus["date"] = pd.to_datetime(corpus["date"])
    corpus["cutoff"] = pd.to_datetime(corpus["cutoff"])
    first = {str(c.date()): d for c, d in corpus.groupby("cutoff")["date"].min().items()}
    adm = LA.evaluate_lineage(man, recipe_id_key="recipe_src_sha256",
                              label_horizon_bdays=60, first_oos_dates=first)
    assert adm["lineage_verdict"] == "admissible"
    # one sampled window end-to-end through score_lineage with the DEFAULT factory
    cut = "2024-04-08"
    dates = sorted(corpus[corpus["cutoff"] == cut]["date"].unique())
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["date"].isin(dates)].set_index("ticker")
    out = LS.score_lineage(
        lineage_manifest=man,
        admissibility={**adm, "windows": [w for w in adm["windows"]
                                          if w["cutoff_date"] == cut]},
        panel=panel.reset_index(),
        oos_dates_by_cutoff={cut: list(dates)},
        min_admissible_windows=1)
    assert out["lineage_scoring_verdict"] == "scored"
    got = out["scores"].set_index(["date", "ticker"])["score"]
    exp_rows = corpus[corpus["cutoff"] == cut]
    expect = exp_rows.set_index(["date", "ticker"])["cal"]
    j = pd.DataFrame({"e": expect, "g": got}).dropna()
    assert len(j) > 3000
    max_d = float((j["e"] - j["g"]).abs().max())
    assert max_d < 1e-6, f"default-factory lineage scoring diverges: {max_d}"


def test_griddate_without_panel_rows_is_a_stamped_scoring_error(tmp_path):
    """Review round 2: a requested date with no rows must refuse, never skip."""
    cutoffs = ["2024-01-15"]
    man, ctx = _bundle(tmp_path, cutoffs)
    grid = {"2024-01-15": [pd.Timestamp("2024-01-16"), pd.Timestamp("2024-01-17")]}
    panel = _panel(cutoffs)                     # rows only for 01-16
    out = LS.score_lineage(lineage_manifest=man, admissibility=ctx["adm"],
                           panel=panel, oos_dates_by_cutoff=grid,
                           min_admissible_windows=1, scorer_factory=_ok_factory)
    w = out["windows"][0]
    assert w["scoring"] == "scoring_error" and "no panel rows" in w["scoring_reason"]
    assert out["lineage_scoring_verdict"] == "refused"


def test_partial_scorer_output_is_a_stamped_scoring_error(tmp_path):
    """Review round 2: a scorer returning a subset (or reorder) must refuse."""
    cutoffs = ["2024-01-15"]
    man, ctx = _bundle(tmp_path, cutoffs)

    def _partial(artifact, path):
        return lambda sub: pd.Series([0.1], index=[sub.index[0]], dtype=float)

    out = LS.score_lineage(lineage_manifest=man, admissibility=ctx["adm"],
                           panel=_panel(cutoffs), oos_dates_by_cutoff=ctx["grid"],
                           min_admissible_windows=1, scorer_factory=_partial)
    w = out["windows"][0]
    assert w["scoring"] == "scoring_error" and "output index != input index" in w["scoring_reason"]
    assert out["lineage_scoring_verdict"] == "refused"
