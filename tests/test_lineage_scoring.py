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
    return lambda sub: pd.Series(range(len(sub)), index=sub["ticker"], dtype=float)


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
