"""Candidate-LINEAGE scoring engine — #94 implementation slice 2 (skeleton).

Consumes slice 1's admissibility verdicts VERBATIM and scores each admissible
window's OOS rows with that window's OWN snapshot. Produces a stamp-ready dict;
performs NO admission decision and touches NO runner behavior.

Design constraints carried over from the merged #94 text and slice 1:
* the CALLER supplies the OOS grid (``oos_dates_by_cutoff``) — never the
  artifacts;
* a window graded admissible that then fails to LOAD or SCORE becomes a stamped
  ``scoring_error`` refusal, and the admissible count is re-checked against the
  lineage minimum (fail-closed, never a silent drop);
* the feature transform is NEVER reimplemented here: the default scorer factory
  defers to the single-sourced pipeline transform used by the gate's sanity
  path; tests may inject a factory, which changes WHO scores, not WHAT decides.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from renquant_backtesting.wf_gate.lineage_admissibility import (
    DEFAULT_MIN_ADMISSIBLE_WINDOWS,
)

#: A scorer factory returns ``score(sub_panel: DataFrame) -> pd.Series`` for one
#: window artifact payload. The default defers to the pipeline's single-sourced
#: transform + booster load; tests inject fakes.
ScorerFactory = Callable[[dict, Path], Callable[[pd.DataFrame], pd.Series]]


def _default_scorer_factory(artifact: dict, artifact_path: Path):
    """Load the window snapshot via the SAME single-sourced machinery the gate's
    sanity path uses (PanelScorer + transform_feature_frame). Kept lazy so tests
    with injected factories never import heavyweight dependencies."""
    from renquant_pipeline.kernel.panel_pipeline.panel_scorer import PanelScorer  # noqa: PLC0415

    scorer = PanelScorer(artifact)

    def _score(sub: pd.DataFrame) -> pd.Series:
        return scorer.score(sub)

    return _score


def score_lineage(*, lineage_manifest: Path, admissibility: dict,
                  panel: pd.DataFrame,
                  oos_dates_by_cutoff: dict[str, list],
                  min_admissible_windows: int = DEFAULT_MIN_ADMISSIBLE_WINDOWS,
                  scorer_factory: ScorerFactory = _default_scorer_factory) -> dict:
    """Score every ADMISSIBLE window on the caller's OOS grid.

    ``panel`` carries at least ``date``/``ticker`` plus the feature columns each
    artifact names. Returns per-window outcomes + pooled per-date scores, ready
    to embed in a ``candidate_lineage_used`` stamp. Statistics on the pooled
    scores are the NEXT slice — this one produces the scores and the fail-closed
    bookkeeping only.
    """
    lineage_manifest = Path(lineage_manifest)
    man = json.loads(lineage_manifest.read_text())
    base = lineage_manifest.parent
    adm_by_cutoff = {w["cutoff_date"]: w for w in admissibility.get("windows", [])}
    fold_by_cutoff = {f["cutoff_date"]: f for f in man.get("folds", [])}

    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    by_date = dict(tuple(panel.groupby("date")))

    windows_out: list[dict] = []
    pooled: list[pd.DataFrame] = []
    n_scored_windows = 0
    for cutoff, adm in adm_by_cutoff.items():
        if adm.get("admissibility") != "admissible":
            windows_out.append({**adm, "scoring": "skipped_inadmissible"})
            continue
        fold = fold_by_cutoff.get(cutoff)
        dates = [pd.Timestamp(d) for d in oos_dates_by_cutoff.get(cutoff, [])]
        if fold is None or not dates:
            windows_out.append({**adm, "scoring": "scoring_error",
                                "scoring_reason": "no fold entry or empty caller grid"})
            continue
        try:
            art_path = base / fold["artifact_path"]
            artifact = json.loads(art_path.read_text())
            score = scorer_factory(artifact, art_path)
            frames = []
            for d in dates:
                sub = by_date.get(pd.Timestamp(d))
                if sub is None or sub.empty:
                    continue
                s = score(sub)
                frames.append(pd.DataFrame({
                    "date": pd.Timestamp(d), "ticker": s.index, "score": s.values,
                    "cutoff_date": cutoff}))
            if not frames:
                raise ValueError("no scorable rows on the caller grid")
            wdf = pd.concat(frames, ignore_index=True)
            pooled.append(wdf)
            n_scored_windows += 1
            windows_out.append({**adm, "scoring": "scored",
                                "n_dates": int(wdf["date"].nunique()),
                                "n_rows": int(len(wdf))})
        except Exception as exc:  # noqa: BLE001 — every failure is a stamped outcome
            windows_out.append({**adm, "scoring": "scoring_error",
                                "scoring_reason": str(exc)[:300]})

    scores = (pd.concat(pooled, ignore_index=True)
              if pooled else pd.DataFrame(columns=["date", "ticker", "score",
                                                   "cutoff_date"]))
    verdict = ("scored" if n_scored_windows >= min_admissible_windows
               else "refused")
    return {
        "lineage_scoring_verdict": verdict,
        "reason": ("ok" if verdict == "scored" else
                   f"only {n_scored_windows} scored windows < minimum "
                   f"{min_admissible_windows} (admissible-then-failed windows count "
                   f"against the lineage, never silently)"),
        "lineage_root_sha": admissibility.get("lineage_root_sha_recomputed"),
        "n_windows": len(windows_out),
        "n_scored_windows": n_scored_windows,
        "windows": windows_out,
        "scores": scores,
    }


def summarize_lineage_scores(scores: pd.DataFrame,
                             labels_by_date: dict) -> dict:
    """Stamp-level descriptive summary of pooled lineage scores.

    ``labels_by_date`` maps date -> ``pd.Series`` (label by ticker), supplied by
    the CALLER (the gate's label contract) — never derived from the lineage.
    Inference (bars, placebo, genuine_ic) belongs to the runner slice; this
    summary records per-date rank ICs and coverage so the stamp is auditable
    on its own.
    """
    from scipy import stats as _st
    per_date = []
    for d, g in scores.groupby("date"):
        lab = labels_by_date.get(pd.Timestamp(d))
        if lab is None:
            continue
        j = pd.DataFrame({"s": g.set_index("ticker")["score"], "y": lab}).dropna()
        if len(j) < 20:
            continue
        per_date.append({"date": str(pd.Timestamp(d).date()),
                         "ic": float(_st.spearmanr(j["s"], j["y"]).statistic),
                         "n": int(len(j))})
    ics = [r["ic"] for r in per_date]
    return {
        "n_dates_scored": int(scores["date"].nunique()) if len(scores) else 0,
        "n_dates_with_labels": len(per_date),
        "mean_ic": (float(np.mean(ics)) if ics else None),
        "per_date": per_date,
    }
