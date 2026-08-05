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
    """The NORMATIVE lineage scorer: the RECIPE transform (renquant_model_gbdt's
    panel_training_matrix + raw booster predict), which reproduces the committed
    evidence corpus at < 1e-6 — measured 2026-08-01 on the repaired clf bundle,
    fold 5's whole OOS window. The serving path (PanelScorer) diverges from the
    same corpus by mean ~1.2e-2 on clf probabilities (±5 post-z clip + fill
    policy; renquant-pipeline#248), so it is NOT the default here: the lineage
    claim is 'this lineage produced THAT evidence', and the evidence was produced
    by the recipe transform. Serving parity is a separate stamped diagnostic
    (``serving_parity_scorer_factory``). Lazy imports keep injected-factory tests
    dependency-free."""
    # The PUBLIC pinned contract (model#183): the one supported external scoring
    # API, fail-closed at load (incl. the stringified-norm_kind incident shape).
    # No training-internal import remains here.
    from renquant_model_gbdt.fold_scoring import load_fold_scorer  # noqa: PLC0415

    return load_fold_scorer(artifact)


def serving_parity_scorer_factory(artifact: dict, artifact_path: Path):
    """The SERVING path (PanelScorer → transform_feature_frame), for the stamped
    parity diagnostic ONLY — measures the live serving skew vs the recipe
    transform (renquant-pipeline#248); never feeds a lineage verdict."""
    from renquant_pipeline.kernel.panel_pipeline.panel_scorer import PanelScorer  # noqa: PLC0415

    scorer = PanelScorer.load(artifact_path)

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
                    # a REQUESTED grid date with no panel rows is missing data,
                    # never a silent skip (review round 2)
                    raise ValueError(f"caller-grid date {pd.Timestamp(d).date()} "
                                     "has no panel rows")
                # factory contract: a TICKER-INDEXED frame with feature columns
                sub = sub.set_index("ticker")
                s = score(sub)
                if not s.index.equals(sub.index):
                    # partial or reordered scorer output silently narrows the
                    # evidence — refuse (review round 2)
                    raise ValueError(
                        f"scorer output index != input index on "
                        f"{pd.Timestamp(d).date()}: {len(s)} vs {len(sub)} rows")
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


UNASSIGNED_REGIME = "__unassigned__"


def summarize_lineage_scores(scores: pd.DataFrame,
                             labels_by_date: dict,
                             regime_by_date: dict | None = None) -> dict:
    """Stamp-level descriptive summary of lineage scores, POOLED AND PER REGIME.

    ``labels_by_date`` maps date -> ``pd.Series`` (label by ticker), supplied by
    the CALLER (the gate's label contract) — never derived from the lineage.
    ``regime_by_date`` maps date -> regime name and follows the same rule: it
    comes from the production regime chain via the caller, so this summary never
    invents a regime.

    WHY PER REGIME (orch#805, measured 2026-08-05): the pooled figure is a
    REGIME-MIX ARTIFACT on this book. The served recipe's genuine IC measured
    +0.335 in BEAR — where the strategy places zero buys — and negative in
    BULL_CALM, where 136 of its 154 buys land, and the pooled number came out
    POSITIVE anyway because BEAR's 50 dates dragged it up. A Stage-2 lane that
    scores a candidate on a pooled mean would rank candidates on the same
    artifact. Pooling is still reported (continuity), now beside the split that
    explains it.

    Absence reads as absence: with no ``regime_by_date`` the split is ``None``
    with a stated reason, never an empty dict that looks like "measured, and
    there were no regimes".
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
        row = {"date": str(pd.Timestamp(d).date()),
               "ic": float(_st.spearmanr(j["s"], j["y"]).statistic),
               "n": int(len(j))}
        if regime_by_date is not None:
            regime = regime_by_date.get(pd.Timestamp(d))
            # A date the caller could not label is UNASSIGNED, not dropped:
            # silently discarding it would change the pooled mean the split is
            # supposed to explain. [codex on bt#107] NaN/NA count as unlabelled —
            # a raw `Series.to_dict()` carries them, and stringifying one would
            # create a literal "nan" bucket that looks like a regime.
            row["regime"] = (UNASSIGNED_REGIME if regime is None or pd.isna(regime)
                             else str(regime))
        per_date.append(row)

    ics = [r["ic"] for r in per_date]
    summary = {
        "n_dates_scored": int(scores["date"].nunique()) if len(scores) else 0,
        "n_dates_with_labels": len(per_date),
        "mean_ic": (float(np.mean(ics)) if ics else None),
        "per_date": per_date,
    }
    summary.update(_by_regime_block(per_date, regime_by_date))
    return summary


def _by_regime_block(per_date: list, regime_by_date: dict | None) -> dict:
    if regime_by_date is None:
        return {"by_regime": None,
                "by_regime_reason": "no regime_by_date supplied by the caller",
                "pooled_ic": None,
                "pooled_sign_carriers": None}
    buckets: dict[str, list] = {}
    for row in per_date:
        buckets.setdefault(row["regime"], []).append(row["ic"])
    by_regime = {
        name: {"n_dates": len(vals),
               "mean_ic": float(np.mean(vals)),
               "min_ic": float(np.min(vals)),
               "max_ic": float(np.max(vals))}
        for name, vals in sorted(buckets.items())
    }
    # DECOMPOSITION, not a sign-disagreement flag. [codex on bt#107] The first
    # version asked whether the pooled mean disagreed with EVERY regime — which
    # is arithmetically impossible once dates are assigned, because the pooled
    # mean is a date-weighted average of the regime means and must lie between
    # them. It was dead code for the very shape it was written for.
    #
    # The live shape is not sign-disagreement, it is DOMINANCE: a small,
    # high-|IC| regime supplying the pooled sign while the regime that carries
    # the trading has the opposite one. So: report each regime's weight and its
    # contribution to the pooled mean, and name any regime whose REMOVAL flips
    # the pooled sign. On this book that is BEAR at ~12% of dates.
    total = len(per_date)
    pooled = float(np.mean([r["ic"] for r in per_date])) if total else None
    for name, cell in by_regime.items():
        cell["weight"] = cell["n_dates"] / total if total else None
        cell["contribution_to_pooled_ic"] = (
            cell["weight"] * cell["mean_ic"] if cell["weight"] is not None else None)

    sign_carriers = []
    if pooled is not None and pooled != 0:
        for name in by_regime:
            rest = [r["ic"] for r in per_date if r["regime"] != name]
            if not rest:
                continue
            without = float(np.mean(rest))
            if (without > 0) != (pooled > 0):
                sign_carriers.append(
                    {"regime": name,
                     "weight": by_regime[name]["weight"],
                     "pooled_ic_without_it": without})
    return {"by_regime": by_regime,
            "by_regime_reason": None,
            "pooled_ic": pooled,
            # Non-empty means the pooled SIGN is supplied by regime(s) whose
            # removal flips it — read the split, not the pool.
            "pooled_sign_carriers": sign_carriers}
