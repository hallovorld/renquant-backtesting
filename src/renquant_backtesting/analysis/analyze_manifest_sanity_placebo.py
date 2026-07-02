#!/usr/bin/env python3
"""Decompose walk-forward sanity IC into real, placebo, and regime buckets.

This is a diagnostic companion to scripts/run_wf_gate.py.  It deliberately
does not mutate production artifacts: it scores a WF manifest through the same
manifest contract, then asks whether the reported IC is model alpha or merely
overlapping-horizon / regime-persistence structure in the labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import subprocess
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from renquant_backtesting.wf_gate.runner import (
    REPO,
    STRATEGY_DIR,
    _load_artifact_payload,
    _load_sanity_panel,
    _score_manifest_sanity,
    _sanity_model_label_col,
)


DEFAULT_SHIFTS = (5, 10, 20, 40, 60, 80, 120, 180, 252)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _mean(xs: Iterable[float]) -> float | None:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else None


def _cs_ic_series(
    pred: np.ndarray | pd.Series,
    label: np.ndarray | pd.Series,
    dates: Iterable[Any],
    *,
    min_names: int = 5,
) -> pd.DataFrame:
    """Per-date cross-sectional Spearman IC."""
    df = pd.DataFrame({
        "pred": np.asarray(pred, dtype=float),
        "label": np.asarray(label, dtype=float),
        "date": pd.to_datetime(list(dates)),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    rows: list[dict[str, Any]] = []
    for d, g in df.groupby("date", sort=True):
        if len(g) < min_names:
            continue
        ic = spearmanr(g["pred"], g["label"])[0]
        if ic == ic:
            rows.append({"date": pd.Timestamp(d), "ic": float(ic), "n": int(len(g))})
    return pd.DataFrame(rows)


def summarize_ic(
    pred: np.ndarray | pd.Series,
    label: np.ndarray | pd.Series,
    dates: Iterable[Any],
    *,
    min_names: int = 5,
) -> dict[str, Any]:
    per_date = _cs_ic_series(pred, label, dates, min_names=min_names)
    if per_date.empty:
        return {"mean_ic": None, "n_dates": 0, "n_rows": 0}
    return {
        "mean_ic": float(per_date["ic"].mean()),
        "median_ic": float(per_date["ic"].median()),
        "std_ic": float(per_date["ic"].std(ddof=1)) if len(per_date) > 1 else 0.0,
        "hit_rate": float((per_date["ic"] > 0).mean()),
        "n_dates": int(len(per_date)),
        "n_rows": int(per_date["n"].sum()),
    }


def shift_diagnostics(
    panel: pd.DataFrame,
    val: pd.DataFrame,
    mu: pd.Series,
    label: str,
    *,
    shifts: Iterable[int] = DEFAULT_SHIFTS,
    min_names: int = 5,
) -> list[dict[str, Any]]:
    """Compare model-placebo IC with raw label persistence by shift."""
    panel_s = panel.sort_values(["ticker", "date"]).copy()
    val_idx = val.set_index(["ticker", "date"])
    mu_by_idx = pd.Series(np.asarray(mu.loc[val.index], dtype=float), index=val_idx.index)
    label_by_idx = val_idx[label].astype(float)
    out: list[dict[str, Any]] = []
    real = summarize_ic(mu_by_idx, label_by_idx, [d for _, d in val_idx.index],
                        min_names=min_names)
    real_ic = real.get("mean_ic")
    for shift_days in shifts:
        col = f"__shift_{int(shift_days)}__"
        panel_s[col] = panel_s.groupby("ticker")[label].shift(-int(shift_days))
        shifted = panel_s.dropna(subset=[col]).set_index(["ticker", "date"])
        common = val_idx.index.intersection(shifted.index)
        if len(common) <= min_names:
            out.append({
                "shift_days": int(shift_days),
                "n_rows": int(len(common)),
                "n_dates": 0,
                "model_placebo_ic": None,
                "label_autocorr_ic": None,
            })
            continue
        dates = [d for _, d in common]
        y_real_aligned = label_by_idx.loc[common].astype(float)
        y_future = shifted.loc[common, col].clip(-0.5, 0.5).astype(float)
        aligned_real_ic = summarize_ic(mu_by_idx.loc[common], y_real_aligned, dates,
                                       min_names=min_names)
        model_ic = summarize_ic(mu_by_idx.loc[common], y_future, dates,
                                min_names=min_names)
        autocorr_ic = summarize_ic(label_by_idx.loc[common], y_future, dates,
                                   min_names=min_names)
        model_mean = model_ic.get("mean_ic")
        aligned_real_mean = aligned_real_ic.get("mean_ic")
        out.append({
            "shift_days": int(shift_days),
            "n_rows": int(len(common)),
            "n_dates": int(model_ic.get("n_dates") or 0),
            "aligned_real_ic": aligned_real_mean,
            "full_real_ic": real_ic,
            "model_placebo_ic": model_mean,
            "label_autocorr_ic": autocorr_ic.get("mean_ic"),
            "model_placebo_abs_ratio_to_aligned_real": (
                abs(float(model_mean)) / abs(float(aligned_real_mean))
                if model_mean is not None and aligned_real_mean not in (None, 0.0)
                else None
            ),
            "model_placebo_abs_ratio_to_full_real": (
                abs(float(model_mean)) / abs(float(real_ic))
                if model_mean is not None and real_ic not in (None, 0.0)
                else None
            ),
            "label_autocorr_hit_rate": autocorr_ic.get("hit_rate"),
        })
    return out


def _resolve_strategy_artifact(strategy_dir: Path, raw: str | None) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    candidates = [p] if p.is_absolute() else [
        strategy_dir / "artifacts" / p,
        strategy_dir / p,
        REPO / p,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _load_config(strategy_dir: Path) -> dict:
    return json.loads((strategy_dir / "strategy_config.json").read_text())


def _load_gmm(strategy_dir: Path, config: dict) -> dict | None:
    p = _resolve_strategy_artifact(
        strategy_dir,
        str((config.get("regime", {}) or {}).get("gmm_artifact") or ""),
    )
    if p is None or not p.exists():
        return None
    return json.loads(p.read_text())


def _load_spy_frame() -> pd.DataFrame:
    df = pd.read_parquet(REPO / "data" / "ohlcv" / "SPY" / "1d.parquet")
    if "date" not in df.columns:
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").set_index("date")


def build_regime_series(
    dates: Iterable[Any],
    *,
    strategy_dir: Path = STRATEGY_DIR,
) -> pd.DataFrame:
    """Run the production regime Task chain for each requested date."""
    logging.getLogger("kernel.pipeline.regime").setLevel(logging.WARNING)
    logging.getLogger("kernel.regime").setLevel(logging.WARNING)
    from renquant_pipeline.kernel.regime import RegimeState  # noqa: PLC0415
    from renquant_pipeline.kernel.pipeline.task_regime import (  # noqa: PLC0415
        BEAROverrideTask,
        CUSUMTask,
        GMMTask,
        HurstTask,
        RegimeFinalizeTask,
    )

    config = _load_config(strategy_dir)
    gmm = _load_gmm(strategy_dir, config)
    spy = _load_spy_frame()
    tasks = [HurstTask(), CUSUMTask(), GMMTask(), BEAROverrideTask(), RegimeFinalizeTask()]
    ctx = SimpleNamespace(
        config=config,
        regime_state=RegimeState(),
        spy_returns=[],
        ohlcv={},
        gmm=gmm,
        regime_counts={},
        today=None,
        regime=None,
        confidence=None,
    )
    out: list[dict[str, Any]] = []
    for raw_d in sorted({pd.Timestamp(d).normalize() for d in dates}):
        hist = spy.loc[spy.index <= raw_d].copy()
        if len(hist) < 30:
            continue
        ctx.today = raw_d.date()
        ctx.ohlcv = {"SPY": hist}
        ctx.spy_returns = hist["close"].pct_change().dropna().values
        for task in tasks:
            task.run(ctx)
        evidence = dict(getattr(ctx, "_regime_evidence", {}) or {})
        out.append({
            "date": raw_d,
            "regime": ctx.regime,
            "confidence": ctx.confidence,
            "source": evidence.get("source"),
            "hurst": evidence.get("hurst"),
            "hurst_regime": evidence.get("hurst_regime"),
            "hard_bear": evidence.get("hard_bear"),
            "vol_cluster_choppy": evidence.get("vol_cluster_choppy"),
            "in_transition": evidence.get("in_transition"),
        })
    return pd.DataFrame(out)


def regime_diagnostics(
    val: pd.DataFrame,
    mu: pd.Series,
    label: str,
    regimes: pd.DataFrame,
    *,
    min_names: int = 5,
) -> dict[str, Any]:
    enriched = val.copy()
    enriched["mu"] = np.asarray(mu.loc[val.index], dtype=float)
    enriched["date"] = pd.to_datetime(enriched["date"]).dt.normalize()
    r = regimes.copy()
    if r.empty:
        enriched["regime"] = "UNKNOWN"
    else:
        r["date"] = pd.to_datetime(r["date"]).dt.normalize()
        enriched = enriched.merge(r[["date", "regime", "confidence", "source"]],
                                  on="date", how="left")
        enriched["regime"] = enriched["regime"].fillna("UNKNOWN")
    out: dict[str, Any] = {}
    for regime, g in enriched.groupby("regime", sort=True):
        stats = summarize_ic(g["mu"], g[label].clip(-0.5, 0.5), g["date"],
                             min_names=min_names)
        stats["n_raw_rows"] = int(len(g))
        stats["mean_confidence"] = _mean(g.get("confidence", pd.Series(dtype=float)))
        out[str(regime)] = stats
    return out


def regime_shift_diagnostics(
    panel: pd.DataFrame,
    val: pd.DataFrame,
    mu: pd.Series,
    label: str,
    regimes: pd.DataFrame,
    *,
    shifts: Iterable[int],
    min_names: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Entry-regime sliced version of shift_diagnostics()."""
    enriched = val.copy()
    enriched["__orig_index"] = val.index
    enriched["date"] = pd.to_datetime(enriched["date"]).dt.normalize()
    r = regimes.copy()
    if r.empty:
        enriched["regime"] = "UNKNOWN"
    else:
        r["date"] = pd.to_datetime(r["date"]).dt.normalize()
        enriched = enriched.merge(r[["date", "regime"]], on="date", how="left")
        enriched["regime"] = enriched["regime"].fillna("UNKNOWN")
    enriched = enriched.set_index("__orig_index", drop=True)
    out: dict[str, list[dict[str, Any]]] = {}
    for regime, g in enriched.groupby("regime", sort=True):
        if g["date"].nunique() < 5:
            continue
        out[str(regime)] = shift_diagnostics(
            panel,
            g,
            mu,
            label,
            shifts=shifts,
            min_names=min_names,
        )
    return out


N_DECILES = 10


def _decile_rank(scores: pd.Series) -> pd.Series:
    """Cross-sectional decile of `scores` within one date, 0 (worst)..9
    (best/top) — decile 9 is the model's top-decile long-side candidates.

    Ported from RenQuant `scripts/regen_oos_pick_table.py` (#430, merged):
    the naive `(rank(pct=True) * 10).astype(int).clip(upper=9)` approach
    breaks on exactly N_DECILES distinct scores (produces ranks 1..10,
    clips 10 to 9, so decile 0 is never populated and decile 9 gets two
    observations) — a common case for small cross-sections, not an edge
    case. Ties are broken by `rank(method="first")` before `qcut` so `qcut`
    bins strictly-ordered integer ranks (never raw floats), making
    `duplicates="drop"` a pure safety net rather than something that
    silently reshuffles bin membership. Falls back to fewer than
    N_DECILES buckets on a date with too few distinct names for 10 clean
    bins (documented, not a crash) — `qcut` needs at least as many distinct
    values as bins.
    """
    n_unique = int(scores.nunique())
    if n_unique < 2:
        return pd.Series(0, index=scores.index, dtype=int)
    n_bins = min(N_DECILES, n_unique)
    ranks = pd.qcut(
        scores.rank(method="first"), n_bins, labels=False, duplicates="drop"
    )
    return ranks.astype(int)


def build_pick_table(
    val: pd.DataFrame,
    mu: pd.Series,
    label: str,
    regimes: pd.DataFrame,
) -> pd.DataFrame:
    """Per-(date,ticker) OOS prediction table for downstream research.

    Columns: date, ticker, score (the manifest-scored mu), <label>, regime,
    decile_rank (per-date score decile, 9 = top). This is the durable pick
    table the Track-A conditional test consumes (orchestrator direction
    decision §4); the scoring stays HERE so faithfulness to the gate's own
    evaluation holds by construction.

    `mu` is aligned to `val`'s index EXPLICITLY (never `.to_numpy()`, which
    discards the index and does positional assignment — if `mu`'s labels
    were the same set as `val`'s but in a different order, that silently
    attaches each score to the wrong ticker). Missing or duplicate index
    entries in `mu` relative to `val` fail loudly rather than producing
    misaligned output.
    """
    if mu.index.has_duplicates:
        dupes = mu.index[mu.index.duplicated()].unique().tolist()
        raise ValueError(f"mu has duplicate index entries: {dupes[:10]}")
    missing = val.index.difference(mu.index)
    if len(missing) > 0:
        raise ValueError(
            f"mu is missing {len(missing)} index entries present in val "
            f"(e.g. {missing[:10].tolist()}) — cannot align scores to tickers"
        )
    out = val[["date", "ticker", label]].copy()
    out["score"] = pd.to_numeric(mu, errors="coerce").reindex(out.index)
    out = out.dropna(subset=["score"])
    if not regimes.empty:
        out = out.merge(regimes, on="date", how="left")
    else:
        out["regime"] = None

    out["decile_rank"] = out.groupby("date")["score"].transform(_decile_rank)
    return out.sort_values(
        ["date", "score"], ascending=[True, False]
    ).reset_index(drop=True)


def canonical_pick_table_content_hash(table: pd.DataFrame, label: str) -> str:
    """SHA256 of `table`'s CONTENT — the actual scores/labels/regimes, not
    just its shape. Ported from RenQuant `canonical_table_content_hash()`
    (#430, merged) — same two required properties: ORDER-INDEPENDENT
    (canonically re-sorted by (date, ticker) here regardless of caller row
    order) and PLATFORM-STABLE FLOAT REPRESENTATION (fixed 10-decimal-place
    string, not raw float64 bytes, which can differ subtly across
    platforms/numpy versions for "the same" value). Parametrized on `label`
    since this repo's label column name varies by call (unlike #430's
    single hardcoded `fwd_60d_excess`)."""
    canon = table[["date", "ticker", "score", "decile_rank", label, "regime"]].copy()
    canon["date"] = pd.to_datetime(canon["date"]).dt.strftime("%Y-%m-%d")
    canon["ticker"] = canon["ticker"].astype(str)
    canon["score"] = canon["score"].astype(float).map(lambda v: f"{v:.10f}")
    canon["decile_rank"] = canon["decile_rank"].astype(int)
    canon[label] = canon[label].astype(float).map(lambda v: f"{v:.10f}")
    canon["regime"] = canon["regime"].astype(str)
    canon = canon.sort_values(["date", "ticker"]).reset_index(drop=True)
    lines = [
        "|".join(str(v) for v in row)
        for row in canon.itertuples(index=False, name=None)
    ]
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head_commit(repo: Path) -> str | None:
    """Best-effort HEAD sha at generation time — informational only, not
    the provenance anchor (see generator_sha256)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo),
            capture_output=True, text=True, check=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


# Substrings that mark a --dump-predictions target as a recognized live/
# production artifact location, never an appropriate destination for a
# research-only artifact carrying REALIZED forward labels (see
# build_pick_table_manifest's research_only stamp). This is a path-name
# heuristic, not an exhaustive registry (this repo has no canonical
# production-path list to check against, unlike renquant-orchestrator's
# PROD_PATH_RULES) — it exists as an active guard on top of the manifest's
# passive research_only metadata, not a substitute for it.
_PRODUCTION_PATH_MARKERS = ("live", "prod", "production")


class ResearchOnlyOutputPathError(ValueError):
    """Raised when --dump-predictions targets what looks like a live/
    production artifact path without --allow-production-path."""


def _guard_research_only_output_path(path: Path, *, allow_production: bool = False) -> None:
    if allow_production:
        return
    parts = [p.lower() for p in path.resolve().parts]
    for marker in _PRODUCTION_PATH_MARKERS:
        if any(marker in part for part in parts):
            raise ResearchOnlyOutputPathError(
                f"--dump-predictions target {path} contains {marker!r} in its "
                "path, which looks like a live/production artifact location. "
                "This table contains REALIZED forward labels and must never "
                "be written where it could be mistaken for or consumed as a "
                "live inference input. Pass --allow-production-path only if "
                "you have verified this destination is genuinely a research/ "
                "experimental artifact contract, not a live-serving path."
            )


def build_pick_table_manifest(
    table: pd.DataFrame,
    *,
    label: str,
    output_path: Path,
) -> dict:
    """Reproducibility recipe + evidence-provenance manifest for a
    --dump-predictions pick table, in the SAME schema shape RenQuant #430's
    regen_oos_pick_table.py manifest uses (schema/recipe/counts/output
    sections, generator_sha256 as the real provenance anchor,
    output_content_sha256 as the content-reproducibility anchor) — a
    consumer of either artifact sees one consistent contract.

    RESEARCH-ONLY: this table contains REALIZED forward labels (`label`),
    not only model predictions. A downstream consumer must never mistake
    this for a live inference input — the manifest stamps that explicitly,
    machine-readably, not just in prose.
    """
    generator_path = Path(__file__).resolve()
    content_hash = canonical_pick_table_content_hash(table, label)
    parquet_hash = _sha256_file(output_path) if output_path.exists() else None
    return {
        "research_only": True,
        "research_only_note": (
            "this artifact contains REALIZED forward labels, not only "
            "predictions — it must never be consumed as a live inference "
            "input. temporal_semantics below states exactly when each field "
            "was actually knowable."
        ),
        "temporal_semantics": {
            "score": "point-in-time model output, knowable AS OF `date`",
            label: (
                "REALIZED forward-looking outcome, NOT knowable as of `date` "
                "— only knowable once the forward window has elapsed"
            ),
            "regime": "live regime label as of `date`",
        },
        "schema": {
            "columns": ["date", "ticker", "score", "decile_rank", label, "regime"],
            "description": (
                "one row per (date, ticker); score = point-in-time model raw "
                "score (mu); decile_rank = cross-sectional decile within date, "
                f"0(worst)-9(best/top); {label} = REALIZED forward outcome "
                "(research-only, not a live-inference field); regime = live "
                "regime label at pick date"
            ),
        },
        "recipe": {
            "generator": str(
                generator_path.relative_to(REPO)
                if generator_path.is_relative_to(REPO)
                else generator_path
            ),
            "generator_sha256": _sha256_file(generator_path),
            "generator_commit": _git_head_commit(REPO),
            "generator_commit_note": (
                "best-effort `git rev-parse HEAD` at generation time — "
                "informational only, NOT the provenance anchor; "
                "generator_sha256 above is a content hash of the generator's "
                "own bytes and is self-consistent regardless of git history"
            ),
            "label": label,
        },
        "counts": {
            "n_rows": int(len(table)),
            "n_dates": int(table["date"].nunique()),
            "n_tickers": int(table["ticker"].nunique()),
        },
        "output": {
            "output_content_sha256": content_hash,
            "output_content_sha256_note": (
                "the PROVENANCE ANCHOR — sha256 of the table's actual content "
                "(canonically re-sorted by (date, ticker), floats formatted "
                "to a fixed 10-decimal-place string), NOT its shape. A fresh "
                "regeneration's output_content_sha256 must match this stamped "
                "value to prove it reproduced this evidence table's content — "
                "matching n_rows/n_dates/n_tickers alone does not."
            ),
            "output_parquet_sha256": parquet_hash,
            "output_parquet_sha256_note": (
                "a SECONDARY, weaker transport hash of the literal on-disk "
                ".parquet file bytes — detects local file corruption, but is "
                "NOT portable across parquet library/compression-setting "
                "versions even for identical logical content. Use "
                "output_content_sha256 above to verify reproducibility."
            ),
        },
    }


def analyze_manifest(
    *,
    artifact_path: Path,
    manifest_path: Path,
    label: str,
    strategy_dir: Path,
    shifts: Iterable[int],
    min_names: int,
    dump_predictions: Path | None = None,
    allow_production_path: bool = False,
) -> dict[str, Any]:
    logging.getLogger("kernel.panel_pipeline.hf_patchtst_scorer").setLevel(logging.WARNING)
    logging.getLogger("kernel.panel_pipeline.patchtst_scorer").setLevel(logging.WARNING)
    artifact = _load_artifact_payload(artifact_path)
    if str(label).lower() in {"", "auto"}:
        label = _sanity_model_label_col(artifact)
    feat_cols = list(artifact.get("feature_cols") or [])
    if not feat_cols:
        raise ValueError(f"artifact missing feature_cols: {artifact_path}")
    panel, panel_meta = _load_sanity_panel(feat_cols, label)
    panel = panel.dropna(subset=[label]).copy()
    panel["date"] = pd.to_datetime(panel["date"])
    distinct = sorted(panel["date"].unique())
    val_cut = pd.Timestamp(distinct[int(len(distinct) * 0.8)])
    val = panel[panel["date"] > val_cut].copy()
    if val.empty:
        raise ValueError("empty validation partition")
    mu, score_meta = _score_manifest_sanity(
        val,
        feat_cols,
        manifest_path,
        artifact_path,
        artifact,
        panel_history=panel,
    )
    val = val.loc[mu.index].copy()
    mu = mu.loc[val.index]
    y = val[label].clip(-0.5, 0.5)
    real = summarize_ic(mu, y, val["date"], min_names=min_names)
    shifts_out = shift_diagnostics(panel, val, mu, label, shifts=shifts,
                                   min_names=min_names)
    regimes = build_regime_series(val["date"].unique(), strategy_dir=strategy_dir)
    if dump_predictions is not None:
        _guard_research_only_output_path(
            dump_predictions, allow_production=allow_production_path,
        )
        table = build_pick_table(val, mu, label, regimes)
        dump_predictions.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(dump_predictions, index=False)
        manifest_out = build_pick_table_manifest(
            table, label=label, output_path=dump_predictions,
        )
        manifest_path_out = dump_predictions.with_suffix(".manifest.json")
        manifest_path_out.write_text(json.dumps(manifest_out, indent=2, sort_keys=True) + "\n")
        logging.getLogger(__name__).info(
            "dump-predictions: wrote %d rows / %d dates -> %s (manifest: %s, "
            "output_content_sha256=%s)",
            len(table), table["date"].nunique(), dump_predictions,
            manifest_path_out, manifest_out["output"]["output_content_sha256"][:12],
        )
    by_regime = regime_diagnostics(val, mu, label, regimes, min_names=min_names)
    by_regime_shift = regime_shift_diagnostics(
        panel,
        val,
        mu,
        label,
        regimes,
        shifts=shifts,
        min_names=min_names,
    )
    placebo_60 = next((r for r in shifts_out if r["shift_days"] == 60), {})
    p60 = placebo_60.get("model_placebo_ic")
    aligned_real_60 = placebo_60.get("aligned_real_ic")
    label_auto60 = placebo_60.get("label_autocorr_ic")
    promotion_evidence = (
        aligned_real_60 is not None
        and abs(float(aligned_real_60)) >= 0.005
        and p60 is not None
        and abs(float(p60)) < max(0.005, 0.5 * abs(float(aligned_real_60)))
    )
    return {
        "artifact": str(artifact_path),
        "manifest": str(manifest_path),
        "label": label,
        "feature_count": len(feat_cols),
        "panel_meta": panel_meta,
        "score_meta": score_meta,
        "validation": {
            "val_cut": val_cut.date().isoformat(),
            "start": pd.Timestamp(val["date"].min()).date().isoformat(),
            "end": pd.Timestamp(val["date"].max()).date().isoformat(),
            "n_rows": int(len(val)),
            "n_dates": int(val["date"].nunique()),
            "n_tickers": int(val["ticker"].nunique()),
        },
        "real_ic": real,
        "shift_diagnostics": shifts_out,
        "by_regime": by_regime,
        "by_regime_shift_diagnostics": by_regime_shift,
        "regime_counts": (
            regimes["regime"].value_counts(dropna=False).to_dict()
            if not regimes.empty else {}
        ),
        "interpretation": {
            "promotion_evidence": bool(promotion_evidence),
            "aligned_real_60_ic": aligned_real_60,
            "placebo_60_ic": p60,
            "label_autocorr_60_ic": label_auto60,
            "primary_warning": (
                "60-day placebo is too large relative to aligned real IC"
                if p60 is not None and aligned_real_60 is not None
                and abs(float(p60)) >= max(0.005, 0.5 * abs(float(aligned_real_60)))
                else None
            ),
        },
    }


def _fmt(v: Any, ndigits: int = 4) -> str:
    if v is None:
        return "NA"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return "NA"
    return f"{f:+.{ndigits}f}"


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# WF Sanity Placebo Diagnostic",
        "",
        f"- Artifact: `{result['artifact']}`",
        f"- Manifest: `{result['manifest']}`",
        f"- Label: `{result['label']}`",
        f"- Validation: {result['validation']['start']} to {result['validation']['end']} "
        f"({result['validation']['n_dates']} dates, {result['validation']['n_rows']} rows)",
        f"- Promotion evidence: `{result['interpretation']['promotion_evidence']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Real mean IC | {_fmt(result['real_ic'].get('mean_ic'))} |",
        f"| 60d aligned real IC | {_fmt(result['interpretation'].get('aligned_real_60_ic'))} |",
        f"| 60d model-placebo IC | {_fmt(result['interpretation'].get('placebo_60_ic'))} |",
        f"| 60d label autocorr IC | {_fmt(result['interpretation'].get('label_autocorr_60_ic'))} |",
        f"| Warning | {result['interpretation'].get('primary_warning') or 'none'} |",
        "",
        "## Shift Profile",
        "",
        "| Shift | Aligned real IC | Model-placebo IC | Label autocorr IC | Rows | Dates |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["shift_diagnostics"]:
        lines.append(
            f"| {row['shift_days']} | {_fmt(row.get('aligned_real_ic'))} | "
            f"{_fmt(row.get('model_placebo_ic'))} | "
            f"{_fmt(row.get('label_autocorr_ic'))} | {row.get('n_rows', 0)} | "
            f"{row.get('n_dates', 0)} |"
        )
    lines.extend([
        "",
        "## By Regime",
        "",
        "| Regime | Mean IC | Hit Rate | Dates | Rows | Mean Conf |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for regime, stats in sorted(result["by_regime"].items()):
        lines.append(
            f"| {regime} | {_fmt(stats.get('mean_ic'))} | "
            f"{_fmt(stats.get('hit_rate'))} | {stats.get('n_dates', 0)} | "
            f"{stats.get('n_raw_rows', 0)} | {_fmt(stats.get('mean_confidence'))} |"
        )
    if result.get("by_regime_shift_diagnostics"):
        lines.extend([
            "",
            "## 60d Placebo By Regime",
            "",
            "| Regime | Model-placebo IC | Label autocorr IC | Rows | Dates |",
            "|---|---:|---:|---:|---:|",
        ])
        for regime, rows in sorted(result["by_regime_shift_diagnostics"].items()):
            row60 = next((r for r in rows if r.get("shift_days") == 60), None)
            if not row60:
                continue
            lines.append(
                f"| {regime} | {_fmt(row60.get('model_placebo_ic'))} | "
                f"{_fmt(row60.get('label_autocorr_ic'))} | "
                f"{row60.get('n_rows', 0)} | {row60.get('n_dates', 0)} |"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument(
        "--label",
        default="auto",
        help=(
            "Label for IC/placebo diagnostics. Default 'auto' uses the "
            "artifact label_col; pass fwd_60d_excess_raw for return-scale "
            "expected-return diagnostics."
        ),
    )
    ap.add_argument("--strategy-dir", default=str(STRATEGY_DIR))
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--shifts", default=",".join(str(x) for x in DEFAULT_SHIFTS))
    ap.add_argument("--min-names", type=int, default=5)
    ap.add_argument(
        "--dump-predictions",
        default="",
        help=("optional parquet path: write the per-(date,ticker) OOS "
              "prediction table {date,ticker,score,<label>,regime,decile_rank} "
              "(the durable pick table for downstream research)"),
    )
    ap.add_argument(
        "--allow-production-path",
        action="store_true",
        help=("bypass the research-only path guard on --dump-predictions "
              "(the output contains REALIZED forward labels; only pass this "
              "if the destination is genuinely a verified research/exp "
              "artifact contract, not a live-serving path)"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    artifact = Path(args.artifact).resolve()
    manifest = Path(args.manifest).resolve()
    strategy_dir = Path(args.strategy_dir).resolve()
    shifts = [int(x) for x in str(args.shifts).split(",") if x.strip()]
    result = analyze_manifest(
        artifact_path=artifact,
        manifest_path=manifest,
        label=args.label,
        strategy_dir=strategy_dir,
        shifts=shifts,
        min_names=int(args.min_names),
        dump_predictions=(
            Path(args.dump_predictions).resolve()
            if str(args.dump_predictions).strip() else None
        ),
        allow_production_path=bool(args.allow_production_path),
    )
    out_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else strategy_dir / "artifacts" / "diagnostics" / (
            "sanity_placebo_" + date.today().strftime("%Y%m%d")
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = artifact.stem.replace(".", "_")
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(result, indent=2, default=_json_default))
    md_path.write_text(render_markdown(result))
    print(json.dumps({
        "json": str(json_path),
        "markdown": str(md_path),
        "real_ic": result["real_ic"].get("mean_ic"),
        "aligned_real_60_ic": result["interpretation"].get("aligned_real_60_ic"),
        "placebo_60_ic": result["interpretation"].get("placebo_60_ic"),
        "label_autocorr_60_ic": result["interpretation"].get("label_autocorr_60_ic"),
        "promotion_evidence": result["interpretation"]["promotion_evidence"],
    }, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
