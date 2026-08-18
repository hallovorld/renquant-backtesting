#!/usr/bin/env python3
"""Publish the committed production-plane regime label corpus.

Implements the data-plane channel of renquant-orchestrator #985 ranked
item 1: import-restricted consumers (renquant-model harnesses, whose
boundary tests forbid importing this repo or renquant_pipeline) consume
production-chain regime labels BY PATH from the committed corpus this
tool emits, instead of recomputing a divergent stateless approximation.

Deterministic by construction: for fixed inputs (SPY parquet, strategy
config, GMM artifact, chain code) two runs produce byte-identical CSV
output and the same series sha256. The manifest records every input's
sha256, the replayed chain-copy identity, the config flag-state under
which the two textually divergent chain copies (umbrella working copy vs
renquant-pipeline lifted copy) are behaviorally equal, date bounds, row
count and the series sha256.

Fail-closed cross-check: when the published window covers the #985 memo
replica span (2016-11-01..2026-08-14) the corpus MUST reproduce the
memo's committed replica facts (2,459 trading days, 413 BEAR days,
57 BEAR episodes) or the publish aborts.

Read-only on production surfaces: SPY parquet, strategy config, GMM
artifact and the served panel artifact are only ever read.

Usage (from the repo root, sibling-layout machine):
    ../RenQuant/.venv/bin/python tools/publish_production_regime_labels.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

#: #985 memo replica span + committed facts the corpus must reproduce
#: [VERIFIED — doc/research/2026-08-17-regime-detector-assessment.md
#:  M:span, M:stats_S.n_days, M:stats_S.occupancy.BEAR,
#:  M:bear_episodes_serving (57 rows) in renquant-orchestrator].
MEMO_SPAN = ("2016-11-01", "2026-08-14")
MEMO_N_DAYS = 2459
MEMO_BEAR_DAYS = 413
MEMO_BEAR_EPISODES = 57

CSV_COLUMNS = [
    "date",
    "regime",
    "confidence",
    "source",
    "hurst",
    "hurst_regime",
    "hard_bear",
    "vol_cluster_choppy",
    "in_transition",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_root_of(path: Path) -> Path | None:
    """Nearest ancestor that is a repo root (pyproject.toml or .git)."""
    for anc in path.resolve().parents:
        if (anc / "pyproject.toml").exists() or (anc / ".git").exists():
            return anc
    return None


def _normalize_path(path: Path, roots: list[Path]) -> str:
    """``<repo>:<relpath>`` id (memo convention) so the committed manifest
    regenerates machine-independently; absolute only as a last resort."""
    rp = path.resolve()
    candidates = [r for r in roots if r is not None]
    own_root = _repo_root_of(rp)
    if own_root is not None:
        candidates.append(own_root)
    for root in candidates:
        try:
            return f"{root.name}:{rp.relative_to(root.resolve())}"
        except ValueError:
            continue
    return str(rp)


def _bear_episodes(regimes: pd.Series) -> int:
    is_bear = (regimes == "BEAR").to_numpy()
    if is_bear.size == 0:
        return 0
    starts = int(is_bear[0]) + int(((~is_bear[:-1]) & is_bear[1:]).sum())
    return starts


def _chain_identity() -> dict:
    """Path + sha256 of the chain modules actually imported for the replay."""
    import renquant_pipeline.kernel.pipeline.task_regime as task_regime
    import renquant_pipeline.kernel.regime as regime
    import renquant_pipeline.kernel.regime_hmm as regime_hmm

    out = {}
    for mod in (regime, task_regime, regime_hmm):
        p = Path(mod.__file__).resolve()
        out[mod.__name__] = {"path": _normalize_path(p, []), "sha256": _sha256(p)}
    return out


def _render_csv(labels: pd.DataFrame) -> str:
    df = labels.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[CSV_COLUMNS].to_csv(
        index=False, float_format="%.10g", lineterminator="\n"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2016-01-04")
    ap.add_argument("--end", default=None,
                    help="default: latest available SPY bar")
    ap.add_argument("--spy", default=None,
                    help="SPY parquet override (default: <umbrella>/data/ohlcv/SPY/1d.parquet)")
    ap.add_argument("--strategy-dir", default=None)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-manifest", default=None)
    args = ap.parse_args(argv)

    from renquant_backtesting.analysis.regime_plane import (
        PRODUCTION_REGIME_CORPUS_RELPATH,
        PRODUCTION_REGIME_MANIFEST_RELPATH,
        load_gmm_artifact,
        load_spy_frame,
        load_strategy_config,
        production_regime_labels,
        resolve_strategy_artifact,
    )
    from renquant_backtesting.wf_gate import runner as _runner

    strategy_dir = Path(args.strategy_dir) if args.strategy_dir else _runner.STRATEGY_DIR
    spy_path = Path(args.spy) if args.spy else (
        _runner.REPO / "data" / "ohlcv" / "SPY" / "1d.parquet"
    )
    out_csv = Path(args.out_csv) if args.out_csv else (
        REPO_ROOT / PRODUCTION_REGIME_CORPUS_RELPATH
    )
    out_manifest = Path(args.out_manifest) if args.out_manifest else (
        REPO_ROOT / PRODUCTION_REGIME_MANIFEST_RELPATH
    )

    config = load_strategy_config(strategy_dir)
    gmm = load_gmm_artifact(strategy_dir, config)
    spy = load_spy_frame(spy_path)

    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end) if args.end else spy.index.max()
    dates = spy.index[(spy.index >= start_ts) & (spy.index <= end_ts)]
    if len(dates) == 0:
        print(f"ERROR: no SPY dates in [{start_ts.date()}, {end_ts.date()}]")
        return 2

    labels = production_regime_labels(
        dates, spy, strategy_dir=strategy_dir, config=config, gmm=gmm,
    )
    if labels.empty:
        print("ERROR: replay produced no labels")
        return 2

    csv_text = _render_csv(labels)
    series_sha = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()

    # ── #985 memo replica cross-check (fail-closed when applicable) ──
    memo_start, memo_end = pd.Timestamp(MEMO_SPAN[0]), pd.Timestamp(MEMO_SPAN[1])
    applicable = bool(labels["date"].min() <= memo_start
                      and labels["date"].max() >= memo_end)
    crosscheck: dict = {
        "span": list(MEMO_SPAN),
        "expected": {
            "n_days": MEMO_N_DAYS,
            "bear_days": MEMO_BEAR_DAYS,
            "bear_episodes": MEMO_BEAR_EPISODES,
        },
        "evaluated": applicable,
    }
    if applicable:
        window = labels[(labels["date"] >= memo_start) & (labels["date"] <= memo_end)]
        measured = {
            "n_days": int(len(window)),
            "bear_days": int((window["regime"] == "BEAR").sum()),
            "bear_episodes": _bear_episodes(window["regime"].reset_index(drop=True)),
        }
        crosscheck["measured"] = measured
        crosscheck["exact_match"] = measured == crosscheck["expected"]
        if not crosscheck["exact_match"]:
            print("ERROR: memo replica cross-check FAILED — refusing to publish:")
            print(json.dumps(crosscheck, indent=2))
            return 3

    # ── chain flag-state under which umbrella/pipeline copies agree ──
    regime_cfg = config.get("regime", {}) or {}
    flag_state = {
        "bear_short_route_require_both": bool(
            regime_cfg.get("bear_short_route_require_both", False)
        ),
        "bear_trend_filter_enabled": bool(
            (regime_cfg.get("bear_trend_filter", {}) or {}).get("enabled", False)
        ),
        "note": (
            "the renquant-pipeline lifted task_regime.py carries 2026-06-11 "
            "bear-audit guards the umbrella working copy lacks; both copies "
            "are behaviorally equal only while these flags stay unset/false "
            "(see the drift issue filed on hallovorld/renquant-pipeline)"
        ),
    }

    gmm_path = resolve_strategy_artifact(
        strategy_dir, str(regime_cfg.get("gmm_artifact") or ""),
    )
    occupancy = {
        str(k): int(v) for k, v in labels["regime"].value_counts().items()
    }
    manifest = {
        "schema": "production_regime_labels/v1",
        "generated_on": date.today().isoformat(),
        "generator": "tools/publish_production_regime_labels.py",
        "plane": "production",
        "chain": _chain_identity(),
        "chain_flag_state": flag_state,
        "inputs": {
            "spy_parquet": {
                "path": _normalize_path(spy_path, [_runner.REPO]),
                "sha256": _sha256(spy_path),
            },
            "strategy_config": {
                "path": _normalize_path(
                    strategy_dir / "strategy_config.json", [_runner.REPO],
                ),
                "sha256": _sha256(strategy_dir / "strategy_config.json"),
            },
            "gmm_artifact": (
                {
                    "path": _normalize_path(gmm_path, [_runner.REPO]),
                    "sha256": _sha256(gmm_path),
                }
                if gmm_path is not None and gmm_path.exists() else None
            ),
        },
        "date_start": str(labels["date"].min().date()),
        "date_end": str(labels["date"].max().date()),
        "row_count": int(len(labels)),
        "occupancy": occupancy,
        "series_sha256": series_sha,
        "memo_985_crosscheck": crosscheck,
        "consumers": [
            "renquant-model harnesses (by path; boundary tests forbid imports)",
            "orch#984 serving-plane power-map re-derivation",
        ],
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text(csv_text)
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_csv} ({len(labels)} rows, {labels['date'].min().date()}"
          f"..{labels['date'].max().date()}, sha256={series_sha[:16]}…)")
    print(f"wrote {out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
