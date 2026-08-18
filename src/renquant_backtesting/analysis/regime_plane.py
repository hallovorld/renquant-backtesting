"""ONE production regime-label plane for research surfaces.

Implements renquant-orchestrator #985 (regime-detector assessment) ranked
item 1: research surfaces must consume regime labels from the SAME task
chain that decides serving — not from the stateless approximation in
``renquant_common.hmm_regime_labels`` (25-70% same-day agreement across the
four historical label planes; memo P2/P8).

The production chain replayed here is ``renquant_pipeline.kernel.regime`` +
``renquant_pipeline.kernel.pipeline.task_regime`` — the lifted copies the WF
gate's ``sanity_regime_ic`` leg has always replayed (via
``analyze_manifest_sanity_placebo.build_regime_series``, which now delegates
to :func:`production_regime_labels` so there is exactly one code path).

Plane selection contract (mirrored read-only in renquant-model, which may
not import this repo — boundary tests + codex round-2 on model#65):

* ``RENQUANT_REGIME_PLANE`` env var — ``production`` (default) or
  ``legacy_stateless`` (reproduction of historical results only).
* The committed per-date corpus for import-restricted consumers lives at
  :data:`PRODUCTION_REGIME_CORPUS_RELPATH`, published by
  ``tools/publish_production_regime_labels.py`` with a provenance manifest.

This module is research metrology ONLY: it reads the serving config, GMM
artifact and SPY data read-only and never writes a production path.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

REGIME_PLANE_ENV = "RENQUANT_REGIME_PLANE"
PLANE_PRODUCTION = "production"
PLANE_LEGACY_STATELESS = "legacy_stateless"
VALID_PLANES = (PLANE_PRODUCTION, PLANE_LEGACY_STATELESS)

#: Committed production-plane label corpus (repo-relative), consumed BY PATH
#: by import-restricted repos (renquant-model) and by the upcoming
#: serving-plane power-map re-derivation (orch#984 consequence (a)).
PRODUCTION_REGIME_CORPUS_RELPATH = Path(
    "doc/research/data/production_regime_labels.csv"
)
PRODUCTION_REGIME_MANIFEST_RELPATH = Path(
    "doc/research/data/production_regime_labels.manifest.json"
)


def resolve_regime_plane(environ: Any | None = None) -> str:
    """Resolve the active regime label plane from the environment.

    Fail-closed: an unrecognized value raises instead of silently falling
    back to either plane (a typo must not silently change the metrology).
    """
    env = os.environ if environ is None else environ
    raw = str(env.get(REGIME_PLANE_ENV, PLANE_PRODUCTION) or PLANE_PRODUCTION)
    if raw not in VALID_PLANES:
        raise ValueError(
            f"{REGIME_PLANE_ENV}={raw!r} is not a valid regime plane; "
            f"expected one of {VALID_PLANES}"
        )
    return raw


def _runner():
    """The wf_gate runner module, resolved at call time.

    Call-time resolution keeps ``REPO`` / ``STRATEGY_DIR`` monkeypatchable in
    tests and avoids freezing the paths at import order.
    """
    from renquant_backtesting.wf_gate import runner  # noqa: PLC0415

    return runner


def resolve_strategy_artifact(strategy_dir: Path, raw: str | None) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    candidates = [p] if p.is_absolute() else [
        strategy_dir / "artifacts" / p,
        strategy_dir / p,
        _runner().REPO / p,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def load_strategy_config(strategy_dir: Path) -> dict:
    return json.loads((strategy_dir / "strategy_config.json").read_text())


def load_gmm_artifact(strategy_dir: Path, config: dict) -> dict | None:
    p = resolve_strategy_artifact(
        strategy_dir,
        str((config.get("regime", {}) or {}).get("gmm_artifact") or ""),
    )
    if p is None or not p.exists():
        return None
    return json.loads(p.read_text())


def load_spy_frame(spy_path: Path | None = None) -> pd.DataFrame:
    path = spy_path or (_runner().REPO / "data" / "ohlcv" / "SPY" / "1d.parquet")
    df = pd.read_parquet(path)
    if "date" not in df.columns:
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").set_index("date")


def production_regime_labels(
    dates: Iterable[Any],
    spy_ohlcv: pd.DataFrame | None = None,
    *,
    strategy_dir: Path | None = None,
    config: dict | None = None,
    gmm: dict | None = None,
) -> pd.DataFrame:
    """Run the production regime Task chain for each requested date.

    This is the ONE accessor for production-plane labels (orch#985 item 1).
    The WF gate's sanity leg calls it through ``build_regime_series``; the
    corpus publisher calls it directly. ``spy_ohlcv`` / ``config`` / ``gmm``
    are injectable for tests; the defaults read the serving strategy config,
    its GMM artifact and the SPY parquet read-only.

    State (CUSUM cooldown, previous regime) is carried ACROSS the requested
    dates in ascending order — callers wanting the serving trajectory must
    pass the full contiguous date range, exactly as the sanity replay does.
    """
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

    if strategy_dir is None:
        strategy_dir = _runner().STRATEGY_DIR
    if config is None:
        config = load_strategy_config(strategy_dir)
    if gmm is None:
        gmm = load_gmm_artifact(strategy_dir, config)
    spy = spy_ohlcv if spy_ohlcv is not None else load_spy_frame()
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


def repo_corpus_path() -> Path:
    """Absolute path of the committed corpus inside THIS repo checkout."""
    return Path(__file__).resolve().parents[3] / PRODUCTION_REGIME_CORPUS_RELPATH


def repo_manifest_path() -> Path:
    return Path(__file__).resolve().parents[3] / PRODUCTION_REGIME_MANIFEST_RELPATH


def load_production_regime_corpus(path: Path | None = None) -> pd.DataFrame:
    """Load the committed production-plane label series (date parsed)."""
    p = path or repo_corpus_path()
    if not p.exists():
        raise FileNotFoundError(
            f"production regime label corpus missing at {p}; regenerate with "
            f"tools/publish_production_regime_labels.py"
        )
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


def sha256_bytes_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
