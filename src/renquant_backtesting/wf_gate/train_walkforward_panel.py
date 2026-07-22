#!/usr/bin/env python
"""Walk-forward panel-LTR training driver (Track P3-v2, 2026-05-10).

Trains one alpha158 panel-LTR artifact per `retrain_date` in
[--start-date, --end-date] by subprocess-invoking
``scripts/train_production_model.py --train-cutoff <date>`` for each
cutoff, and emits a manifest indexed by cutoff_date.

This is the v2 path. v1 (legacy PanelTrainingPipeline / 21-feat) is
deprecated: it trained the legacy 21-feature artifact while SimAdapter
feeds the production alpha158 169-feature panel, producing 100% NaN
predictions. v2 calls the same single-source-of-truth alpha158 training
script that daily prod retrain uses (§5.13.5), guaranteeing feature-shape
parity with SimAdapter.

Sim adapters bind to the manifest via
``kernel.walk_forward.WalkForwardModelLoader.model_as_of(today)``. No
look-ahead leakage: every model used at sim bar `t` was trained
strictly before `t`.

Usage::

    # Dry-run: print the retrain dates without training
    python scripts/train_walkforward_panel.py \\
        --start-date 2024-01-01 --end-date 2026-03-26 \\
        --cadence-days 21 --dry-run

    # Real walk-forward training (≈ 1-2 min per cutoff × N cutoffs)
    python scripts/train_walkforward_panel.py \\
        --start-date 2024-01-01 --end-date 2026-03-26 \\
        --cadence-days 21 \\
        --manifest-output artifacts/walkforward_manifest_v2.json

CLAUDE.md §5.10 hardware saturation: train_production_model.py uses
xgb_params.nthread=8 internally; the subprocess inherits OMP/MKL/OPENBLAS
env vars set here.

CLAUDE.md §5.13.13 isolation: per-cutoff artifacts land under
``backtesting/renquant_104/artifacts/walkforward_v2/<cutoff>/`` so they
cannot collide with v1 (``walkforward/``) or production artifacts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# §5.10 hardware saturation — exported to subprocess env.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "10")

import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
TRAIN_PROD_SCRIPT = REPO_ROOT / "scripts" / "train_production_model.py"
CALIBRATOR_SCRIPT = REPO_ROOT / "scripts" / "fit_calibrator_alpha158_fund.py"
WF_V2_SUBDIR = "walkforward_v2"
# The XGB training panel the umbrella train_production_model.py reads (169-feat
# alpha158 + 5-fund, fwd_60d_excess label). The freshness gate below loads its
# date coverage; it does NOT change what the training subprocess reads.
DEFAULT_DATASET = "data/alpha158_291_fundamental_dataset.parquet"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(STRATEGY_DIR))

from renquant_backtesting.repo_root import resolve_repo_root  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("train-walkforward")


# ── Pure helpers (§1c — small, single-responsibility, ≤ 50 lines each) ──

def compute_retrain_dates(
    start: pd.Timestamp, end: pd.Timestamp, cadence_days: int,
) -> list[pd.Timestamp]:
    """Return retrain cutoff dates spanning [start, end] at cadence_days."""
    if cadence_days <= 0:
        raise ValueError(f"cadence_days must be > 0, got {cadence_days}")
    return list(pd.date_range(start, end, freq=f"{cadence_days}D"))


def make_artifact_path(strategy_dir: Path, cutoff: pd.Timestamp) -> Path:
    """Per-cutoff artifact path: artifacts/walkforward_v2/<YYYY-MM-DD>/panel-ltr.json."""
    sub = strategy_dir / "artifacts" / WF_V2_SUBDIR / cutoff.date().isoformat()
    sub.mkdir(parents=True, exist_ok=True)
    return sub / "panel-ltr.json"


def make_calibrator_path(artifact_path: Path) -> Path:
    """The per-fold calibrator lives beside its scorer artifact."""
    return artifact_path.with_name("panel-rank-calibration.json")


def infer_label_lookahead_days(label: str | None) -> int:
    import re
    m = re.search(r"fwd_(\d+)d", str(label or "fwd_60d_excess"))
    return int(m.group(1)) if m else 60


# ── AC7 training-panel freshness/coverage gate (GOAL-5) ─────────────────────
# Sibling of the PatchTST driver's gate (train_walkforward_patchtst.py). The
# per-fold path only rejects an EMPTY post-cutoff slice; a stale-but-nonempty
# panel that stops short of a fold's ``data_end`` (each fold trains on rows
# strictly before ``cutoff − label_horizon`` business days — see
# fit_calibrator_for_cutoff / train_production_model.py) silently trains on a
# truncated window. Before dispatching ANY cutoff we assert the ONE canonical
# renquant-common freshness contract over the union window ``max(data_end)`` —
# fail-closed, never a silent proceed. The GBDT recipe's label is
# ``fwd_60d_excess`` (horizon 60), matching PatchTST, so the same defaults hold.
def data_end_for_cutoff(cutoff: pd.Timestamp, label: str | None) -> str:
    """Latest training-row date a fold at ``cutoff`` uses: cutoff − horizon BDays.

    Mirrors the embargo the training subprocess applies (rows strictly before
    ``cutoff − label_lookahead`` business days) so the freshness gate checks the
    exact window the folds train on.
    """
    lookahead = infer_label_lookahead_days(label)
    return (cutoff - pd.offsets.BDay(lookahead)).date().isoformat()


def required_through_date(dates: list[pd.Timestamp],
                          label: str | None) -> pd.Timestamp:
    """Latest ``data_end`` any fold needs: ``max(data_end_for_cutoff(c))``.

    A panel that stops short of this date silently truncates the most-recent
    folds, so it is the load-bearing input to the pre-dispatch freshness gate.
    """
    if not dates:
        raise ValueError("required_through_date: no retrain dates")
    return max(pd.Timestamp(data_end_for_cutoff(c, label)) for c in dates)


def resolve_dataset_path(args: argparse.Namespace) -> Path:
    """Absolute path to the XGB training panel. ``--dataset`` is relative to the
    umbrella repo root (matching how the training subprocess, run with
    ``cwd=<repo-root>``, resolves it)."""
    ds = Path(args.dataset)
    if ds.is_absolute():
        return ds
    return resolve_repo_root(getattr(args, "repo_root", None)) / ds


def assert_training_panel_fresh(args: argparse.Namespace,
                                dates: list[pd.Timestamp]) -> None:
    """Fail-closed AC7 gate: verify the training panel COVERS the window every
    fold needs (+ density floors) BEFORE dispatching any cutoff. Raises with the
    contract's reasons rather than let a silently-truncated/thin panel train.
    """
    from renquant_common.training_freshness import (  # noqa: PLC0415
        assess_training_panel_freshness,
    )
    required = required_through_date(dates, args.label)
    dataset_path = resolve_dataset_path(args)
    if not dataset_path.exists():
        raise FileNotFoundError(
            "train_walkforward_panel: training panel not found at "
            f"{dataset_path} (--dataset={args.dataset}); cannot verify "
            "freshness/coverage before dispatch"
        )
    max_gap = int(args.max_gap_days)
    verdict = assess_training_panel_freshness(
        dataset_path,
        required_through_date=required,
        min_tickers_per_day=int(args.min_tickers_per_day),
        min_rows=int(args.min_rows),
        max_gap_days=(None if max_gap <= 0 else max_gap),
        max_staleness_days=(int(args.max_staleness_days)
                            if args.max_staleness_days is not None else None),
    )
    if not verdict.ok:
        panel_max = verdict.max_date.date() if verdict.max_date else None
        raise RuntimeError(
            "train_walkforward_panel: FAIL-CLOSED — training panel "
            f"{dataset_path} does not satisfy the AC7 freshness/coverage gate "
            f"(required_through_date={required.date()}, panel_max_date="
            f"{panel_max}, n_rows={verdict.n_rows}). Reasons: "
            + "; ".join(verdict.reasons)
            + ". Refuse to train on a silently-truncated/thin panel — refresh "
            "the panel, or for a deliberately historical run lower --end-date "
            "so the required window matches the panel (or relax the floors)."
        )
    log.info(
        "AC7 freshness gate PASS: panel=%s covers required_through_date=%s "
        "(max_date=%s, n_days=%d, n_rows=%d, min_tickers/day=%s, max_gap=%s)",
        dataset_path, required.date(), verdict.max_date.date(),
        verdict.n_days, verdict.n_rows, verdict.min_tickers_per_day_observed,
        verdict.max_gap_days_observed,
    )


def configure_panel_cutoff(cfg: dict, cutoff: pd.Timestamp,
                           artifact_path: Path) -> dict:
    """LEGACY v1 helper — retained for back-compat with regression tests.

    NOTE: v2 driver does NOT use this function. It exists solely so the
    audit-regression suite in tests/test_walkforward_artifact_isolation.py
    keeps passing — that test pins the v1 invariant that BOTH
    panel_ltr.artifact_path AND ranking.panel_scoring.artifact_path must
    point at the per-cutoff walkforward path.

    Per §5.13.13 the path is asserted to contain 'walkforward' to forbid
    accidental production overwrite.
    """
    p_str = str(artifact_path)
    assert "walkforward" in p_str, (
        f"configure_panel_cutoff: artifact_path {p_str!r} does not "
        f"contain 'walkforward' — refusing to risk overwriting "
        f"production artifact"
    )
    pl = cfg.setdefault("panel_ltr", {})
    pl["train_cutoff"] = cutoff.isoformat()
    pl["artifact_path"] = p_str
    rk = cfg.setdefault("ranking", {}).setdefault("panel_scoring", {})
    rk["artifact_path"] = p_str
    rk.setdefault("global_calibration", {})["auto_refresh"] = False
    return cfg


def build_retrain_entry(cutoff: pd.Timestamp, trained_dt: datetime,
                         artifact_uri: str, lookahead_days: int = 60,
                         calibrator_uri: str | None = None,
                         effective_train_cutoff_date: pd.Timestamp | None = None):
    """Build a RetrainEntry — wrapper so callers don't have to import it.

    2026-05-11 Round 3 audit (G3): lookahead_days propagated so the
    on-disk manifest carries the forward-label horizon and the leakage
    guard can enforce `cutoff + lookahead < today` per bar. Default 60
    matches the production training label `fwd_60d_excess` in
    train_production_model.py.
    """
    from renquant_backtesting.walk_forward.loader import RetrainEntry  # noqa: PLC0415
    return RetrainEntry(
        cutoff_date=cutoff,
        trained_date=pd.Timestamp(trained_dt),
        artifact_uri=artifact_uri,
        lookahead_days=int(lookahead_days),
        calibrator_uri=calibrator_uri,
        effective_train_cutoff_date=effective_train_cutoff_date,
    )


# ── Per-cutoff training (subprocesses train_production_model.py) ────────

def train_one_cutoff(cutoff: pd.Timestamp, strategy_dir: Path,
                     label: str | None = None,
                     watchlist_file: str | None = None,
                     artifact_root: str | None = None,
                     fingerprint_config: str | None = None,
                     fit_calibrator: bool = True,
                     calibrator_method: str = "platt") -> tuple[bool, Path, Path | None, str]:
    """Subprocess train_production_model.py for one cutoff.

    Optional args (2026-05-13 Track 6 / Track 1):
        label: --label passthrough (e.g. fwd_5d_excess for horizon retest)
        watchlist_file: --watchlist-file passthrough (wl174 retrained variant)
        artifact_root: override WF_V2_SUBDIR (e.g. 'walkforward_horizon_5d')

    Returns (success, artifact_path, calibrator_path, error_msg). On non-zero exit, success=False
    and the caller logs + continues (does not abort the whole batch).
    """
    cutoff_iso = cutoff.date().isoformat()
    if artifact_root:
        artifact_path = strategy_dir / "artifacts" / artifact_root / cutoff_iso / "panel-ltr.json"
    else:
        artifact_path = make_artifact_path(strategy_dir, cutoff)
    side_label = f"walkforward_v2_{cutoff_iso}"
    if label:
        side_label = f"walkforward_{label.replace('_excess','')}_{cutoff_iso}"
    if watchlist_file:
        side_label = f"walkforward_wl_{cutoff_iso}"
    cmd = [
        sys.executable, str(TRAIN_PROD_SCRIPT),
        "--train-cutoff", cutoff_iso,
        "--output-path", str(artifact_path),
        "--side-label", side_label,
    ]
    if label:
        cmd.extend(["--label", label])
    if watchlist_file:
        cmd.extend(["--watchlist-file", watchlist_file])
    if fingerprint_config:
        cmd.extend(["--fingerprint-config", fingerprint_config])
    log.info("train_one_cutoff: cutoff=%s start  cmd=%s",
             cutoff_iso, " ".join(cmd))
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT), check=False,
            capture_output=True, text=True,
        )
    except Exception as exc:  # noqa: BLE001
        return False, artifact_path, None, f"subprocess.run raised: {exc}"
    elapsed = time.monotonic() - t0
    if proc.returncode != 0:
        msg = f"exit={proc.returncode}; stderr_tail={proc.stderr[-500:]!r}"
        log.error("train_one_cutoff: cutoff=%s FAILED  %.1fs  %s",
                  cutoff_iso, elapsed, msg)
        return False, artifact_path, None, msg
    log.info("train_one_cutoff: cutoff=%s DONE  %.1fs  artifact=%s",
             cutoff_iso, elapsed, artifact_path)
    if not fit_calibrator:
        return True, artifact_path, None, ""
    ok, cal_path, err = fit_calibrator_for_cutoff(
        cutoff,
        artifact_path,
        lookahead_days=infer_label_lookahead_days(label),
        method=calibrator_method,
    )
    if not ok:
        return False, artifact_path, cal_path, err
    return True, artifact_path, cal_path, ""


def fit_calibrator_for_cutoff(
    cutoff: pd.Timestamp,
    artifact_path: Path,
    *,
    lookahead_days: int,
    method: str,
) -> tuple[bool, Path, str]:
    """Fit the matching causal calibrator for one WF scorer artifact."""
    cal_path = make_calibrator_path(artifact_path)
    data_end = (cutoff - pd.offsets.BDay(max(0, int(lookahead_days)))).date().isoformat()
    cmd = [
        sys.executable,
        str(CALIBRATOR_SCRIPT),
        "--scorer-artifact",
        str(artifact_path),
        "--out",
        str(cal_path),
        "--data-end",
        data_end,
        "--method",
        method,
    ]
    log.info("fit_calibrator_for_cutoff: cutoff=%s data_end<%s",
             cutoff.date().isoformat(), data_end)
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT), check=False,
            capture_output=True, text=True,
        )
    except Exception as exc:  # noqa: BLE001
        return False, cal_path, f"calibrator subprocess raised: {exc}"
    if proc.returncode != 0:
        return (
            False,
            cal_path,
            f"calibrator exit={proc.returncode}; stderr_tail={proc.stderr[-500:]!r}",
        )
    return True, cal_path, ""


def read_trained_date(artifact_path: Path) -> datetime:
    """Pull trained_date from the artifact (stamped by train_production_model.py)."""
    art = json.loads(artifact_path.read_text())
    return datetime.fromisoformat(art["trained_date"])


def read_effective_train_cutoff_date(artifact_path: Path) -> pd.Timestamp | None:
    """Return the artifact's feature-row cutoff, if stamped.

    Walk-forward production training receives a selection cutoff, then trains
    only on rows before selection_cutoff - label_lookahead. Stamping this
    effective cutoff into the manifest lets the loader enforce label safety
    without applying the lookahead embargo twice.
    """
    if not artifact_path.exists():
        return None
    raw = json.loads(artifact_path.read_text()).get("effective_train_cutoff_date")
    return pd.Timestamp(raw) if raw else None


# ── CLI driver ──────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start-date", required=True,
                   help="First retrain cutoff (YYYY-MM-DD).")
    p.add_argument("--end-date", required=True,
                   help="Last retrain cutoff (YYYY-MM-DD).")
    p.add_argument("--cadence-days", type=int, default=21,
                   help="Days between retrain cutoffs (default: 21).")
    p.add_argument("--repo-root", default=None,
                   help="Umbrella RenQuant root holding data/ (default: "
                        "$RENQUANT_REPO_ROOT or cwd). Used to resolve the "
                        "training panel for the AC7 freshness gate.")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="XGB training panel parquet the AC7 freshness gate "
                        "inspects (relative to --repo-root; default: "
                        f"{DEFAULT_DATASET}).")
    p.add_argument("--manifest-output",
                   default=str(STRATEGY_DIR / "artifacts" / "walkforward_manifest_v2.json"),
                   help="Where to write the merged manifest JSON (v2 default).")
    p.add_argument("--label", default=None,
                   help="Forward label column to use (default: panel default fwd_60d_excess)")
    p.add_argument("--watchlist-file", default=None,
                   help="JSON config file to filter panel to a custom watchlist")
    p.add_argument("--fingerprint-config", default=None,
                   help="Strategy config whose model-relevant fields are stamped into each WF artifact")
    p.add_argument("--artifact-root", default=None,
                   help="Override artifacts/<root>/ subdirectory (default: walkforward_v2)")
    p.add_argument("--jobs", type=int, default=1,
                   help="Number of cutoff retrains to run concurrently. "
                        "Default 1 preserves historical behavior.")
    p.add_argument("--skip-calibrators", action="store_true",
                   help="Research-only escape hatch. By default each WF scorer "
                        "gets a matching causal calibrator and manifest "
                        "calibrator_uri so strict scoring can run.")
    p.add_argument("--calibrator-method", default="platt",
                   choices=["platt", "isotonic"],
                   help="Method passed to fit_calibrator_alpha158_fund.py.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print retrain dates and exit (no training).")
    p.add_argument(
        "--allow-partial-manifest",
        action="store_true",
        help="Research-only escape hatch. By default any failed cutoff aborts "
             "before writing a partial WF manifest.",
    )
    # ── AC7 training-panel freshness/coverage gate (GOAL-5) ──────────────────
    # Same flags/defaults as train_walkforward_patchtst.py so both WF trainers
    # apply identical discipline.
    p.add_argument("--min-tickers-per-day", type=int, default=20,
                   help="AC7 gate: min distinct tickers required on every "
                        "training-window day (0 disables). Thin days train on "
                        "a degenerate cross-section.")
    p.add_argument("--min-rows", type=int, default=0,
                   help="AC7 gate: min total rows in the panel (0 disables).")
    p.add_argument("--max-gap-days", type=int, default=5,
                   help="AC7 gate: max calendar-day gap between consecutive "
                        "training dates (0 disables; weekends are ≤4d so 5 "
                        "flags a real hole).")
    p.add_argument("--max-staleness-days", type=int, default=None,
                   help="AC7 gate (OFF by default): if set, require the panel "
                        "to reach within N days of today. WF corpora train on "
                        "historical windows, so COVERAGE — not calendar "
                        "recency — is the load-bearing check.")
    return p.parse_args(argv)


def train_cutoffs(retrain_dates: list[pd.Timestamp],
                  args: argparse.Namespace) -> tuple[list, list[tuple[str, str]]]:
    """Train all requested cutoffs, optionally in parallel."""
    jobs = max(1, min(int(args.jobs), len(retrain_dates)))
    entries_by_cutoff: dict[str, object] = {}
    failed: list[tuple[str, str]] = []

    def _run(cutoff: pd.Timestamp):
        ok, artifact_path, calibrator_path, err = train_one_cutoff(
            cutoff, STRATEGY_DIR,
            label=args.label,
            watchlist_file=args.watchlist_file,
            artifact_root=args.artifact_root,
            fingerprint_config=args.fingerprint_config,
            fit_calibrator=not args.skip_calibrators,
            calibrator_method=args.calibrator_method,
        )
        if not ok:
            return cutoff, None, err
        try:
            trained_dt = read_trained_date(artifact_path)
        except Exception as exc:  # noqa: BLE001
            return cutoff, None, f"read_trained_date: {exc}"
        effective_cutoff = read_effective_train_cutoff_date(artifact_path)
        return cutoff, build_retrain_entry(
            cutoff=cutoff,
            trained_dt=trained_dt,
            artifact_uri=str(artifact_path),
            lookahead_days=infer_label_lookahead_days(args.label),
            calibrator_uri=str(calibrator_path) if calibrator_path else None,
            effective_train_cutoff_date=effective_cutoff,
        ), ""

    if jobs == 1:
        for i, cutoff in enumerate(retrain_dates):
            log.info("── retrain %d/%d  cutoff=%s ──",
                     i + 1, len(retrain_dates), cutoff.date().isoformat())
            cutoff, entry, err = _run(cutoff)
            cutoff_iso = cutoff.date().isoformat()
            if entry is None:
                failed.append((cutoff_iso, err))
            else:
                entries_by_cutoff[cutoff_iso] = entry
    else:
        log.info("Running %d cutoff retrains with jobs=%d", len(retrain_dates), jobs)
        with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="wf-train") as pool:
            futures = {pool.submit(_run, cutoff): cutoff for cutoff in retrain_dates}
            for fut in as_completed(futures):
                cutoff = futures[fut]
                cutoff_iso = cutoff.date().isoformat()
                try:
                    _, entry, err = fut.result()
                except Exception as exc:  # noqa: BLE001
                    failed.append((cutoff_iso, repr(exc)))
                    log.exception("cutoff=%s crashed", cutoff_iso)
                    continue
                if entry is None:
                    failed.append((cutoff_iso, err))
                else:
                    entries_by_cutoff[cutoff_iso] = entry
                    log.info("cutoff=%s collected (%d/%d)",
                             cutoff_iso, len(entries_by_cutoff), len(retrain_dates))

    entries = [
        entries_by_cutoff[d.date().isoformat()]
        for d in retrain_dates
        if d.date().isoformat() in entries_by_cutoff
    ]
    if failed and not bool(getattr(args, "allow_partial_manifest", False)):
        preview = ", ".join(f"{cutoff}: {err}" for cutoff, err in failed[:5])
        raise RuntimeError(
            "train_walkforward_panel: refusing to write partial manifest "
            f"({len(entries)}/{len(retrain_dates)} retrains succeeded, "
            f"{len(failed)} failed). First failures: {preview}. "
            "Re-run failed cutoffs or pass --allow-partial-manifest for an "
            "explicit research-only diagnostic."
        )
    return entries, failed


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    retrain_dates = compute_retrain_dates(start, end, args.cadence_days)
    log.info("Walk-forward v2 plan: start=%s end=%s cadence=%dd → %d retrains",
             start.date(), end.date(), args.cadence_days, len(retrain_dates))

    if args.dry_run:
        for i, d in enumerate(retrain_dates):
            print(f"[{i+1:02d}/{len(retrain_dates)}] cutoff={d.date().isoformat()}")
        print(f"Total retrain dates: {len(retrain_dates)}")
        print(f"Artifact root: {STRATEGY_DIR / 'artifacts' / WF_V2_SUBDIR}/")
        print(f"Manifest output: {args.manifest_output}")
        return

    # AC7 fail-closed gate: never dispatch cutoffs against a panel that would
    # silently truncate the training window (or is too thin/gappy to train).
    assert_training_panel_fresh(args, retrain_dates)

    # Lazy imports — only when actually training.
    from renquant_backtesting.walk_forward.manifest import WalkForwardManifest, write_manifest  # noqa: PLC0415

    entries, failed = train_cutoffs(retrain_dates, args)

    manifest = WalkForwardManifest(
        cadence_days=int(args.cadence_days),
        training_window_years=0.0,  # v2 uses cutoff-only slicing, no window
        retrains=entries,
    )
    out = write_manifest(manifest, args.manifest_output)
    log.info("Wrote manifest with %d/%d retrains → %s",
             len(entries), len(retrain_dates), out)
    if failed:
        log.warning("FAILED cutoffs (%d): %s", len(failed), failed)


if __name__ == "__main__":
    main()
