#!/usr/bin/env python
"""Walk-forward HF PatchTST training driver for renquant_104.

This is the sequence-model companion to ``train_walkforward_panel.py``. It
does not train inside this file. Each cutoff is a subprocess call to the
renquant-model HF Trainer module (``renquant_model_patchtst.hf_trainer``),
followed by a causal per-fold calibrator fit from the same model repo
(``renquant_model_patchtst.fit_calibrator``) and a standard
``renquant_backtesting.walk_forward`` manifest entry.

Repo boundary: model-training internals live in **renquant-model**; this
orchestrator only sequences per-cutoff subprocesses and assembles the
manifest. It therefore invokes the model repo *as a module* (``python -m
renquant_model_patchtst.hf_trainer``) rather than shelling out to a
co-located ``scripts/patchtst_hf.py`` — that script only ever lived in the
umbrella working tree and is not part of the renquant-backtesting checkout,
so the old script-path invocation raised ``No such file or directory`` for
every fold when run from a clean/pinned checkout.

Data & artifact root: ``data/*.parquet`` and ``backtesting/<strategy>/``
live in the umbrella RenQuant checkout, not next to this package. The root
is resolved explicitly via ``renquant_backtesting.repo_root.resolve_repo_root``
(``--repo-root`` / ``$RENQUANT_REPO_ROOT`` / cwd), matching the other
package CLIs (e.g. ``wf_gate.check_active_scorer``).

Usage::

    # Dry-run: print the retrain dates without training
    python -m renquant_backtesting.wf_gate.train_walkforward_patchtst \\
        --start-date 2024-01-01 --end-date 2024-03-01 \\
        --cadence-days 21 --repo-root /path/to/RenQuant --dry-run

    # Real walk-forward training (run from / point --repo-root at the
    # umbrella checkout that holds data/ and backtesting/renquant_104/)
    python -m renquant_backtesting.wf_gate.train_walkforward_patchtst \\
        --start-date 2024-01-01 --end-date 2026-03-26 --cadence-days 21 \\
        --device cpu --epochs 5
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
from pathlib import Path

# Make the renquant_backtesting package importable when this driver is run as
# a bare script path (``python .../train_walkforward_patchtst.py``) as well as
# via ``python -m``. ``parents[2]`` is ``<checkout>/src``.
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd  # noqa: E402

from renquant_backtesting.repo_root import (  # noqa: E402
    resolve_repo_root,
    strategy_dir,
)

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "6")

# ── Pinned subrepo assembly ─────────────────────────────────────────────────
# The per-cutoff training/calibration subprocess imports ``renquant_model_
# patchtst`` (+ deps) and this outer driver imports ``renquant_pipeline``
# transitively via ``renquant_backtesting.walk_forward``. Those MUST resolve
# from the SAME pinned subrepo assembly this driver was loaded from — never an
# arbitrary developer checkout — so a full WF run cannot silently derive
# artifacts from branches outside the pinned assembly. ``parents[3]`` is the
# renquant-backtesting checkout root; its parent is the assembly root that holds
# every ``<repo>/src`` (``.subrepo_runtime/repos`` in the pinned runtime).
_BACKTESTING_REPO_ROOT = Path(__file__).resolve().parents[3]

# Repos whose ``<repo>/src`` the subprocess (and this driver) require.
# renquant-model ships both renquant_model_patchtst and renquant_model_common.
REQUIRED_SUBREPOS = (
    "renquant-model",
    "renquant-common",
    "renquant-base-data",
    "renquant-artifacts",
    "renquant-pipeline",
)


def resolve_subrepo_root() -> Path:
    """Root of the pinned subrepo assembly (the dir holding ``<repo>/src`` for
    each pinned repo).

    Precedence: ``$RENQUANT_SUBREPO_ROOT`` (the standard injection point) →
    otherwise the assembly THIS driver was loaded from (the parent of the
    renquant-backtesting checkout, i.e. ``.subrepo_runtime/repos`` in the pinned
    runtime). There is deliberately NO ``~/git/github`` fallback and NO sibling
    globbing: the single root is either explicitly injected or the driver's own
    pinned assembly, so imports cannot leak in from ad-hoc dev checkouts.
    """
    env = os.environ.get("RENQUANT_SUBREPO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return _BACKTESTING_REPO_ROOT.parent


def required_subrepo_src_paths(root: Path | None = None) -> list[Path]:
    """``<repo>/src`` trees the subprocess must import from the pinned assembly.

    FAIL CLOSED: raise if the assembly is missing any required repo rather than
    let the import silently fall through to some other checkout — a full WF
    re-score must derive artifacts only from the pinned assembly.
    """
    root = root if root is not None else resolve_subrepo_root()
    srcs: list[Path] = []
    missing: list[Path] = []
    for repo in REQUIRED_SUBREPOS:
        src = root / repo / "src"
        (srcs if src.is_dir() else missing).append(src)
    if missing:
        raise RuntimeError(
            "train_walkforward_patchtst: pinned subrepo assembly at "
            f"{root} is missing required repo src trees "
            f"{[str(p) for p in missing]}. Point $RENQUANT_SUBREPO_ROOT at the "
            "assembly whose <repo>/src hold the pinned checkouts — refusing an "
            "ambiguous/partial assembly to keep WF artifacts pinned."
        )
    return srcs


# Put the pinned assembly's src trees on the OUTER interpreter path so a
# standalone ``python -m ...`` in the runtime can import renquant_pipeline (the
# manifest/loader). Single-root only (see resolve_subrepo_root); tolerant here
# so lightweight imports don't need the full assembly — the STRICT fail-closed
# check runs in ``subprocess_env`` right before any artifact is produced.
for _src_path in reversed([resolve_subrepo_root() / r / "src" for r in REQUIRED_SUBREPOS]):
    if _src_path.is_dir() and str(_src_path) not in sys.path:
        sys.path.insert(0, str(_src_path))

TRAIN_MODULE = "renquant_model_patchtst.hf_trainer"
CALIBRATOR_MODULE = "renquant_model_patchtst.fit_calibrator"
DEFAULT_ROOT = "walkforward_patchtst"
DEFAULT_STRATEGY = "renquant_104"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("train-walkforward-patchtst")


def infer_label_lookahead_days(label: str | None) -> int:
    import re
    m = re.search(r"fwd_(\d+)d", str(label or "fwd_60d_excess"))
    return int(m.group(1)) if m else 60


def data_end_for_cutoff(cutoff: pd.Timestamp, label: str | None) -> str:
    lookahead = infer_label_lookahead_days(label)
    return (cutoff - pd.offsets.BDay(lookahead)).date().isoformat()


def compute_retrain_dates(start: pd.Timestamp, end: pd.Timestamp,
                          cadence_days: int) -> list[pd.Timestamp]:
    if cadence_days <= 0:
        raise ValueError(f"cadence_days must be > 0, got {cadence_days}")
    return list(pd.date_range(start, end, freq=f"{cadence_days}D"))


# ── AC7 training-panel freshness/coverage gate (GOAL-5) ─────────────────────
# The per-fold trainer only rejects an EMPTY post-cutoff slice; a stale but
# nonempty parquet that stops short of a fold's ``data_end`` silently trains on
# a truncated window (each fold slices ``date < data_end``). Before dispatching
# ANY fold we assert the ONE canonical renquant-common freshness contract over
# the union window ``max(data_end)`` — fail-closed, never a silent proceed.
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
    """Absolute path to the training panel. ``--dataset`` is relative to the
    umbrella repo root (matching how the training subprocess, run with
    ``cwd=repo_root``, resolves it)."""
    ds = Path(args.dataset)
    if ds.is_absolute():
        return ds
    return resolve_repo_root(getattr(args, "repo_root", None)) / ds


def assert_training_panel_fresh(args: argparse.Namespace,
                                dates: list[pd.Timestamp]) -> None:
    """Fail-closed AC7 gate: verify the training panel COVERS the window every
    fold needs (+ density floors) BEFORE dispatching any fold. Raises with the
    contract's reasons rather than let a silently-truncated/thin panel train.
    """
    from renquant_common.training_freshness import (  # noqa: PLC0415
        assess_training_panel_freshness,
    )
    required = required_through_date(dates, args.label)
    dataset_path = resolve_dataset_path(args)
    if not dataset_path.exists():
        raise FileNotFoundError(
            "train_walkforward_patchtst: training panel not found at "
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
            "train_walkforward_patchtst: FAIL-CLOSED — training panel "
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


def strategy_dir_for(args: argparse.Namespace) -> Path:
    """Resolve the umbrella strategy dir (holds ``artifacts/`` and configs)."""
    repo_root = resolve_repo_root(getattr(args, "repo_root", None))
    return strategy_dir(repo_root, getattr(args, "strategy", DEFAULT_STRATEGY))


def default_manifest_output(args: argparse.Namespace) -> str:
    return str(strategy_dir_for(args) / "artifacts"
               / "walkforward_patchtst_manifest.json")


def artifact_dir(args: argparse.Namespace, cutoff: pd.Timestamp) -> Path:
    root = args.artifact_root or DEFAULT_ROOT
    return strategy_dir_for(args) / "artifacts" / root / cutoff.date().isoformat()


def model_path_for(out_dir: Path, seed: int) -> Path:
    return out_dir / f"hf_patchtst_all_seed{seed}_model.pt"


def calibrator_path_for(model_path: Path) -> Path:
    return model_path.with_name("hf_patchtst-calibration.json")


def sidecar_path_for(model_path: Path) -> Path:
    return model_path.with_name(model_path.name + ".metadata.json")


def train_cmd(args: argparse.Namespace, cutoff: pd.Timestamp,
              out_dir: Path) -> list[str]:
    cmd = [
        sys.executable, "-m", TRAIN_MODULE,
        "--dataset", args.dataset,
        "--cut", "all",
        "--train-cutoff", cutoff.date().isoformat(),
        "--label", args.label,
        "--epochs", str(args.epochs),
        "--seq-len", str(args.seq_len),
        "--patch-length", str(args.patch_length),
        "--d-model", str(args.d_model),
        "--n-heads", str(args.n_heads),
        "--n-layers", str(args.n_layers),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--device", args.device,
        "--seed", str(args.seed),
        "--save-model",
        "--output-dir", str(out_dir),
    ]
    if args.strategy_config:
        cmd.extend(["--strategy-config", args.strategy_config])
    if args.film_regime_cond:
        cmd.append("--film-regime-cond")
    if args.cross_stock_attn:
        cmd.append("--cross-stock-attn")
    return cmd


def calibrator_cmd(args: argparse.Namespace, cutoff: pd.Timestamp,
                   model_path: Path, cal_path: Path) -> list[str]:
    return [
        sys.executable, "-m", CALIBRATOR_MODULE,
        "--scorer-artifact", str(model_path),
        "--out", str(cal_path),
        "--panel", args.dataset,
        "--raw-label-panel", args.raw_label_panel,
        "--label-col", args.label,
        "--data-end", data_end_for_cutoff(cutoff, args.label),
        "--batch-size", str(args.calibrator_batch_size),
        "--method", args.calibrator_method,
        "--min-rows", str(args.calibrator_min_rows),
    ]


def read_contract(model_path: Path) -> dict:
    sidecar = sidecar_path_for(model_path)
    if not sidecar.exists():
        raise FileNotFoundError(f"missing PatchTST metadata sidecar: {sidecar}")
    payload = json.loads(sidecar.read_text())
    contract = payload.get("training_contract") or {}
    if not contract.get("trained_date") or not contract.get("effective_train_cutoff_date"):
        raise ValueError(f"incomplete PatchTST sidecar contract: {sidecar}")
    return contract


def build_entry(cutoff: pd.Timestamp, model_path: Path,
                cal_path: Path | None, label: str | None):
    from renquant_backtesting.walk_forward.loader import RetrainEntry  # noqa: PLC0415
    contract = read_contract(model_path)
    return RetrainEntry(
        cutoff_date=cutoff,
        trained_date=pd.Timestamp(contract["trained_date"]),
        artifact_uri=str(model_path),
        lookahead_days=infer_label_lookahead_days(label),
        calibrator_uri=str(cal_path) if cal_path else None,
        effective_train_cutoff_date=pd.Timestamp(
            contract["effective_train_cutoff_date"]
        ),
    )


def subprocess_env() -> dict[str, str]:
    """Env for the training/calibration subprocess: pin ``PYTHONPATH`` to the
    required ``<repo>/src`` trees of the resolved subrepo assembly so
    ``renquant_model_patchtst`` (and its deps) import ONLY from the pinned
    checkouts regardless of cwd. FAIL CLOSED via ``required_subrepo_src_paths``
    if the assembly is incomplete — an artifact must never be derived from an
    unpinned checkout."""
    env = os.environ.copy()
    src_paths = [str(p) for p in required_subrepo_src_paths()]
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        src_paths + ([existing] if existing else [])
    )
    return env


def run_subprocess(cmd: list[str], label: str, cwd: Path) -> tuple[bool, str]:
    log.info("%s start: %s", label, " ".join(cmd))
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd, cwd=str(cwd), check=False,
        capture_output=True, text=True, env=subprocess_env(),
    )
    elapsed = time.monotonic() - t0
    if proc.returncode != 0:
        return False, f"{label} exit={proc.returncode} stderr_tail={proc.stderr[-800:]!r}"
    log.info("%s done in %.1fs", label, elapsed)
    return True, ""


def train_one_cutoff(args: argparse.Namespace, cutoff: pd.Timestamp):
    cwd = resolve_repo_root(getattr(args, "repo_root", None))
    out_dir = artifact_dir(args, cutoff)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_path_for(out_dir, int(args.seed))
    if args.reuse_existing and model_path.exists() and sidecar_path_for(model_path).exists():
        log.info("train cutoff=%s reuse existing model: %s",
                 cutoff.date(), model_path)
    else:
        ok, err = run_subprocess(train_cmd(args, cutoff, out_dir),
                                 f"train cutoff={cutoff.date()}", cwd)
        if not ok:
            return cutoff, None, err
    cal_path = None
    if not args.skip_calibrators:
        cal_path = calibrator_path_for(model_path)
        if args.reuse_existing and cal_path.exists():
            log.info("calibrate cutoff=%s reuse existing calibrator: %s",
                     cutoff.date(), cal_path)
        else:
            ok, err = run_subprocess(calibrator_cmd(args, cutoff, model_path, cal_path),
                                     f"calibrate cutoff={cutoff.date()}", cwd)
            if not ok:
                return cutoff, None, err
    return cutoff, build_entry(cutoff, model_path, cal_path, args.label), ""


def train_cutoffs(args: argparse.Namespace, dates: list[pd.Timestamp]):
    jobs = max(1, min(int(args.jobs), len(dates)))
    entries_by_cutoff: dict[str, object] = {}
    failed: list[tuple[str, str]] = []

    if jobs == 1:
        for cutoff in dates:
            _, entry, err = train_one_cutoff(args, cutoff)
            key = cutoff.date().isoformat()
            if entry is None:
                failed.append((key, err))
            else:
                entries_by_cutoff[key] = entry
    else:
        with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="wf-pt") as pool:
            futures = {pool.submit(train_one_cutoff, args, d): d for d in dates}
            for fut in as_completed(futures):
                cutoff = futures[fut]
                key = cutoff.date().isoformat()
                try:
                    _, entry, err = fut.result()
                except Exception as exc:  # noqa: BLE001
                    failed.append((key, repr(exc)))
                    log.exception("cutoff=%s crashed", key)
                    continue
                if entry is None:
                    failed.append((key, err))
                else:
                    entries_by_cutoff[key] = entry

    entries = [
        entries_by_cutoff[d.date().isoformat()]
        for d in dates
        if d.date().isoformat() in entries_by_cutoff
    ]
    if failed and not args.allow_partial_manifest:
        preview = ", ".join(f"{c}: {e}" for c, e in failed[:5])
        raise RuntimeError(
            "train_walkforward_patchtst: refusing partial manifest "
            f"({len(entries)}/{len(dates)} succeeded). First failures: {preview}"
        )
    return entries, failed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--cadence-days", type=int, default=21)
    p.add_argument("--repo-root", default=None,
                   help="umbrella RenQuant root holding data/ and "
                        "backtesting/<strategy>/ (default: $RENQUANT_REPO_ROOT "
                        "or cwd)")
    p.add_argument("--strategy", default=DEFAULT_STRATEGY)
    p.add_argument("--manifest-output", default=None,
                   help="default: <repo-root>/backtesting/<strategy>/artifacts/"
                        "walkforward_patchtst_manifest.json")
    p.add_argument("--artifact-root", default=None)
    p.add_argument("--dataset", default="data/transformer_v4_wl200_clean.parquet")
    p.add_argument("--raw-label-panel",
                   default="data/alpha158_291_fundamental_dataset_rawlabel.parquet")
    p.add_argument("--label", default="fwd_60d_excess")
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--patch-length", type=int, default=4)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--strategy-config", default=None)
    p.add_argument("--film-regime-cond", action="store_true")
    p.add_argument("--cross-stock-attn", action="store_true")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--skip-calibrators", action="store_true")
    p.add_argument("--calibrator-batch-size", type=int, default=512)
    p.add_argument("--calibrator-method", default="platt",
                   choices=["platt", "isotonic"])
    p.add_argument("--calibrator-min-rows", type=int, default=1000)
    p.add_argument("--reuse-existing", action="store_true",
                   help="Reuse existing model sidecar/calibrator artifacts for "
                        "a cutoff instead of rerunning completed subprocesses.")
    p.add_argument("--allow-partial-manifest", action="store_true")
    # ── AC7 training-panel freshness/coverage gate (GOAL-5) ──────────────────
    p.add_argument("--min-tickers-per-day", type=int, default=20,
                   help="AC7 gate: min distinct tickers required on every "
                        "training-window day (0 disables). PerDayDataset "
                        "silently drops <5-ticker days.")
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
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    dates = compute_retrain_dates(
        pd.Timestamp(args.start_date), pd.Timestamp(args.end_date),
        int(args.cadence_days),
    )
    if args.manifest_output is None:
        args.manifest_output = default_manifest_output(args)
    if args.dry_run:
        for i, cutoff in enumerate(dates):
            out_dir = artifact_dir(args, cutoff)
            print(f"[{i + 1:02d}/{len(dates)}] cutoff={cutoff.date()} "
                  f"data_end={data_end_for_cutoff(cutoff, args.label)} "
                  f"model={model_path_for(out_dir, int(args.seed))}")
        print(f"Manifest output: {args.manifest_output}")
        return

    # AC7 fail-closed gate: never dispatch folds against a panel that would
    # silently truncate the training window (or is too thin/gappy to train).
    assert_training_panel_fresh(args, dates)

    from renquant_backtesting.walk_forward.manifest import WalkForwardManifest, write_manifest  # noqa: PLC0415
    entries, failed = train_cutoffs(args, dates)
    manifest = WalkForwardManifest(
        cadence_days=int(args.cadence_days),
        training_window_years=0.0,
        retrains=entries,
    )
    out = write_manifest(manifest, args.manifest_output)
    log.info("Wrote PatchTST manifest with %d/%d retrains -> %s",
             len(entries), len(dates), out)
    if failed:
        log.warning("FAILED cutoffs (%d): %s", len(failed), failed)


if __name__ == "__main__":
    main()
