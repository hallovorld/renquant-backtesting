#!/usr/bin/env python
"""Modal-scaled walk-forward PatchTST re-score — driver side (modal-free).

This module owns everything that does NOT need the Modal SDK imported at module
scope: the image spec (single source of truth), the recipe/provenance helpers,
the fold-request builder, data/code staging, dispatch orchestration, artifact
collection, manifest assembly, and the CLI.

``modal`` is imported **lazily** (inside :func:`stage_inputs_to_volume` and
:func:`dispatch_folds`) so that the recipe/provenance/manifest logic — and every
unit test — imports with no cloud dependency. The ``@app.function`` worker lives
in :mod:`renquant_backtesting.wf_gate.modal.app`, imported only at dispatch time
after the ``RENQUANT_WF_MODAL_*`` env vars are set (Modal bakes ``gpu`` /
``timeout`` / ``retries`` into the decorator at import time — mirroring the
orchestrator ``cloud/`` two-file split).

Repo boundary: model-training internals stay in **renquant-model**; this file
only sequences per-cutoff work and stamps provenance. The per-fold unit of work
is the existing, reviewed driver
``renquant_backtesting.wf_gate.train_walkforward_patchtst.train_one_cutoff``
(PR #74) — the Modal worker runs exactly that, one cutoff per GPU pod.

Usage::

    # Plan only — print the folds + recipe_id, no cloud calls
    python -m renquant_backtesting.wf_gate.modal.executor \\
        --start-date 2023-10-02 --end-date 2026-03-02 --cadence-days 21 --dry-run

    # Staged directional read — the 8 most-recent folds on a T4 GPU
    python -m renquant_backtesting.wf_gate.modal.executor \\
        --start-date 2023-10-02 --end-date 2026-03-02 --cadence-days 21 \\
        --staged 8 --gpu T4 --execute

    # Full 43-fold corpus
    python -m renquant_backtesting.wf_gate.modal.executor \\
        --start-date 2023-10-02 --end-date 2026-03-02 --cadence-days 21 \\
        --gpu A10G --execute
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Make the renquant_backtesting package importable when this module is run as a
# bare path as well as via ``python -m``. ``parents[3]`` is ``<checkout>/src``.
_SRC_DIR = Path(__file__).resolve().parents[3]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("wf-patchtst-modal")

# ── Constants ───────────────────────────────────────────────────────────────
APP_NAME = "renquant-wf-patchtst"
VOLUME_NAME = "renquant-wf-patchtst-data"
#: Bumped whenever the provenance sidecar layout changes.
PROVENANCE_SCHEMA_VERSION = "1.0"

DEFAULT_STRATEGY = "renquant_104"
DEFAULT_DATASET = "data/transformer_v4_wl200_clean.parquet"
DEFAULT_RAW_LABEL_PANEL = "data/alpha158_291_fundamental_dataset_rawlabel.parquet"
DEFAULT_ARTIFACT_ROOT = "walkforward_patchtst"

# Sibling repos whose ``src`` must be in the code bundle so the driver +
# training subprocess resolve inside the container. Order irrelevant; the
# driver's own sys.path logic + subprocess PYTHONPATH consume all of them.
BUNDLE_REPOS = [
    "renquant-backtesting",
    "renquant-model",
    "renquant-common",
    "renquant-base-data",
    "renquant-artifacts",
    "renquant-pipeline",
    "renquant-strategy-104",
]

# The container mounts the Volume at ``/data``; the code bundle lands at
# ``/data/app/repos/<repo>/src`` so the driver file's ``parents[3].parent``
# resolves to ``/data/app/repos`` and every sibling ``src`` is discovered.
CONTAINER_VOLUME_MOUNT = "/data"
CONTAINER_BUNDLE_ROOT = "/data/app/repos"
CONTAINER_REPO_ROOT = "/data"  # holds data/ and backtesting/<strategy>/

# ── Image spec (single source of truth; app.py re-declares the literals) ─────
# The GPU image carries the PatchTST training stack (torch cuda build from PyPI
# + HF transformers/accelerate) plus the shared pipeline deps the driver imports
# transitively. Kept as a plain dict here (no ``modal`` import) so a test can
# assert app.py's decoration-time image inputs match byte-for-byte.
IMAGE_SPEC: dict[str, Any] = {
    "base": "debian_slim",
    "python_version": "3.10",
    "pip_packages": [
        # PatchTST training stack (PyPI torch is the CUDA build on linux).
        "torch>=2.2",
        "transformers>=4.40",
        "accelerate>=0.26",
        # Shared pipeline / common deps the driver imports transitively —
        # superset of the orchestrator cloud image (proven to import the full
        # kernel in-container) so the fail-closed assembly resolves cleanly.
        "pandas>=2.0",
        "numpy>=1.26",
        "scipy>=1.11",
        "scikit-learn>=1.4",
        "pyarrow>=15.0",
        "joblib>=1.2",
        "pyyaml>=6.0",
        "pandas-market-calendars>=4.0",
        "hmmlearn>=0.3",
        "cvxpy>=1.3",
        "pydantic>=2.0",
    ],
    "run_commands": [],
}


def image_spec_fingerprint() -> str:
    """Stable sha256 of the image spec — recorded in every provenance sidecar."""
    payload = json.dumps(IMAGE_SPEC, sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── Recipe identity ──────────────────────────────────────────────────────────
# Mirror the established WF-gate convention (wf_gate.recipe_match.recipe_fingerprint):
# a ``"sha256:<16 hex>"`` string. Here the recipe is the walk-forward TRAINING
# request (dataset + label + cadence + the PatchTST hyperparameters) — the
# identity every fold in this corpus shares. Distinct from the per-artifact
# recipe fingerprint the gate recomputes downstream; this is the run-level recipe.
RECIPE_FIELDS = (
    "dataset", "label", "cadence_days", "seed", "epochs", "seq_len",
    "patch_length", "d_model", "n_heads", "n_layers", "lr", "weight_decay",
    "film_regime_cond", "cross_stock_attn", "calibrator_method",
)


def compute_recipe_id(recipe: dict[str, Any]) -> str:
    """Stable recipe id for the WF training request (``sha256:<16hex>``)."""
    projection = {k: recipe.get(k) for k in RECIPE_FIELDS}
    payload = json.dumps(projection, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ── Fold selection ───────────────────────────────────────────────────────────
def compute_retrain_cutoffs(start: str, end: str, cadence_days: int) -> list[str]:
    """Isoformat cutoff dates — delegated to the reviewed #74 driver helper."""
    from renquant_backtesting.wf_gate.train_walkforward_patchtst import (  # noqa: PLC0415
        compute_retrain_dates,
    )
    import pandas as pd  # noqa: PLC0415

    dates = compute_retrain_dates(
        pd.Timestamp(start), pd.Timestamp(end), int(cadence_days)
    )
    return [d.date().isoformat() for d in dates]


def select_staged_cutoffs(cutoffs: list[str], staged: int | None) -> list[str]:
    """Return the ``staged`` most-recent cutoffs (directional read) or all."""
    if not staged or staged <= 0 or staged >= len(cutoffs):
        return list(cutoffs)
    return list(cutoffs[-staged:])


# ── Requests + recipe ────────────────────────────────────────────────────────
@dataclass
class WfRescorePlan:
    """Everything needed to dispatch + stamp a WF PatchTST re-score run."""

    cutoffs: list[str]
    recipe: dict[str, Any]
    recipe_id: str
    gpu: str
    dataset: str = DEFAULT_DATASET
    raw_label_panel: str = DEFAULT_RAW_LABEL_PANEL
    strategy: str = DEFAULT_STRATEGY
    artifact_root: str = DEFAULT_ARTIFACT_ROOT
    skip_calibrators: bool = False
    manifest_output: str | None = None
    fold_requests: list[dict[str, Any]] = field(default_factory=list)


def build_recipe(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset": args.dataset,
        "raw_label_panel": args.raw_label_panel,
        "label": args.label,
        "cadence_days": int(args.cadence_days),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "seq_len": int(args.seq_len),
        "patch_length": int(args.patch_length),
        "d_model": int(args.d_model),
        "n_heads": int(args.n_heads),
        "n_layers": int(args.n_layers),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "film_regime_cond": bool(args.film_regime_cond),
        "cross_stock_attn": bool(args.cross_stock_attn),
        "calibrator_method": args.calibrator_method,
        "calibrator_min_rows": int(args.calibrator_min_rows),
        "calibrator_batch_size": int(args.calibrator_batch_size),
        "device": args.device,
        "skip_calibrators": bool(args.skip_calibrators),
        "strategy": args.strategy,
        "artifact_root": args.artifact_root or DEFAULT_ARTIFACT_ROOT,
    }


def build_fold_request(cutoff: str, recipe: dict[str, Any], recipe_id: str,
                       image_sha: str) -> dict[str, Any]:
    """One JSON-able request per fold (one Modal pod trains this cutoff)."""
    return {
        "cutoff_date": cutoff,
        "recipe": recipe,
        "recipe_id": recipe_id,
        "image_spec_sha256": image_sha,
        "container_repo_root": CONTAINER_REPO_ROOT,
        "container_bundle_root": CONTAINER_BUNDLE_ROOT,
    }


def build_plan(args: argparse.Namespace) -> WfRescorePlan:
    recipe = build_recipe(args)
    recipe_id = compute_recipe_id(recipe)
    all_cutoffs = compute_retrain_cutoffs(
        args.start_date, args.end_date, int(args.cadence_days)
    )
    cutoffs = select_staged_cutoffs(all_cutoffs, getattr(args, "staged", None))
    image_sha = image_spec_fingerprint()
    plan = WfRescorePlan(
        cutoffs=cutoffs,
        recipe=recipe,
        recipe_id=recipe_id,
        gpu=args.gpu,
        dataset=args.dataset,
        raw_label_panel=args.raw_label_panel,
        strategy=args.strategy,
        artifact_root=args.artifact_root or DEFAULT_ARTIFACT_ROOT,
        skip_calibrators=bool(args.skip_calibrators),
        manifest_output=args.manifest_output,
    )
    plan.fold_requests = [
        build_fold_request(c, recipe, recipe_id, image_sha) for c in cutoffs
    ]
    return plan


# ── Auth precheck ────────────────────────────────────────────────────────────
def modal_readiness() -> dict[str, Any]:
    """Report exactly what (if anything) blocks a real Modal run.

    Never raises — the caller decides whether to fail-closed. ``ready`` is True
    only when the SDK imports AND a token/profile is discoverable.
    """
    report: dict[str, Any] = {"sdk_importable": False, "token_present": False,
                              "missing": [], "ready": False}
    try:
        import modal  # noqa: F401,PLC0415
        report["sdk_importable"] = True
        report["modal_version"] = getattr(modal, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        report["missing"].append(f"modal SDK import failed ({exc!r}); "
                                 "`pip install modal`")
    token_file = Path.home() / ".modal.toml"
    env_token = bool(os.environ.get("MODAL_TOKEN_ID")
                     and os.environ.get("MODAL_TOKEN_SECRET"))
    if token_file.exists() or env_token:
        report["token_present"] = True
    else:
        report["missing"].append(
            "no Modal credentials: neither ~/.modal.toml nor "
            "MODAL_TOKEN_ID/MODAL_TOKEN_SECRET env vars set "
            "(run `modal token new`)"
        )
    report["ready"] = report["sdk_importable"] and report["token_present"]
    return report


# ── Code bundle + Volume staging ─────────────────────────────────────────────
def _sibling_src(repo: str, code_roots: list[Path]) -> Path | None:
    for root in code_roots:
        p = root / repo / "src"
        if p.exists():
            return p
    return None


def bundle_code(bundle_dir: Path, code_roots: list[Path]) -> dict[str, str]:
    """Copy each sibling repo's ``src`` into ``bundle_dir/<repo>/src``.

    Returns ``{repo: git_head}`` for provenance. Missing repos are skipped and
    reported to the caller (a missing bundle repo is a fatal staging error).
    """
    import subprocess  # noqa: PLC0415

    heads: dict[str, str] = {}
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for repo in BUNDLE_REPOS:
        src = _sibling_src(repo, code_roots)
        if src is None:
            continue
        dst = bundle_dir / repo / "src"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".pytest_cache", "*.egg-info"))
        try:
            head = subprocess.run(
                ["git", "-C", str(src.parent), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=False,
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            head = "unknown"
        heads[repo] = head or "unknown"
    return heads


def stage_inputs_to_volume(plan: WfRescorePlan, *, bundle_dir: Path,
                           dataset_path: Path, raw_label_path: Path,
                           volume_name: str = VOLUME_NAME) -> dict[str, Any]:
    """Batch-upload the code bundle + the two parquet panels to the Volume.

    Layout on the Volume (mounted at ``/data`` in the container):
      * ``/app/repos/<repo>/src``  — code bundle
      * ``/data/<dataset>.parquet`` — training panel (kept under ``data/`` so the
        driver's ``--dataset data/...`` resolves against ``--repo-root /data``)
      * ``/data/<rawlabel>.parquet`` — calibrator raw-label panel

    ``modal`` is imported here (lazily). Returns a content-addressed commit id.
    """
    import modal  # noqa: PLC0415

    vol = modal.Volume.from_name(volume_name, create_if_missing=True)
    uploaded: list[tuple[str, str]] = []  # (local, remote)
    for repo_src in sorted(bundle_dir.rglob("*")):
        if repo_src.is_file():
            rel = repo_src.relative_to(bundle_dir)
            uploaded.append((str(repo_src), f"/app/repos/{rel.as_posix()}"))
    uploaded.append((str(dataset_path), f"/data/{Path(plan.dataset).name}"))
    uploaded.append((str(raw_label_path),
                     f"/data/{Path(plan.raw_label_panel).name}"))

    hasher = hashlib.sha256()
    with vol.batch_upload(force=True) as batch:
        for local, remote in uploaded:
            batch.put_file(local, remote)
            hasher.update(remote.encode())
            hasher.update(str(Path(local).stat().st_size).encode())
    commit_id = "sha256:" + hasher.hexdigest()[:16]
    log.info("staged %d files to Volume %s (commit=%s)",
             len(uploaded), volume_name, commit_id)
    return {"volume_name": volume_name, "volume_commit_id": commit_id,
            "n_files": len(uploaded)}


# ── Dispatch ─────────────────────────────────────────────────────────────────
def _import_app_with_env(gpu: str, timeout_s: int, retries: int):
    """Set the ``RENQUANT_WF_MODAL_*`` env vars then import the app module.

    Modal bakes gpu/timeout/retries into ``@app.function`` at import time, so
    they must be in the environment BEFORE the app module is first imported
    (identical constraint + guard to the orchestrator cloud executor).
    """
    module_name = "renquant_backtesting.wf_gate.modal.app"
    desired = (str(gpu), int(timeout_s), int(retries))
    if module_name in sys.modules:
        existing = sys.modules[module_name]
        current = (str(getattr(existing, "WORKER_GPU", None)),
                   int(getattr(existing, "WORKER_TIMEOUT_SECONDS", -1)),
                   int(getattr(existing, "WORKER_RETRIES", -1)))
        if current != desired:
            raise RuntimeError(
                "modal.app already imported with gpu/timeout/retries="
                f"{current}; requested {desired} needs a fresh process."
            )
    else:
        os.environ["RENQUANT_WF_MODAL_GPU"] = str(gpu)
        os.environ["RENQUANT_WF_MODAL_TIMEOUT_SECONDS"] = str(int(timeout_s))
        os.environ["RENQUANT_WF_MODAL_RETRIES"] = str(int(retries))
    import importlib  # noqa: PLC0415
    return importlib.import_module(module_name)


def dispatch_folds(plan: WfRescorePlan, *, timeout_s: int, retries: int,
                   volume_commit_id: str | None) -> list[dict[str, Any]]:
    """Fan out one pod per fold via ``train_fold_remote.map`` and collect JSON.

    Mirrors the orchestrator ``execute_batch`` dispatch: ``with app.run():`` +
    ``.map(order_outputs=False, return_exceptions=True)`` so a single fold's
    failure is reported, not fatal to the batch.
    """
    mod = _import_app_with_env(plan.gpu, timeout_s, retries)
    payloads = []
    for req in plan.fold_requests:
        r = dict(req)
        r["volume_commit_id"] = volume_commit_id
        payloads.append(json.dumps(r))

    results: list[dict[str, Any]] = []
    with mod.app.run():
        # wrap_returned_exceptions=False → a failed pod yields its underlying
        # exception directly (opt into the post-2025-06-27 Modal behavior;
        # otherwise it leaks a modal.exceptions.UserCodeException wrapper).
        for item in mod.train_fold_remote.map(
            payloads, order_outputs=False, return_exceptions=True,
            wrap_returned_exceptions=False,
        ):
            if isinstance(item, Exception):
                results.append({"ok": False, "cutoff_date": None,
                                "error": repr(item)})
                continue
            results.append(json.loads(item))
    return results


# ── Artifact collection + manifest + provenance ──────────────────────────────
def _write_bytes_b64gz(b64gz: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(gzip.decompress(base64.b64decode(b64gz)))


def collect_fold_artifacts(result: dict[str, Any], strategy_artifacts: Path,
                           artifact_root: str) -> dict[str, Any]:
    """Materialise one pod's returned artifacts under the local strategy tree.

    Returns the manifest-entry dict (already carrying effective_train_cutoff_date).
    """
    cutoff = result["cutoff_date"]
    out_dir = strategy_artifacts / artifact_root / cutoff
    arts = result.get("artifacts") or {}
    model_rel = result["entry"]["artifact_uri"]
    # ``artifact_uri`` on the pod is an absolute container path; re-root it under
    # the local strategy artifacts tree by filename so we stay independent of the
    # container's paths.
    model_path = out_dir / Path(model_rel).name
    _write_bytes_b64gz(arts["model_pt_b64gz"], model_path)
    if arts.get("sidecar_json"):
        (out_dir / (model_path.name + ".metadata.json")).write_text(
            arts["sidecar_json"])
    entry = dict(result["entry"])
    entry["artifact_uri"] = str(model_path)
    if arts.get("calibrator_json"):
        cal_path = model_path.with_name("hf_patchtst-calibration.json")
        cal_path.write_text(arts["calibrator_json"])
        entry["calibrator_uri"] = str(cal_path)
    return entry


def assemble_manifest(entries: list[dict[str, Any]], cadence_days: int,
                      manifest_output: Path) -> Path:
    """Write the standard WF manifest via the reviewed writer (validates leakage)."""
    import pandas as pd  # noqa: PLC0415
    from renquant_backtesting.walk_forward.loader import RetrainEntry  # noqa: PLC0415
    from renquant_backtesting.walk_forward.manifest import (  # noqa: PLC0415
        WalkForwardManifest, write_manifest,
    )

    retrains = []
    for e in entries:
        eff = e.get("effective_train_cutoff_date")
        retrains.append(RetrainEntry(
            cutoff_date=pd.Timestamp(e["cutoff_date"]),
            trained_date=pd.Timestamp(e["trained_date"]),
            artifact_uri=str(e["artifact_uri"]),
            lookahead_days=int(e.get("lookahead_days", 60)),
            calibrator_uri=(str(e["calibrator_uri"])
                            if e.get("calibrator_uri") else None),
            effective_train_cutoff_date=(pd.Timestamp(eff) if eff else None),
        ))
    manifest = WalkForwardManifest(
        cadence_days=int(cadence_days), training_window_years=0.0,
        retrains=retrains,
    )
    return write_manifest(manifest, manifest_output)


def build_provenance(plan: WfRescorePlan, results: list[dict[str, Any]],
                     entries: list[dict[str, Any]], *, code_heads: dict[str, str],
                     staging: dict[str, Any], manifest_path: str) -> dict[str, Any]:
    """The FRESH-corpus provenance sidecar (GOAL-2 AC2/AC3 stamps)."""
    fold_prov = []
    for e in entries:
        fold_prov.append({
            "cutoff_date": e["cutoff_date"],
            "trained_date": e["trained_date"],
            "effective_train_cutoff_date": e.get("effective_train_cutoff_date"),
            "artifact_uri": e["artifact_uri"],
            "calibrator_uri": e.get("calibrator_uri"),
        })
    # Per-pod Modal facts, keyed by cutoff.
    pod_facts = {r.get("cutoff_date"): {
        "worker_id": r.get("worker_id"),
        "code_image_id": r.get("code_image_id"),
        "elapsed_seconds": r.get("elapsed_seconds"),
        "device": r.get("device"),
        "result_checksum": r.get("result_checksum"),
    } for r in results if r.get("ok")}
    failed = [{"cutoff_date": r.get("cutoff_date"), "error": r.get("error")}
              for r in results if not r.get("ok")]
    return {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "recipe_id": plan.recipe_id,
        "recipe": plan.recipe,
        "built_by": "renquant_backtesting.wf_gate.modal.executor",
        "expert_role": "patchtst_fresh_2nd_expert",
        "goal": "GOAL-2 AC2/AC3 (fresh PatchTST 2nd expert for GOAL-4 ensemble)",
        "manifest": manifest_path,
        "n_folds_requested": len(plan.cutoffs),
        "n_folds_succeeded": len(entries),
        "modal": {
            "app_name": APP_NAME,
            "gpu": plan.gpu,
            "image_spec_sha256": image_spec_fingerprint(),
            "volume_name": staging.get("volume_name"),
            "volume_commit_id": staging.get("volume_commit_id"),
            "code_git_heads": code_heads,
        },
        "folds": fold_prov,
        "pod_facts": pod_facts,
        "failed_folds": failed,
    }


def collect_and_write(plan: WfRescorePlan, results: list[dict[str, Any]], *,
                      repo_root: Path, code_heads: dict[str, str],
                      staging: dict[str, Any]) -> dict[str, Any]:
    """Materialise artifacts, write the manifest + provenance sidecar locally."""
    strategy_artifacts = repo_root / "backtesting" / plan.strategy / "artifacts"
    entries = []
    for r in results:
        if not r.get("ok"):
            continue
        entries.append(collect_fold_artifacts(r, strategy_artifacts,
                                               plan.artifact_root))
    manifest_output = Path(plan.manifest_output) if plan.manifest_output else (
        strategy_artifacts / "walkforward_patchtst_manifest.json")
    manifest_path = ""
    if entries:
        manifest_path = str(assemble_manifest(
            entries, plan.recipe["cadence_days"], manifest_output))
    provenance = build_provenance(
        plan, results, entries, code_heads=code_heads, staging=staging,
        manifest_path=manifest_path)
    prov_path = Path(str(manifest_output) + ".provenance.json")
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    prov_path.write_text(json.dumps(provenance, indent=2, sort_keys=True))
    log.info("wrote %d/%d folds; manifest=%s provenance=%s",
             len(entries), len(plan.cutoffs), manifest_path, prov_path)
    return {"manifest": manifest_path, "provenance": str(prov_path),
            "n_folds": len(entries), "provenance_obj": provenance}


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start-date", default="2023-10-02")
    p.add_argument("--end-date", default="2026-03-02")
    p.add_argument("--cadence-days", type=int, default=21)
    p.add_argument("--staged", type=int, default=None,
                   help="Run only the N most-recent folds (directional read).")
    p.add_argument("--gpu", default="T4",
                   help="Modal GPU type (T4|A10G|L4|A100|...). Use 'cpu' to "
                        "run CPU-only pods (slower, cheaper).")
    p.add_argument("--repo-root", default=None,
                   help="umbrella RenQuant root holding data/ and "
                        "backtesting/<strategy>/ (default: $RENQUANT_REPO_ROOT "
                        "or cwd)")
    p.add_argument("--strategy", default=DEFAULT_STRATEGY)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--raw-label-panel", default=DEFAULT_RAW_LABEL_PANEL)
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
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"],
                   help="Device passed to the trainer INSIDE the pod.")
    p.add_argument("--film-regime-cond", action="store_true")
    p.add_argument("--cross-stock-attn", action="store_true")
    p.add_argument("--skip-calibrators", action="store_true",
                   help="Skip the fit_calibrator leg (NOT recommended: the "
                        "fresh corpus needs calibrators to be usable).")
    p.add_argument("--calibrator-method", default="platt",
                   choices=["platt", "isotonic"])
    p.add_argument("--calibrator-min-rows", type=int, default=1000)
    p.add_argument("--calibrator-batch-size", type=int, default=512)
    p.add_argument("--artifact-root", default=None)
    p.add_argument("--manifest-output", default=None)
    p.add_argument("--timeout-seconds", type=int, default=3600)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--dry-run", action="store_true",
                   help="Plan only: print folds + recipe_id, make no cloud calls.")
    p.add_argument("--execute", action="store_true",
                   help="Actually dispatch to Modal (default is plan-only).")
    return p.parse_args(argv)


def resolve_repo_root(value: str | None) -> Path:
    from renquant_backtesting.repo_root import resolve_repo_root as _rrr  # noqa: PLC0415
    return _rrr(value)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)

    print(f"WF PatchTST Modal re-score plan")
    print(f"  recipe_id     : {plan.recipe_id}")
    print(f"  image_spec    : {image_spec_fingerprint()}")
    print(f"  gpu           : {plan.gpu}")
    print(f"  folds         : {len(plan.cutoffs)} "
          f"({plan.cutoffs[0]} .. {plan.cutoffs[-1]})" if plan.cutoffs else "  folds: 0")
    print(f"  calibrators   : {'SKIPPED' if plan.skip_calibrators else 'RUN'}")

    if args.dry_run or not args.execute:
        for i, c in enumerate(plan.cutoffs):
            print(f"    [{i + 1:02d}/{len(plan.cutoffs)}] cutoff={c}")
        if not args.execute:
            print("\n(plan-only; pass --execute to dispatch to Modal)")
        return 0

    readiness = modal_readiness()
    if not readiness["ready"]:
        print("\nMODAL NOT READY — cannot dispatch. Missing:")
        for m in readiness["missing"]:
            print(f"  - {m}")
        return 2

    repo_root = resolve_repo_root(args.repo_root)
    dataset_path = repo_root / plan.dataset
    raw_label_path = repo_root / plan.raw_label_panel
    for pth in (dataset_path, raw_label_path):
        if not pth.exists():
            print(f"\nMissing required input panel: {pth}")
            return 2

    import tempfile  # noqa: PLC0415
    code_roots = [repo_root.parent, Path.home() / "git" / "github"]
    with tempfile.TemporaryDirectory(prefix="wf-pt-bundle-") as td:
        bundle_dir = Path(td)
        code_heads = bundle_code(bundle_dir, code_roots)
        staging = stage_inputs_to_volume(
            plan, bundle_dir=bundle_dir, dataset_path=dataset_path,
            raw_label_path=raw_label_path)
        results = dispatch_folds(
            plan, timeout_s=args.timeout_seconds, retries=args.retries,
            volume_commit_id=staging.get("volume_commit_id"))
    out = collect_and_write(
        plan, results, repo_root=repo_root, code_heads=code_heads,
        staging=staging)
    print(f"\nDONE: {out['n_folds']}/{len(plan.cutoffs)} folds")
    print(f"  manifest   : {out['manifest']}")
    print(f"  provenance : {out['provenance']}")
    return 0 if out["n_folds"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
