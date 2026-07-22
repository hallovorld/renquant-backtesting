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

# ``parents[4]`` is the renquant-backtesting checkout THIS executor runs from;
# its parent is the code-assembly root that holds every ``<repo>/src``. Bundling
# from here (NOT from an arbitrary ``repo_root.parent``) is what keeps the staged
# code identical to the reviewed checkout — the same anti-contamination invariant
# the #74 driver enforces for its own subprocess (``resolve_subrepo_root``).
_EXECUTOR_CHECKOUT_ROOT = Path(__file__).resolve().parents[4]

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

# Non-``src`` subdirs a repo must ALSO contribute to the bundle because the
# trainer reads them at runtime. ``renquant-strategy-104/configs`` holds
# ``strategy_config.json``, which ``hf_trainer.build_config_contract()`` loads
# from ``<assembly>/renquant-strategy-104/configs/`` at the END of a fit — a
# bundle without it wastes a full training run then dies with FileNotFoundError.
EXTRA_BUNDLE_SUBDIRS: dict[str, tuple[str, ...]] = {
    "renquant-strategy-104": ("configs",),
}

# ── Run namespacing (quarantine; codex #76 blocker 3) ────────────────────────
# The executor NEVER writes the canonical serving manifest. Every run lands under
# an isolated, run-id'd namespace so a partial/unverified corpus cannot be picked
# up as a serving artifact; promotion to the canonical name is a separate,
# reviewed step that must validate every requested fold first.
RUN_NAMESPACE_ROOT = "walkforward_patchtst_runs"
#: The canonical serving manifest the WF gate consumes — this executor refuses to
#: write it (guarded in ``collect_and_write``).
CANONICAL_SERVING_MANIFEST = "walkforward_patchtst_manifest.json"

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
    run_id: str
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


def _default_run_id(recipe_id: str) -> str:
    """Isolated run namespace: ``wf-pt-<recipe8>-<utcstamp>`` (never canonical)."""
    from datetime import datetime, timezone  # noqa: PLC0415
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = recipe_id.split(":")[-1][:8]
    return f"wf-pt-{short}-{stamp}"


def build_plan(args: argparse.Namespace) -> WfRescorePlan:
    recipe = build_recipe(args)
    recipe_id = compute_recipe_id(recipe)
    run_id = getattr(args, "run_id", None) or _default_run_id(recipe_id)
    all_cutoffs = compute_retrain_cutoffs(
        args.start_date, args.end_date, int(args.cadence_days)
    )
    cutoffs = select_staged_cutoffs(all_cutoffs, getattr(args, "staged", None))
    image_sha = image_spec_fingerprint()
    # Quarantine: artifacts + manifest always land under a run-id'd namespace,
    # never the canonical serving tree (codex #76 blocker 3).
    plan = WfRescorePlan(
        cutoffs=cutoffs,
        recipe=recipe,
        recipe_id=recipe_id,
        gpu=args.gpu,
        run_id=run_id,
        dataset=args.dataset,
        raw_label_panel=args.raw_label_panel,
        strategy=args.strategy,
        artifact_root=f"{RUN_NAMESPACE_ROOT}/{run_id}",
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
def bundle_code(bundle_dir: Path, code_root: Path, *,
                assembly_lock: dict[str, str] | None = None) -> dict[str, str]:
    """Copy each REQUIRED repo's ``src`` from ONE pinned assembly into the bundle.

    ``code_root`` is a single, explicit pinned-assembly root holding
    ``<repo>/src`` for every :data:`BUNDLE_REPOS`. There is deliberately NO
    ``~/git/github`` fallback and NO per-repo root search — the same
    single-pinned-assembly invariant the #74 driver's ``resolve_subrepo_root``
    enforces, so a WF corpus cannot be silently sourced from an arbitrary/ambient
    checkout (codex #76 blocker 1).

    FAIL CLOSED:
      * any required repo missing under ``code_root`` → refuse;
      * any staged repo with no resolvable git HEAD (unpinned checkout) → refuse
        (every fold's provenance must name the exact commit it was built from);
      * ``assembly_lock`` given and any staged HEAD drifts from it → refuse
        (verify every staged commit against the reviewed candidate lock before
        dispatch).

    Returns ``{repo: git_head}`` for provenance.
    """
    import subprocess  # noqa: PLC0415

    _ignore = shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".pytest_cache", "*.egg-info")
    heads: dict[str, str] = {}
    missing: list[str] = []
    unpinned: list[str] = []
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for repo in BUNDLE_REPOS:
        src = code_root / repo / "src"
        if not src.is_dir():
            missing.append(repo)
            continue
        checkout = src.parent
        dst = bundle_dir / repo / "src"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=_ignore)
        # Bundle the non-src subdirs the trainer reads at runtime (strategy
        # config), else training dies AFTER a full fit with FileNotFoundError.
        for extra in EXTRA_BUNDLE_SUBDIRS.get(repo, ()):
            esrc = checkout / extra
            if esrc.exists():
                edst = bundle_dir / repo / extra
                if edst.exists():
                    shutil.rmtree(edst)
                shutil.copytree(esrc, edst, ignore=_ignore)
        head = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        if not head:
            unpinned.append(repo)
        heads[repo] = head or "unknown"

    if missing:
        raise RuntimeError(
            f"bundle_code: pinned assembly at {code_root} is missing required "
            f"repo src trees {missing}. Point --code-root at the assembly whose "
            "<repo>/src hold the pinned checkouts — refusing a partial assembly "
            "so a WF corpus cannot be sourced from an ambient/arbitrary checkout."
        )
    if unpinned:
        raise RuntimeError(
            f"bundle_code: staged repos {unpinned} have no resolvable git HEAD "
            f"under {code_root} — refusing an unpinned assembly (every fold's "
            "provenance must name the exact commit it was built from)."
        )
    if assembly_lock:
        drift = {r: {"staged": heads.get(r), "lock": assembly_lock.get(r)}
                 for r in assembly_lock if heads.get(r) != assembly_lock.get(r)}
        if drift:
            raise RuntimeError(
                f"bundle_code: staged commit(s) drifted from the candidate lock: "
                f"{drift}. Refusing to dispatch a corpus whose code does not match "
                "the reviewed lock."
            )
    _assert_fresh_driver(bundle_dir)
    _assert_strategy_config(bundle_dir)
    return heads


def _assert_strategy_config(bundle_dir: Path) -> None:
    """Fail closed if the strategy config the trainer needs isn't bundled.

    ``hf_trainer.build_config_contract()`` reads
    ``<assembly>/renquant-strategy-104/configs/strategy_config.json`` at the END
    of a fit; a bundle without it wastes a full training run then dies with
    FileNotFoundError. Stage it (or a shadow) or refuse to dispatch.
    """
    cfg_dir = bundle_dir / "renquant-strategy-104" / "configs"
    wanted = ("strategy_config.json", "strategy_config.shadow.json")
    if not any((cfg_dir / w).exists() for w in wanted):
        raise RuntimeError(
            "bundle is missing renquant-strategy-104/configs/strategy_config.json"
            " — hf_trainer.build_config_contract() needs it and would fail AFTER "
            "a full fit. Refusing to dispatch."
        )


def _assert_fresh_driver(bundle_dir: Path) -> None:
    """Fail closed if the bundled WF driver is a pre-#74 (script-path) copy.

    A stale ``renquant-backtesting`` checkout at ``code_root`` would bundle a
    driver that shells out to the removed ``scripts/patchtst_hf.py`` instead of
    ``python -m renquant_model_patchtst.hf_trainer`` — producing an all-failed
    corpus (or, worse, a silently wrong one). Refuse to stage it.
    """
    drv = (bundle_dir / "renquant-backtesting" / "src" / "renquant_backtesting"
           / "wf_gate" / "train_walkforward_patchtst.py")
    if not drv.exists():
        raise RuntimeError(f"bundle missing WF driver: {drv}")
    text = drv.read_text()
    if "renquant_model_patchtst.hf_trainer" not in text or (
            "scripts/patchtst_hf.py" in text and "TRAIN_SCRIPT" in text):
        raise RuntimeError(
            "bundle_code staged a STALE (pre-#74) WF driver that invokes "
            "scripts/patchtst_hf.py — refusing. Point the bundle at the reviewed "
            "checkout (the assembly this executor runs from)."
        )


def _file_sha256(path: Path) -> str:
    """Streaming SHA-256 of a file's CONTENT (chunked for large parquet panels)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_inputs_to_volume(plan: WfRescorePlan, *, bundle_dir: Path,
                           dataset_path: Path, raw_label_path: Path,
                           volume_name: str = VOLUME_NAME) -> dict[str, Any]:
    """Batch-upload the code bundle + the two parquet panels to the Volume.

    Layout on the Volume (mounted at ``/data`` in the container):
      * ``/app/repos/<repo>/src``  — code bundle
      * ``/data/<dataset>.parquet`` — training panel (kept under ``data/`` so the
        driver's ``--dataset data/...`` resolves against ``--repo-root /data``)
      * ``/data/<rawlabel>.parquet`` — calibrator raw-label panel

    ``modal`` is imported here (lazily).

    Provenance (codex #76 blocker 2): ``volume_commit_id`` is a digest of every
    staged file's CONTENT (SHA-256), not its size — so same-size code/data changes
    can no longer share a provenance id. The two leakage-relevant DATA panels also
    get explicit per-file content digests in the return under ``data_digests``.
    """
    import modal  # noqa: PLC0415

    vol = modal.Volume.from_name(volume_name, create_if_missing=True)
    uploaded: list[tuple[str, str]] = []  # (local, remote)
    for repo_src in sorted(bundle_dir.rglob("*")):
        if repo_src.is_file():
            rel = repo_src.relative_to(bundle_dir)
            uploaded.append((str(repo_src), f"/app/repos/{rel.as_posix()}"))
    dataset_remote = f"/data/{Path(plan.dataset).name}"
    rawlabel_remote = f"/data/{Path(plan.raw_label_panel).name}"
    uploaded.append((str(dataset_path), dataset_remote))
    uploaded.append((str(raw_label_path), rawlabel_remote))

    hasher = hashlib.sha256()
    data_digests: dict[str, str] = {}
    with vol.batch_upload(force=True) as batch:
        for local, remote in uploaded:
            batch.put_file(local, remote)
            content_sha = _file_sha256(Path(local))
            hasher.update(remote.encode())
            hasher.update(b"\0")
            hasher.update(content_sha.encode())
            if remote in (dataset_remote, rawlabel_remote):
                data_digests[remote] = "sha256:" + content_sha
    commit_id = "sha256:" + hasher.hexdigest()[:16]
    log.info("staged %d files to Volume %s (content-commit=%s)",
             len(uploaded), volume_name, commit_id)
    return {"volume_name": volume_name, "volume_commit_id": commit_id,
            "n_files": len(uploaded), "data_digests": data_digests}


# ── Dispatch ─────────────────────────────────────────────────────────────────
def _import_app_with_env(gpu: str, timeout_s: int, retries: int):
    """Set the ``RENQUANT_WF_MODAL_*`` env vars then import the app module.

    Modal bakes gpu/timeout/retries into ``@app.function`` at import time, so
    they must be in the environment BEFORE the app module is first imported
    (identical constraint + guard to the orchestrator cloud executor).

    The app is a STANDALONE top-level module (``wf_patchtst_modal_app``), NOT
    under ``renquant_backtesting`` — see that module's docstring for why (Modal
    imports the worker's defining module at container load, before the pinned
    Volume bundle is on ``sys.path``, so it must import with only ``os + modal``).
    """
    module_name = "wf_patchtst_modal_app"
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
    with mod.app.run() as running_app:
        log.info("Modal app dispatched: app_id=%s folds=%d gpu=%s",
                 getattr(running_app, "app_id", "?"), len(payloads), plan.gpu)
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
    # The distinct Modal-built image ids the pods actually ran (the RESOLVED,
    # immutable image snapshot — a stronger dep lock than the spec fingerprint).
    resolved_image_ids = sorted({
        r.get("code_image_id") for r in results
        if r.get("ok") and r.get("code_image_id") not in (None, "unknown")
    })
    n_requested = len(plan.cutoffs)
    n_succeeded = len(entries)
    return {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "recipe_id": plan.recipe_id,
        "recipe": plan.recipe,
        "run_id": plan.run_id,
        "built_by": "renquant_backtesting.wf_gate.modal.executor",
        "expert_role": "patchtst_fresh_2nd_expert",
        "goal": "GOAL-2 AC2/AC3 (fresh PatchTST 2nd expert for GOAL-4 ensemble)",
        "manifest": manifest_path,
        "n_folds_requested": n_requested,
        "n_folds_succeeded": n_succeeded,
        # A run is promotable ONLY when every requested fold succeeded; a partial
        # corpus is quarantined and must not be promoted to the serving manifest.
        "promotion_ready": bool(n_requested > 0 and n_succeeded == n_requested),
        "quarantined": bool(n_succeeded != n_requested),
        "modal": {
            "app_name": APP_NAME,
            "gpu": plan.gpu,
            "image_spec_sha256": image_spec_fingerprint(),
            "resolved_image_ids": resolved_image_ids,
            "volume_name": staging.get("volume_name"),
            "volume_commit_id": staging.get("volume_commit_id"),
            "data_digests": staging.get("data_digests") or {},
            "code_git_heads": code_heads,
        },
        "folds": fold_prov,
        "pod_facts": pod_facts,
        "failed_folds": failed,
    }


def _assert_not_canonical_manifest(manifest_output: Path,
                                   strategy_artifacts: Path) -> None:
    """Refuse to let the executor write the canonical serving manifest.

    Promotion to ``walkforward_patchtst_manifest.json`` (the name the WF gate
    consumes) is a SEPARATE reviewed step that must validate every requested fold
    first; this executor only ever writes into a quarantined run namespace
    (codex #76 blocker 3).
    """
    canonical = (strategy_artifacts / CANONICAL_SERVING_MANIFEST).resolve()
    if manifest_output.resolve() == canonical:
        raise RuntimeError(
            "refusing to write the canonical serving manifest "
            f"{canonical} from the WF re-score executor. It writes a quarantined "
            f"run-namespaced manifest under {RUN_NAMESPACE_ROOT}/<run_id>/; "
            "promotion to the serving name is a separate reviewed step."
        )


def collect_and_write(plan: WfRescorePlan, results: list[dict[str, Any]], *,
                      repo_root: Path, code_heads: dict[str, str],
                      staging: dict[str, Any]) -> dict[str, Any]:
    """Materialise artifacts, write the manifest + provenance sidecar locally.

    All outputs land under a quarantined run namespace
    (``.../artifacts/walkforward_patchtst_runs/<run_id>/``); the canonical serving
    manifest is never written here.
    """
    strategy_artifacts = repo_root / "backtesting" / plan.strategy / "artifacts"
    run_dir = strategy_artifacts / RUN_NAMESPACE_ROOT / plan.run_id
    entries = []
    for r in results:
        if not r.get("ok"):
            continue
        entries.append(collect_fold_artifacts(r, strategy_artifacts,
                                               plan.artifact_root))
    manifest_output = (Path(plan.manifest_output) if plan.manifest_output
                       else run_dir / CANONICAL_SERVING_MANIFEST)
    _assert_not_canonical_manifest(manifest_output, strategy_artifacts)
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
    log.info("wrote %d/%d folds (run_id=%s promotion_ready=%s); "
             "manifest=%s provenance=%s",
             len(entries), len(plan.cutoffs), plan.run_id,
             provenance["promotion_ready"], manifest_path, prov_path)
    return {"manifest": manifest_path, "provenance": str(prov_path),
            "n_folds": len(entries), "provenance_obj": provenance,
            "promotion_ready": provenance["promotion_ready"]}


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
    p.add_argument("--run-id", default=None,
                   help="Isolated run namespace for artifacts + manifest "
                        "(default: wf-pt-<recipe8>-<utc>). NEVER the canonical "
                        "serving tree; promotion is a separate reviewed step.")
    p.add_argument("--code-root", default=None,
                   help="SINGLE pinned-assembly root holding <repo>/src for every "
                        "bundled repo (default: the assembly THIS executor runs "
                        "from). No ~/git/github fallback — fail closed if any repo "
                        "is missing.")
    p.add_argument("--assembly-lock", default=None,
                   help="Optional JSON file {repo: git_sha} the staged bundle "
                        "commits must match exactly (fail closed on drift).")
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
    print(f"  run_id        : {plan.run_id}")
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
    # ONE explicit pinned assembly (codex #76 blocker 1): the reviewed checkout
    # this executor runs from, or an explicit --code-root. NO ~/git/github
    # fallback and NO per-repo search — bundle_code fails closed if the single
    # root is missing any required repo, so a corpus can't be sourced from an
    # ambient/arbitrary checkout.
    code_root = (Path(args.code_root).expanduser().resolve()
                 if args.code_root else _EXECUTOR_CHECKOUT_ROOT.parent)
    assembly_lock = None
    if args.assembly_lock:
        assembly_lock = json.loads(Path(args.assembly_lock).read_text())
    with tempfile.TemporaryDirectory(prefix="wf-pt-bundle-") as td:
        bundle_dir = Path(td)
        code_heads = bundle_code(bundle_dir, code_root,
                                 assembly_lock=assembly_lock)
        staging = stage_inputs_to_volume(
            plan, bundle_dir=bundle_dir, dataset_path=dataset_path,
            raw_label_path=raw_label_path)
        results = dispatch_folds(
            plan, timeout_s=args.timeout_seconds, retries=args.retries,
            volume_commit_id=staging.get("volume_commit_id"))
    out = collect_and_write(
        plan, results, repo_root=repo_root, code_heads=code_heads,
        staging=staging)
    print(f"\nDONE: {out['n_folds']}/{len(plan.cutoffs)} folds "
          f"(run_id={plan.run_id})")
    print(f"  manifest   : {out['manifest']}  [QUARANTINED run namespace]")
    print(f"  provenance : {out['provenance']}")
    if out["promotion_ready"]:
        print("  status     : all folds succeeded — eligible for a SEPARATE "
              "reviewed promotion to the serving manifest.")
        return 0
    # A partial corpus is quarantined, NOT valid evidence: exit nonzero so no
    # caller mistakes it for a complete, promotable run.
    print(f"  status     : PARTIAL ({out['n_folds']}/{len(plan.cutoffs)}) — "
          "QUARANTINED, not promotable. Re-run the missing folds.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
