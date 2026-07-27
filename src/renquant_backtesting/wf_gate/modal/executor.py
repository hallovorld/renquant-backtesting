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

Bounded two-phase dispatch (model#82 P0 — a hard budget cap must be
operationally enforceable, not a prose tripwire)::

    # Phase 1 — pilot: dispatch EXACTLY three explicit folds into a named run
    python -m renquant_backtesting.wf_gate.modal.executor \\
        --run-id wf-pt-pilot --select-cutoffs 2023-10-02,2024-12-02,2026-03-02 \\
        --gpu T4 --execute

    # Between phases — observed-cost projection (pure stdout, NO dispatch)
    python -m renquant_backtesting.wf_gate.modal.executor \\
        --print-cost-projection --run-id wf-pt-pilot \\
        --project-folds 40 --rate-usd-per-hour 0.59

    # Phase 2 — resume INTO the same run: pilot folds that already exist and
    # pass integrity are SKIPPED (never retrained/overwritten); exactly the
    # remaining folds dispatch; ONE manifest is rebuilt over the union.
    python -m renquant_backtesting.wf_gate.modal.executor \\
        --run-id wf-pt-pilot --gpu T4 --execute \\
        --dispatch-note "projection $15.6 < remaining cap $18.55 -> GO"
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


def select_explicit_cutoffs(cutoffs: list[str], spec: str) -> list[str]:
    """Explicit, auditable fold selection (model#82 P0 bounded dispatch).

    ``spec`` is a comma-separated list of ISO cutoff dates. Every date must be
    a member of the computed corpus grid (``start/end/cadence``-derived), with
    no duplicates — anything else is a hard error, so the dispatched set is
    EXACTLY what the prereg froze. The result is normalised to grid
    (chronological) order regardless of input order, for determinism.
    """
    wanted = [s.strip() for s in str(spec).split(",") if s.strip()]
    if not wanted:
        raise ValueError("--select-cutoffs given but no dates could be parsed "
                         f"from {spec!r}")
    dupes = sorted({c for c in wanted if wanted.count(c) > 1})
    if dupes:
        raise ValueError(f"--select-cutoffs contains duplicate dates: {dupes}")
    unknown = sorted(set(wanted) - set(cutoffs))
    if unknown:
        raise ValueError(
            f"--select-cutoffs dates not on the corpus grid: {unknown}. The "
            "grid is derived from --start-date/--end-date/--cadence-days "
            f"({cutoffs[0]} .. {cutoffs[-1]}, {len(cutoffs)} folds); an "
            "off-grid cutoff would break corpus identity — refusing."
        )
    selected = set(wanted)
    return [c for c in cutoffs if c in selected]


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
    select_spec = getattr(args, "select_cutoffs", None)
    if select_spec:
        cutoffs = select_explicit_cutoffs(all_cutoffs, select_spec)
    else:
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
                   volume_commit_id: str | None
                   ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fan out one pod per fold via ``train_fold_remote.map`` and collect JSON.

    Mirrors the orchestrator ``execute_batch`` dispatch: ``with app.run():`` +
    ``.map(order_outputs=False, return_exceptions=True)`` so a single fold's
    failure is reported, not fatal to the batch.

    Returns ``(results, dispatch_info)`` where ``dispatch_info`` carries the
    Modal ``app_id`` (+ the dispatched cutoffs) for the provenance sidecar's
    per-dispatch audit record (model#82 P0).
    """
    mod = _import_app_with_env(plan.gpu, timeout_s, retries)
    payloads = []
    for req in plan.fold_requests:
        r = dict(req)
        r["volume_commit_id"] = volume_commit_id
        payloads.append(json.dumps(r))

    results: list[dict[str, Any]] = []
    dispatch_info: dict[str, Any] = {
        "app_id": None,
        "dispatched_cutoffs": [r["cutoff_date"] for r in plan.fold_requests],
    }
    with mod.app.run() as running_app:
        dispatch_info["app_id"] = getattr(running_app, "app_id", None)
        log.info("Modal app dispatched: app_id=%s folds=%d gpu=%s",
                 dispatch_info["app_id"] or "?", len(payloads), plan.gpu)
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
    return results, dispatch_info


# ── Artifact collection + manifest + provenance ──────────────────────────────
def _write_bytes_b64gz(b64gz: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(gzip.decompress(base64.b64decode(b64gz)))


def collect_fold_artifacts(result: dict[str, Any], strategy_artifacts: Path,
                           artifact_root: str) -> dict[str, Any]:
    """Materialise one pod's returned artifacts under the local strategy tree.

    Returns the manifest-entry dict (already carrying effective_train_cutoff_date).

    Tolerant of an incomplete pod payload (a worker can report ``ok`` yet omit a
    model/sidecar/calibrator blob): whatever IS present is written, whatever is
    missing is simply not materialised — the promotion gate
    (:func:`validate_fold_promotable`) decides eligibility from what actually
    landed on disk, so a partial fold cannot KeyError the collector nor slip
    through as promotion-ready.
    """
    cutoff = result["cutoff_date"]
    out_dir = strategy_artifacts / artifact_root / cutoff
    arts = result.get("artifacts") or {}
    model_rel = result["entry"]["artifact_uri"]
    # ``artifact_uri`` on the pod is an absolute container path; re-root it under
    # the local strategy artifacts tree by filename so we stay independent of the
    # container's paths.
    model_path = out_dir / Path(model_rel).name
    if arts.get("model_pt_b64gz"):
        _write_bytes_b64gz(arts["model_pt_b64gz"], model_path)
    if arts.get("sidecar_json"):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / (model_path.name + ".metadata.json")).write_text(
            arts["sidecar_json"])
    entry = dict(result["entry"])
    entry["artifact_uri"] = str(model_path)
    if arts.get("calibrator_json"):
        cal_path = model_path.with_name("hf_patchtst-calibration.json")
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        cal_path.write_text(arts["calibrator_json"])
        entry["calibrator_uri"] = str(cal_path)
    return entry


def _norm_date(value: Any) -> str | None:
    """ISO-date prefix (``YYYY-MM-DD``) of a date/timestamp/string, else None."""
    if value in (None, ""):
        return None
    return str(value).strip()[:10]


def _sidecar_path_for_model(model_path: Path) -> Path:
    return model_path.with_name(model_path.name + ".metadata.json")


def validate_fold_promotable(entry: dict[str, Any], *,
                             skip_calibrators: bool) -> tuple[bool, list[str]]:
    """Fail-closed check that ONE collected fold is fit to promote to serving.

    A fold is promotable only when ALL of the following hold on disk:

    1. the model ``.pt`` is present AND non-empty;
    2. a metadata sidecar exists, parses, and its ``training_contract``
       (a) names a ``train_cutoff_date`` that AGREES with the requested fold,
       (b) carries ``trained_date`` + ``effective_train_cutoff_date``, and
       (c) carries provenance/recipe (``dataset`` + non-empty ``hyperparameters``);
    3. a calibrator payload is present, parses, and is non-empty — UNLESS this is
       a ``--skip-calibrators`` diagnostic run, in which case the fold is never
       promotable regardless (the caller also enforces this at run level).

    Returns ``(promotable, reasons)``; ``reasons`` lists every gap for the
    provenance audit trail. This validates ONLY what materialised — it never
    changes the worker/training contract.
    """
    reasons: list[str] = []
    cutoff = entry.get("cutoff_date")

    # (1) model artifact present + non-empty
    model_uri = entry.get("artifact_uri")
    model_path = Path(model_uri) if model_uri else None
    if not model_path or not model_path.exists():
        reasons.append("model_pt_missing")
    elif model_path.stat().st_size == 0:
        reasons.append("model_pt_empty")

    # (2) validated metadata sidecar (cutoff agrees + provenance/recipe present)
    if model_path is None:
        reasons.append("sidecar_unreachable_no_model")
    else:
        sidecar_path = _sidecar_path_for_model(model_path)
        if not sidecar_path.exists():
            reasons.append("sidecar_missing")
        else:
            sidecar: dict[str, Any] | None
            try:
                sidecar = json.loads(sidecar_path.read_text())
            except (ValueError, OSError):
                sidecar = None
                reasons.append("sidecar_unparseable")
            if isinstance(sidecar, dict):
                contract = sidecar.get("training_contract") or {}
                if not contract:
                    reasons.append("sidecar_no_training_contract")
                else:
                    sc_cutoff = _norm_date(contract.get("train_cutoff_date"))
                    req_cutoff = _norm_date(cutoff)
                    if sc_cutoff is None:
                        reasons.append("sidecar_no_train_cutoff_date")
                    elif sc_cutoff != req_cutoff:
                        reasons.append(
                            f"sidecar_cutoff_mismatch({sc_cutoff}!={req_cutoff})")
                    if not contract.get("trained_date"):
                        reasons.append("sidecar_no_trained_date")
                    if not contract.get("effective_train_cutoff_date"):
                        reasons.append("sidecar_no_effective_train_cutoff_date")
                    if not contract.get("dataset"):
                        reasons.append("sidecar_no_provenance_dataset")
                    if not contract.get("hyperparameters"):
                        reasons.append("sidecar_no_recipe_hyperparameters")

    # (3) calibrator payload — a --skip-calibrators run is diagnostic-only.
    if skip_calibrators:
        reasons.append("skip_calibrators_diagnostic")
    else:
        cal_uri = entry.get("calibrator_uri")
        cal_path = Path(cal_uri) if cal_uri else None
        if not cal_path or not cal_path.exists():
            reasons.append("calibrator_missing")
        elif cal_path.stat().st_size == 0:
            reasons.append("calibrator_empty")
        else:
            try:
                payload = json.loads(cal_path.read_text())
            except (ValueError, OSError):
                payload = None
                reasons.append("calibrator_unparseable")
            if payload is not None and not payload:
                reasons.append("calibrator_empty_payload")

    return (not reasons, reasons)


# ── Resume into an existing run namespace (model#82 P0 bounded dispatch) ─────
#: The one integrity marker that does NOT indicate a broken on-disk fold: it
#: only flags a diagnostic --skip-calibrators run as never-promotable. For
#: resume purposes (skip vs hard-error) it is ignored.
_DIAGNOSTIC_ONLY_REASON = "skip_calibrators_diagnostic"


def _looks_like_iso_date(name: str) -> bool:
    if len(name) != 10 or name[4] != "-" or name[7] != "-":
        return False
    return (name[:4] + name[5:7] + name[8:]).isdigit()


def load_existing_fold_entry(fold_dir: Path, cutoff: str) -> dict[str, Any] | None:
    """Reconstruct one already-materialised fold's manifest entry from disk.

    Returns ``None`` when no unambiguous model ``.pt`` exists in the fold dir
    (zero or multiple candidates) — the caller records that as a failed
    integrity check. ``trained_date`` / ``effective_train_cutoff_date`` come
    from the metadata sidecar's ``training_contract`` (the same contract
    :func:`validate_fold_promotable` validates).
    """
    pts = sorted(p for p in fold_dir.glob("*.pt") if p.is_file())
    if len(pts) != 1:
        return None
    model = pts[0]
    entry: dict[str, Any] = {
        "cutoff_date": cutoff,
        "trained_date": None,
        "effective_train_cutoff_date": None,
        "artifact_uri": str(model),
        "lookahead_days": 60,
    }
    sidecar = _sidecar_path_for_model(model)
    if sidecar.exists():
        try:
            contract = (json.loads(sidecar.read_text())
                        .get("training_contract") or {})
        except (ValueError, OSError):
            contract = {}
        entry["trained_date"] = contract.get("trained_date")
        entry["effective_train_cutoff_date"] = contract.get(
            "effective_train_cutoff_date")
        try:
            entry["lookahead_days"] = int(contract.get("lookahead_days") or 60)
        except (TypeError, ValueError):
            pass
    cal = model.with_name("hf_patchtst-calibration.json")
    if cal.exists():
        entry["calibrator_uri"] = str(cal)
    return entry


def scan_existing_folds(run_art_dir: Path, *,
                        skip_calibrators: bool) -> dict[str, dict[str, Any]]:
    """Inventory every fold already materialised under a run namespace.

    Returns ``{cutoff: {"entry", "promotable", "resume_ok", "reasons"}}`` where
    ``resume_ok`` is the RESUME integrity verdict: the fold passes every
    :func:`validate_fold_promotable` check except the diagnostic-run marker
    (a valid --skip-calibrators fold is skippable on resume even though it is
    never promotable). Any fold directory that exists — even a partial or
    empty one — is reported: something wrote into the namespace, so resume
    must either skip it (valid) or hard-error (invalid), never overwrite it.
    """
    out: dict[str, dict[str, Any]] = {}
    if not run_art_dir.is_dir():
        return out
    for d in sorted(run_art_dir.iterdir()):
        if not d.is_dir() or not _looks_like_iso_date(d.name):
            continue
        entry = load_existing_fold_entry(d, d.name)
        if entry is None:
            n_pt = len(list(d.glob("*.pt")))
            reason = ("model_pt_missing" if n_pt == 0
                      else f"model_pt_ambiguous({n_pt} candidates)")
            out[d.name] = {"entry": None, "promotable": False,
                           "resume_ok": False, "reasons": [reason]}
            continue
        promotable, reasons = validate_fold_promotable(
            entry, skip_calibrators=skip_calibrators)
        blocking = [r for r in reasons if r != _DIAGNOSTIC_ONLY_REASON]
        out[d.name] = {"entry": entry, "promotable": promotable,
                       "resume_ok": not blocking, "reasons": reasons}
    return out


def partition_resume(plan: WfRescorePlan,
                     strategy_artifacts: Path) -> dict[str, Any]:
    """Split the selected cutoffs into skip / dispatch / hard-error sets.

    The bounded-dispatch invariant (model#82 P0): a fold that already exists in
    the run namespace is NEVER retrained or overwritten —

      * exists and passes integrity → ``skipped`` (not dispatched);
      * exists but FAILS integrity  → ``invalid_selected`` (the caller must
        hard-error before any cloud call — ambiguous state, refusing to
        overwrite);
      * absent → ``dispatch``.

    ``existing`` carries the full namespace inventory (selected or not) so the
    manifest + provenance can be rebuilt over the union.
    """
    existing = scan_existing_folds(
        strategy_artifacts / plan.artifact_root,
        skip_calibrators=plan.skip_calibrators)
    skipped = [c for c in plan.cutoffs
               if c in existing and existing[c]["resume_ok"]]
    invalid_selected = {c: list(existing[c]["reasons"]) for c in plan.cutoffs
                        if c in existing and not existing[c]["resume_ok"]}
    dispatch = [c for c in plan.cutoffs if c not in existing]
    return {"existing": existing, "skipped": skipped,
            "dispatch": dispatch, "invalid_selected": invalid_selected}


def _manifest_output_path(plan: WfRescorePlan,
                          strategy_artifacts: Path) -> Path:
    """The run's single manifest path (explicit override or run-namespace default)."""
    if plan.manifest_output:
        return Path(plan.manifest_output)
    return (strategy_artifacts / RUN_NAMESPACE_ROOT / plan.run_id
            / CANONICAL_SERVING_MANIFEST)


def read_prior_provenance(manifest_output: Path) -> dict[str, Any] | None:
    """Load the run's existing provenance sidecar, or None if absent/unreadable."""
    prov_path = Path(str(manifest_output) + ".provenance.json")
    if not prov_path.exists():
        return None
    try:
        loaded = json.loads(prov_path.read_text())
    except (ValueError, OSError):
        return None
    return loaded if isinstance(loaded, dict) else None


# ── Cost projection (model#82 P0 — pure stdout, no dispatch) ─────────────────
def project_cost_from_provenance(provenance: dict[str, Any], n_folds: int,
                                 rate_usd_per_hour: float) -> dict[str, Any]:
    """Project the cost of ``n_folds`` more folds from observed pod runtimes.

    Averages the ``elapsed_seconds`` of every completed pod recorded in the
    run's provenance sidecar (``pod_facts``) and prices ``n_folds`` more at
    ``rate_usd_per_hour``. Raises ``ValueError`` when there is nothing to
    average — a projection with zero observations would be fiction.
    """
    if int(n_folds) <= 0:
        raise ValueError(f"n_folds must be > 0 (got {n_folds})")
    if float(rate_usd_per_hour) <= 0:
        raise ValueError(
            f"rate_usd_per_hour must be > 0 (got {rate_usd_per_hour})")
    pods = {c: f for c, f in (provenance.get("pod_facts") or {}).items()
            if isinstance(f, dict)
            and isinstance(f.get("elapsed_seconds"), (int, float))
            and not isinstance(f.get("elapsed_seconds"), bool)}
    if not pods:
        raise ValueError(
            "provenance has no completed pods with elapsed_seconds — cannot "
            "project cost from zero observations")
    avg = sum(f["elapsed_seconds"] for f in pods.values()) / len(pods)
    per_fold_usd = avg / 3600.0 * float(rate_usd_per_hour)
    return {
        "n_pods_observed": len(pods),
        "observed_cutoffs": sorted(pods),
        "avg_elapsed_seconds": avg,
        "rate_usd_per_hour": float(rate_usd_per_hour),
        "per_fold_usd": per_fold_usd,
        "n_folds_projected": int(n_folds),
        "projected_usd": per_fold_usd * int(n_folds),
    }


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


def _pod_facts_of(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-pod Modal facts for ok results, keyed by cutoff."""
    return {r.get("cutoff_date"): {
        "worker_id": r.get("worker_id"),
        "code_image_id": r.get("code_image_id"),
        "elapsed_seconds": r.get("elapsed_seconds"),
        "device": r.get("device"),
        "result_checksum": r.get("result_checksum"),
    } for r in results if r.get("ok")}


def _failed_of(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"cutoff_date": r.get("cutoff_date"), "error": r.get("error")}
            for r in results if not r.get("ok")]


def build_provenance(plan: WfRescorePlan, results: list[dict[str, Any]],
                     entries: list[dict[str, Any]], *, code_heads: dict[str, str],
                     staging: dict[str, Any], manifest_path: str,
                     fold_validation: dict[str, dict[str, Any]] | None = None,
                     requested_cutoffs: list[str] | None = None,
                     dispatches: list[dict[str, Any]] | None = None,
                     prior_pod_facts: dict[str, dict[str, Any]] | None = None,
                     ) -> dict[str, Any]:
    """The FRESH-corpus provenance sidecar (GOAL-2 AC2/AC3 stamps).

    ``fold_validation`` maps each collected cutoff → the
    :func:`validate_fold_promotable` verdict (``{"promotable": bool,
    "reasons": [...]}``). Promotion eligibility FAILS CLOSED: the run is
    ``promotion_ready`` only when EVERY requested fold is promotable AND this is
    not a diagnostic ``--skip-calibrators`` run; anything else stays quarantined
    (codex #76). ``fold_validation`` defaults to empty → nothing promotable.

    Resume support (model#82 P0): ``requested_cutoffs`` is the UNION of this
    invocation's selection and every fold already in the run namespace (default:
    this invocation's cutoffs); ``dispatches`` is the full per-dispatch audit
    history to persist; ``prior_pod_facts`` are earlier dispatches' pod facts,
    merged under this invocation's (so the sidecar stays the run's ONE record).
    ``failed_folds`` reflects only THIS invocation — a previously failed fold
    that was re-dispatched successfully must not linger as a failure; each
    dispatch record in ``dispatches`` retains its own failure list.
    """
    fold_validation = fold_validation or {}
    fold_prov = []
    for e in entries:
        verdict = fold_validation.get(e["cutoff_date"], {})
        fold_prov.append({
            "cutoff_date": e["cutoff_date"],
            "trained_date": e["trained_date"],
            "effective_train_cutoff_date": e.get("effective_train_cutoff_date"),
            "artifact_uri": e["artifact_uri"],
            "calibrator_uri": e.get("calibrator_uri"),
            "promotable": bool(verdict.get("promotable")),
            "quarantine_reasons": list(verdict.get("reasons") or []),
        })
    pod_facts = dict(prior_pod_facts or {})
    pod_facts.update(_pod_facts_of(results))
    failed = _failed_of(results)
    # The distinct Modal-built image ids the pods actually ran (the RESOLVED,
    # immutable image snapshot — a stronger dep lock than the spec fingerprint).
    resolved_image_ids = sorted({
        r.get("code_image_id") for r in results
        if r.get("ok") and r.get("code_image_id") not in (None, "unknown")
    })
    requested = (list(requested_cutoffs) if requested_cutoffs is not None
                 else list(plan.cutoffs))
    n_requested = len(requested)
    n_succeeded = len(entries)
    promotable_cutoffs = sorted(
        c for c, v in fold_validation.items() if v.get("promotable"))
    n_promotable = len(promotable_cutoffs)
    # FAIL CLOSED (codex #76): a run is promotion_ready ONLY when every requested
    # fold has a materialised+non-empty model, a validated metadata sidecar whose
    # cutoff/provenance/recipe agree, AND a valid calibrator — and never for a
    # diagnostic --skip-calibrators run. Fold counts alone are NOT sufficient:
    # a missing/invalid sidecar or calibrator keeps the run quarantined even at
    # n_succeeded == n_requested.
    promotion_ready = bool(
        n_requested > 0
        and not plan.skip_calibrators
        and n_promotable == n_requested
    )
    quarantine_reasons = sorted({
        r for v in fold_validation.values() for r in (v.get("reasons") or [])})
    if plan.skip_calibrators and "skip_calibrators_diagnostic" not in quarantine_reasons:
        quarantine_reasons.append("skip_calibrators_diagnostic")
    return {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "recipe_id": plan.recipe_id,
        "recipe": plan.recipe,
        "run_id": plan.run_id,
        "built_by": "renquant_backtesting.wf_gate.modal.executor",
        "expert_role": "patchtst_fresh_2nd_expert",
        "goal": "GOAL-2 AC2/AC3 (fresh PatchTST 2nd expert for GOAL-4 ensemble)",
        "manifest": manifest_path,
        "requested_cutoffs": requested,
        "n_folds_requested": n_requested,
        "n_folds_succeeded": n_succeeded,
        "n_folds_promotable": n_promotable,
        "promotion_ready": promotion_ready,
        "quarantined": not promotion_ready,
        "promotion_gate": {
            "requires": [
                "every requested fold has a non-empty model .pt",
                "every requested fold has a validated metadata sidecar "
                "(cutoff agrees; provenance + recipe present)",
                "every requested fold has a valid calibrator",
                "run is not --skip-calibrators (diagnostic)",
            ],
            "skip_calibrators": bool(plan.skip_calibrators),
            "n_folds_requested": n_requested,
            "n_folds_promotable": n_promotable,
            "promotable_cutoffs": promotable_cutoffs,
            "quarantine_reasons": quarantine_reasons,
        },
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
        "dispatches": list(dispatches or []),
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


def _manifest_worthy(entry: dict[str, Any] | None) -> bool:
    """Only a materialised, non-empty model with a dated contract may be listed."""
    if not entry or not entry.get("artifact_uri") or not entry.get("trained_date"):
        return False
    model_path = Path(entry["artifact_uri"])
    return model_path.exists() and model_path.stat().st_size > 0


def collect_and_write(plan: WfRescorePlan, results: list[dict[str, Any]], *,
                      repo_root: Path, code_heads: dict[str, str],
                      staging: dict[str, Any],
                      existing_folds: dict[str, dict[str, Any]] | None = None,
                      dispatch_meta: dict[str, Any] | None = None,
                      ) -> dict[str, Any]:
    """Materialise artifacts, write the manifest + provenance sidecar locally.

    All outputs land under a quarantined run namespace
    (``.../artifacts/walkforward_patchtst_runs/<run_id>/``); the canonical serving
    manifest is never written here.

    Resume (model#82 P0): ``existing_folds`` is the run namespace's prior-fold
    inventory (:func:`scan_existing_folds`). Existing folds are NEVER
    overwritten — a returned pod result colliding with one is a hard error —
    and the run's ONE manifest is REBUILT over the union of existing + new
    folds via the same reviewed :func:`assemble_manifest` writer. The
    provenance sidecar is likewise rebuilt over the union, appending one
    per-dispatch audit record (``dispatch_meta`` + this invocation's pod facts
    and failures) to the run's ``dispatches`` history.
    """
    existing_folds = existing_folds or {}
    strategy_artifacts = repo_root / "backtesting" / plan.strategy / "artifacts"
    manifest_output = _manifest_output_path(plan, strategy_artifacts)
    _assert_not_canonical_manifest(manifest_output, strategy_artifacts)
    prior_raw = read_prior_provenance(manifest_output)
    # One-run-one-recipe, enforced at the FUNCTION SEAM too (PR #81 review
    # MED; second belt behind main()'s pre-dispatch refusal). Existing folds
    # may only be absorbed into a union rebuild under a readable prior
    # provenance whose recipe_id matches this plan. The per-fold
    # .metadata.json carries NO run-level recipe identity (hf_trainer's
    # training_contract has no recipe_id and omits run-level fields such as
    # cadence_days/calibrator_method), so with the sidecar missing/unreadable
    # the invariant is UNVERIFIABLE → fail closed rather than silently
    # restamp the run's provenance with a new recipe_id.
    if existing_folds:
        if prior_raw is None:
            raise RuntimeError(
                f"run namespace {plan.run_id!r} holds {len(existing_folds)} "
                "existing fold(s) but its provenance sidecar "
                f"({manifest_output}.provenance.json) is missing or unreadable "
                "— the one-run-one-recipe invariant cannot be verified "
                "(per-fold metadata sidecars carry no run-level recipe_id). "
                "Refusing to rebuild/restamp; restore the sidecar or use a "
                "fresh --run-id.")
        prior_recipe = prior_raw.get("recipe_id")
        if prior_recipe and prior_recipe != plan.recipe_id:
            raise RuntimeError(
                f"run namespace {plan.run_id!r} was built with recipe_id "
                f"{prior_recipe} but this rebuild computes {plan.recipe_id} — "
                "one run namespace = one recipe; refusing to restamp a mixed "
                "corpus.")
    prior = prior_raw or {}

    entries: list[dict[str, Any]] = []
    fold_validation: dict[str, dict[str, Any]] = {}
    # Existing folds first: reconstructed from disk, never re-collected.
    for cutoff, info in existing_folds.items():
        fold_validation[cutoff] = {
            "promotable": bool(info.get("promotable")),
            "reasons": list(info.get("reasons") or [])}
        if _manifest_worthy(info.get("entry")):
            entries.append(dict(info["entry"]))
    for r in results:
        if not r.get("ok"):
            continue
        if r.get("cutoff_date") in existing_folds:
            raise RuntimeError(
                f"pod returned cutoff {r.get('cutoff_date')} which ALREADY "
                f"exists in run namespace {plan.run_id!r} — refusing to "
                "overwrite an existing fold (model#82 P0: pilot folds are "
                "never retrained/overwritten).")
        entry = collect_fold_artifacts(r, strategy_artifacts, plan.artifact_root)
        promotable, reasons = validate_fold_promotable(
            entry, skip_calibrators=plan.skip_calibrators)
        fold_validation[entry["cutoff_date"]] = {
            "promotable": promotable, "reasons": reasons}
        # Only a materialised (present + non-empty) model belongs in the manifest;
        # an ``ok`` fold that returned no model blob is recorded (quarantined) but
        # never referenced as a serving artifact.
        if _manifest_worthy(entry):
            entries.append(entry)
    entries.sort(key=lambda e: str(e["cutoff_date"]))

    manifest_path = ""
    if entries:
        manifest_path = str(assemble_manifest(
            entries, plan.recipe["cadence_days"], manifest_output))

    # ONE auditable corpus: requested = this selection ∪ every namespace fold.
    requested_union = sorted(set(plan.cutoffs) | set(existing_folds))
    dispatches = list(prior.get("dispatches") or [])
    if dispatch_meta is not None:
        from datetime import datetime, timezone  # noqa: PLC0415
        record = dict(dispatch_meta)
        record.setdefault(
            "dispatched_at",
            datetime.now(timezone.utc).isoformat(timespec="seconds"))
        record.setdefault("gpu", plan.gpu)
        record.setdefault("volume_commit_id", staging.get("volume_commit_id"))
        record["pod_facts"] = _pod_facts_of(results)
        record["failed_folds"] = _failed_of(results)
        dispatches.append(record)

    provenance = build_provenance(
        plan, results, entries, code_heads=code_heads, staging=staging,
        manifest_path=manifest_path, fold_validation=fold_validation,
        requested_cutoffs=requested_union, dispatches=dispatches,
        prior_pod_facts=prior.get("pod_facts") or {})
    prov_path = Path(str(manifest_output) + ".provenance.json")
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    prov_path.write_text(json.dumps(provenance, indent=2, sort_keys=True))
    log.info("wrote %d/%d folds (run_id=%s promotion_ready=%s); "
             "manifest=%s provenance=%s",
             len(entries), len(requested_union), plan.run_id,
             provenance["promotion_ready"], manifest_path, prov_path)
    return {"manifest": manifest_path, "provenance": str(prov_path),
            "n_folds": len(entries), "provenance_obj": provenance,
            "promotion_ready": provenance["promotion_ready"]}


# ── CLI ──────────────────────────────────────────────────────────────────────
def _assert_panel_fresh_or_report(plan: WfRescorePlan,
                                  args: argparse.Namespace,
                                  dataset_path: Path) -> int:
    """AC7 fail-closed freshness/coverage gate (GOAL-5). Returns 0 on pass,
    2 (fail-closed, matching this CLI's input-error code) on breach, printing
    the contract's reasons. Uses the same canonical renquant-common contract
    and the #74 driver's ``data_end_for_cutoff`` so the union-window logic is
    shared, not re-implemented."""
    import pandas as pd  # noqa: PLC0415

    from renquant_backtesting.wf_gate.train_walkforward_patchtst import (  # noqa: PLC0415
        data_end_for_cutoff,
    )
    from renquant_common.training_freshness import (  # noqa: PLC0415
        assess_training_panel_freshness,
    )

    if not plan.cutoffs:
        print("\nAC7 freshness gate: no folds planned — nothing to check.")
        return 0
    required = max(
        pd.Timestamp(data_end_for_cutoff(pd.Timestamp(c), args.label))
        for c in plan.cutoffs
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
        print("\nAC7 FRESHNESS GATE FAILED — refusing to stage/dispatch a "
              "stale/truncated panel:")
        print(f"  panel            : {dataset_path}")
        print(f"  required_through : {required.date()}")
        print(f"  panel_max_date   : {panel_max}")
        for r in verdict.reasons:
            print(f"  - {r}")
        return 2
    log.info(
        "AC7 freshness gate PASS: panel=%s covers required_through_date=%s "
        "(max_date=%s, n_rows=%d)", dataset_path, required.date(),
        verdict.max_date.date(), verdict.n_rows,
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start-date", default="2023-10-02")
    p.add_argument("--end-date", default="2026-03-02")
    p.add_argument("--cadence-days", type=int, default=21)
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--staged", type=int, default=None,
                     help="Run only the N most-recent folds (directional read).")
    sel.add_argument("--select-cutoffs", default=None,
                     help="Comma-separated ISO cutoff dates to run — each must "
                          "be on the start/end/cadence corpus grid (duplicates "
                          "or off-grid dates are hard errors). Deterministic, "
                          "auditable bounded dispatch (model#82 P0); mutually "
                          "exclusive with --staged.")
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
                        "serving tree; promotion is a separate reviewed step. "
                        "Naming an EXISTING run resumes into it: folds already "
                        "on disk that pass integrity are skipped (never "
                        "retrained/overwritten), an existing-but-invalid "
                        "selected fold is a hard error, and the run's single "
                        "manifest + provenance sidecar are rebuilt over the "
                        "union (model#82 P0).")
    p.add_argument("--dispatch-note", default=None,
                   help="Optional note stamped into this dispatch's provenance "
                        "audit record — e.g. the observed-cost GO decision "
                        "between the pilot and the remainder (model#82 P0).")
    # ── Cost projection (pure stdout, no dispatch; model#82 P0) ──────────────
    p.add_argument("--print-cost-projection", action="store_true",
                   help="Read the run's provenance sidecar (requires --run-id), "
                        "average completed pods' elapsed_seconds, and print the "
                        "projected cost of --project-folds more folds at "
                        "--rate-usd-per-hour. Pure stdout — never dispatches "
                        "(and refuses --execute).")
    p.add_argument("--project-folds", type=int, default=None,
                   help="N additional folds to project the cost of "
                        "(required with --print-cost-projection).")
    p.add_argument("--rate-usd-per-hour", type=float, default=None,
                   help="GPU $/hour used for the projection — explicit on "
                        "purpose, no baked-in price (required with "
                        "--print-cost-projection).")
    p.add_argument("--code-root", default=None,
                   help="SINGLE pinned-assembly root holding <repo>/src for every "
                        "bundled repo (default: the assembly THIS executor runs "
                        "from). No ~/git/github fallback — fail closed if any repo "
                        "is missing.")
    p.add_argument("--assembly-lock", default=None,
                   help="Optional JSON file {repo: git_sha} the staged bundle "
                        "commits must match exactly (fail closed on drift).")
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
                        "historical windows, so COVERAGE is the load-bearing "
                        "check, not calendar recency.")
    p.add_argument("--timeout-seconds", type=int, default=7200,
                   help="Per-fold Modal function timeout. The 2026-07-27 "
                        "staged-1 T4 probe (1 fold) measured train=2388.1s "
                        "and the calibrator leg still running when the old "
                        "3600s default killed the fold (FunctionTimeoutError) "
                        "— 3600s is confirmed below at least one "
                        "production-like T4 fold's true runtime.")
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--dry-run", action="store_true",
                   help="Plan only: print folds + recipe_id, make no cloud calls.")
    p.add_argument("--execute", action="store_true",
                   help="Actually dispatch to Modal (default is plan-only).")
    return p.parse_args(argv)


def resolve_repo_root(value: str | None) -> Path:
    from renquant_backtesting.repo_root import resolve_repo_root as _rrr  # noqa: PLC0415
    return _rrr(value)


def _print_cost_projection(args: argparse.Namespace) -> int:
    """The --print-cost-projection subpath: pure stdout, never dispatches."""
    if args.execute:
        print("--print-cost-projection is a read-only helper and refuses "
              "--execute (it NEVER dispatches).")
        return 2
    if not args.run_id:
        print("--print-cost-projection requires --run-id (the run whose "
              "provenance sidecar holds the observed pod runtimes).")
        return 2
    if args.project_folds is None or args.rate_usd_per_hour is None:
        print("--print-cost-projection requires BOTH --project-folds and "
              "--rate-usd-per-hour (no baked-in GPU price).")
        return 2
    repo_root = resolve_repo_root(args.repo_root)
    strategy_artifacts = repo_root / "backtesting" / args.strategy / "artifacts"
    manifest_output = (Path(args.manifest_output) if args.manifest_output
                       else strategy_artifacts / RUN_NAMESPACE_ROOT
                       / args.run_id / CANONICAL_SERVING_MANIFEST)
    prov = read_prior_provenance(manifest_output)
    if prov is None:
        print(f"No readable provenance sidecar at "
              f"{manifest_output}.provenance.json — cannot project.")
        return 2
    try:
        proj = project_cost_from_provenance(
            prov, args.project_folds, args.rate_usd_per_hour)
    except ValueError as exc:
        print(f"Cost projection failed: {exc}")
        return 2
    print(f"COST PROJECTION (run_id={args.run_id})")
    print(f"  provenance          : {manifest_output}.provenance.json")
    print(f"  pods observed       : {proj['n_pods_observed']} "
          f"({', '.join(proj['observed_cutoffs'])})")
    print(f"  avg elapsed         : {proj['avg_elapsed_seconds']:.1f} s/fold")
    print(f"  rate                : ${proj['rate_usd_per_hour']:.4f}/h")
    print(f"  per-fold projected  : ${proj['per_fold_usd']:.4f}")
    print(f"  folds projected     : {proj['n_folds_projected']}")
    print(f"  PROJECTED TOTAL     : ${proj['projected_usd']:.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.print_cost_projection:
        return _print_cost_projection(args)
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
        # Best-effort resume preview: annotate folds already in the namespace.
        existing_preview: dict[str, dict[str, Any]] = {}
        try:
            preview_arts = (resolve_repo_root(args.repo_root) / "backtesting"
                            / plan.strategy / "artifacts")
            existing_preview = scan_existing_folds(
                preview_arts / plan.artifact_root,
                skip_calibrators=plan.skip_calibrators)
        except Exception:  # noqa: BLE001 — preview only, never blocks a plan
            existing_preview = {}
        for i, c in enumerate(plan.cutoffs):
            tag = ""
            if c in existing_preview:
                tag = ("  [EXISTS: would skip]"
                       if existing_preview[c]["resume_ok"]
                       else "  [EXISTS: INVALID — hard error on --execute]")
            print(f"    [{i + 1:02d}/{len(plan.cutoffs)}] cutoff={c}{tag}")
        if not args.execute:
            print("\n(plan-only; pass --execute to dispatch to Modal)")
        return 0

    repo_root = resolve_repo_root(args.repo_root)
    strategy_artifacts = repo_root / "backtesting" / plan.strategy / "artifacts"

    # ── Resume partition (model#82 P0 bounded dispatch) ─────────────────────
    # Decided from the run namespace BEFORE any cloud call: existing-and-valid
    # folds are skipped (never retrained/overwritten); an existing-but-invalid
    # selected fold is a hard error; only absent folds dispatch.
    part = partition_resume(plan, strategy_artifacts)
    prior_prov = read_prior_provenance(
        _manifest_output_path(plan, strategy_artifacts))
    if part["existing"] and not (prior_prov and prior_prov.get("recipe_id")):
        print(f"\nRESUME REFUSED: run {plan.run_id!r} already has "
              f"{len(part['existing'])} materialised fold(s) on disk but the "
              "provenance sidecar is missing or unreadable — the prior "
              "recipe_id can't be verified. Resuming into an unverifiable "
              "namespace risks silently mixing recipes (one run namespace = "
              "one recipe). Restore the sidecar or start a new --run-id.")
        return 2
    if (prior_prov and prior_prov.get("recipe_id")
            and prior_prov["recipe_id"] != plan.recipe_id):
        print(f"\nRESUME REFUSED: run {plan.run_id!r} was built with recipe_id "
              f"{prior_prov['recipe_id']} but this invocation computes "
              f"{plan.recipe_id}. One run namespace = one recipe — a mixed "
              "corpus would not be auditable.")
        return 2
    if part["invalid_selected"]:
        print("\nRESUME REFUSED: selected fold(s) already exist in run "
              f"namespace {plan.run_id!r} but FAIL integrity — refusing to "
              "retrain/overwrite an ambiguous fold. Inspect (and, if truly "
              "dead, manually quarantine) these before re-dispatching:")
        for c, reasons in sorted(part["invalid_selected"].items()):
            print(f"  - {c}: {', '.join(reasons)}")
        return 2
    to_dispatch = part["dispatch"]
    skipped = part["skipped"]
    if skipped:
        print(f"\nRESUME: {len(skipped)} fold(s) already exist and pass "
              f"integrity — SKIPPED (never retrained): {', '.join(skipped)}")
    plan.fold_requests = [r for r in plan.fold_requests
                          if r["cutoff_date"] in set(to_dispatch)]
    dispatch_meta: dict[str, Any] = {
        "app_id": None,
        "dispatched_cutoffs": list(to_dispatch),
        "skipped_existing_cutoffs": list(skipped),
        "timeout_seconds": int(args.timeout_seconds),
        "retries": int(args.retries),
        "note": args.dispatch_note,
    }

    if not to_dispatch:
        print("\nRESUME: every selected fold already exists and passes "
              "integrity — NOTHING to dispatch. Rebuilding the run's single "
              "manifest + provenance over the union (no cloud calls).")
        prior_modal = (prior_prov or {}).get("modal") or {}
        out = collect_and_write(
            plan, [], repo_root=repo_root,
            code_heads=prior_modal.get("code_git_heads") or {},
            staging={"volume_name": prior_modal.get("volume_name"),
                     "volume_commit_id": prior_modal.get("volume_commit_id"),
                     "data_digests": prior_modal.get("data_digests") or {}},
            existing_folds=part["existing"], dispatch_meta=dispatch_meta)
    else:
        readiness = modal_readiness()
        if not readiness["ready"]:
            print("\nMODAL NOT READY — cannot dispatch. Missing:")
            for m in readiness["missing"]:
                print(f"  - {m}")
            return 2

        dataset_path = repo_root / plan.dataset
        raw_label_path = repo_root / plan.raw_label_panel
        for pth in (dataset_path, raw_label_path):
            if not pth.exists():
                print(f"\nMissing required input panel: {pth}")
                return 2

        # AC7 fail-closed freshness/coverage gate (GOAL-5) — the SAME canonical
        # renquant-common contract the #74 driver runs, applied to the LOCAL
        # panel BEFORE staging it to the Volume. A Modal corpus runs each fold's
        # train_one_cutoff directly (never the driver's main()), so without this
        # pre-dispatch check a stale/truncated panel would silently short-train
        # every pod.
        rc = _assert_panel_fresh_or_report(plan, args, dataset_path)
        if rc != 0:
            return rc

        import tempfile  # noqa: PLC0415
        # ONE explicit pinned assembly (codex #76 blocker 1): the reviewed
        # checkout this executor runs from, or an explicit --code-root. NO
        # ~/git/github fallback and NO per-repo search — bundle_code fails
        # closed if the single root is missing any required repo, so a corpus
        # can't be sourced from an ambient/arbitrary checkout.
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
            results, dispatch_info = dispatch_folds(
                plan, timeout_s=args.timeout_seconds, retries=args.retries,
                volume_commit_id=staging.get("volume_commit_id"))
        dispatch_meta["app_id"] = dispatch_info.get("app_id")
        out = collect_and_write(
            plan, results, repo_root=repo_root, code_heads=code_heads,
            staging=staging, existing_folds=part["existing"],
            dispatch_meta=dispatch_meta)
    print(f"\nDONE: {out['n_folds']}/"
          f"{out['provenance_obj']['n_folds_requested']} folds "
          f"(run_id={plan.run_id})")
    print(f"  manifest   : {out['manifest']}  [QUARANTINED run namespace]")
    print(f"  provenance : {out['provenance']}")
    if out["promotion_ready"]:
        print("  status     : all folds materialised a valid model + sidecar + "
              "calibrator — eligible for a SEPARATE reviewed promotion to the "
              "serving manifest.")
        return 0
    # A run that is not promotion_ready is quarantined, NOT valid evidence: exit
    # nonzero so no caller mistakes it for a complete, promotable run. Missing
    # folds AND materialised-but-invalid payloads (bad/missing sidecar or
    # calibrator, or a diagnostic --skip-calibrators run) both land here.
    gate = out["provenance_obj"].get("promotion_gate", {})
    print(f"  status     : QUARANTINED — not promotable "
          f"({gate.get('n_folds_promotable', 0)}/"
          f"{out['provenance_obj']['n_folds_requested']} folds "
          f"passed the fail-closed gate).")
    reasons = gate.get("quarantine_reasons") or []
    if reasons:
        print(f"  reasons    : {', '.join(reasons)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
