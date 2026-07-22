"""Modal app definition for the walk-forward PatchTST re-score worker.

The ``@app.function``-decorated worker lives here at module scope so Modal can
reference it by name instead of pickling it.

**Why this is a STANDALONE top-level module (not under ``renquant_backtesting``):**
Modal's container entrypoint imports the worker's *defining module* to locate the
function (``importlib.import_module(function_def.module_name)``) BEFORE any
function body runs. If the worker lived under the ``renquant_backtesting``
package, that import would run ``renquant_backtesting/__init__.py`` →
``from .simulation import …`` → ``from renquant_common import …`` at container-load
time — and the pinned sibling assembly is only on the Volume (put on ``sys.path``
*inside* the function body), so the load fails with ``ModuleNotFoundError:
renquant_common`` before the body ever executes. Homing the app in a top-level
module whose import surface is exactly ``os + modal`` avoids that (same invariant
the orchestrator ``cloud/modal_app.py`` relies on — its package ``__init__`` is
light). Everything heavy — the driver, torch, the model repo — is imported at
RUNTIME inside the container from the Volume-staged pinned bundle.

The image literals below re-declare
:data:`renquant_backtesting.wf_gate.modal.executor.IMAGE_SPEC` (the single source
of truth; a unit test asserts they stay in lockstep).

Repo boundary: the per-fold unit of work is the reviewed local driver
``renquant_backtesting.wf_gate.train_walkforward_patchtst.train_one_cutoff``
(PR #74), which itself invokes ``python -m renquant_model_patchtst.hf_trainer``
/ ``renquant_model_patchtst.fit_calibrator``. Model-training internals therefore
stay in renquant-model; this worker only runs the driver on a GPU pod.
"""
from __future__ import annotations

import os

import modal

APP_NAME = "renquant-wf-patchtst"
VOLUME_NAME = "renquant-wf-patchtst-data"

# gpu/timeout/retries are baked into @app.function at DECORATION (import) time —
# Modal has no per-call override for a module-scope function. The executor sets
# these env vars before importing this module (its only import site) so the
# caller's requested values are what gets baked in.
DEFAULT_GPU = "T4"
DEFAULT_TIMEOUT_SECONDS = 3600
DEFAULT_RETRIES = 1
WORKER_GPU = os.environ.get("RENQUANT_WF_MODAL_GPU", DEFAULT_GPU)
WORKER_TIMEOUT_SECONDS = int(
    os.environ.get("RENQUANT_WF_MODAL_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
)
WORKER_RETRIES = int(
    os.environ.get("RENQUANT_WF_MODAL_RETRIES", DEFAULT_RETRIES)
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# These literals MUST stay in lockstep with executor.IMAGE_SPEC (asserted by
# tests/test_modal_wf_patchtst.py). Kept as literals here — rather than importing
# executor.IMAGE_SPEC — so the container-side import surface stays os + modal.
_BASE_IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.2",
        "transformers>=4.40",
        "accelerate>=0.26",
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
    )
)

_WORKER_FUNCTION_KWARGS: dict = dict(
    image=_BASE_IMAGE,
    volumes={"/data": data_volume},
    timeout=WORKER_TIMEOUT_SECONDS,
    retries=WORKER_RETRIES,
)
# "cpu" (case-insensitive) means: request no GPU. Any other value is passed to
# Modal as the GPU type; an unsupported type fails loudly at dispatch.
if WORKER_GPU and str(WORKER_GPU).lower() != "cpu":
    _WORKER_FUNCTION_KWARGS["gpu"] = WORKER_GPU


@app.function(**_WORKER_FUNCTION_KWARGS)
def train_fold_remote(request_json: str) -> str:
    """Train ONE walk-forward PatchTST fold on a Modal pod.

    Runs the reviewed #74 driver's ``train_one_cutoff`` (which shells out to the
    renquant-model trainer + calibrator) against the Volume-staged panels, then
    returns the fold's ``.pt`` (gzip+base64), calibrator JSON, metadata sidecar,
    the manifest-entry dict, and per-pod provenance.
    """
    import argparse
    import base64
    import gzip
    import json
    import sys
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    t0 = time.time()
    request = json.loads(request_json)
    cutoff = request["cutoff_date"]
    recipe = request["recipe"]
    repo_root = request.get("container_repo_root", "/data")
    bundle_root = request.get("container_bundle_root", "/data/app/repos")

    # Point the driver + subprocess at the umbrella-style layout on the Volume.
    # RENQUANT_SUBREPO_ROOT is the driver's fail-closed assembly injection point
    # (PR #74 0744d14): it pins every training import to the staged bundle at
    # ``/data/app/repos/<repo>/src`` instead of any ambient checkout.
    os.environ["RENQUANT_REPO_ROOT"] = repo_root
    os.environ["RENQUANT_SUBREPO_ROOT"] = bundle_root
    os.environ.setdefault("RENQUANT_DATA_ROOT", repo_root)
    for repo in (
        "renquant-common", "renquant-base-data", "renquant-artifacts",
        "renquant-model", "renquant-pipeline", "renquant-strategy-104",
        "renquant-backtesting",
    ):
        p = f"{bundle_root}/{repo}/src"
        if p not in sys.path:
            sys.path.insert(0, p)

    def _fail(stage: str, err: str) -> str:
        return json.dumps({
            "ok": False, "cutoff_date": cutoff, "stage": stage,
            "error": err, "worker_id": os.environ.get("MODAL_TASK_ID", "unknown"),
            "code_image_id": os.environ.get("MODAL_IMAGE_ID", "unknown"),
            "elapsed_seconds": time.time() - t0,
        })

    try:
        import torch  # noqa: PLC0415
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        device = str(recipe.get("device", "cpu"))

    try:
        from renquant_backtesting.wf_gate import train_walkforward_patchtst as drv
    except Exception as exc:  # noqa: BLE001
        return _fail("import_driver", repr(exc))

    args = argparse.Namespace(
        repo_root=repo_root,
        strategy=recipe.get("strategy", "renquant_104"),
        artifact_root=recipe.get("artifact_root", "walkforward_patchtst"),
        dataset=recipe["dataset"],
        raw_label_panel=recipe["raw_label_panel"],
        label=recipe["label"],
        seed=int(recipe["seed"]),
        epochs=int(recipe["epochs"]),
        seq_len=int(recipe["seq_len"]),
        patch_length=int(recipe["patch_length"]),
        d_model=int(recipe["d_model"]),
        n_heads=int(recipe["n_heads"]),
        n_layers=int(recipe["n_layers"]),
        lr=float(recipe["lr"]),
        weight_decay=float(recipe["weight_decay"]),
        device=device,
        strategy_config=None,
        film_regime_cond=bool(recipe.get("film_regime_cond", False)),
        cross_stock_attn=bool(recipe.get("cross_stock_attn", False)),
        reuse_existing=False,
        skip_calibrators=bool(recipe.get("skip_calibrators", False)),
        calibrator_batch_size=int(recipe.get("calibrator_batch_size", 512)),
        calibrator_method=recipe.get("calibrator_method", "platt"),
        calibrator_min_rows=int(recipe.get("calibrator_min_rows", 1000)),
    )

    import pandas as pd  # noqa: PLC0415
    cutoff_ts = pd.Timestamp(cutoff)
    try:
        _, entry, err = drv.train_one_cutoff(args, cutoff_ts)
    except Exception as exc:  # noqa: BLE001
        return _fail("train_one_cutoff", repr(exc))
    if entry is None:
        return _fail("train_one_cutoff", err or "unknown fold failure")

    out_dir = drv.artifact_dir(args, cutoff_ts)
    model_path = drv.model_path_for(out_dir, int(args.seed))
    sidecar_path = drv.sidecar_path_for(model_path)
    cal_path = drv.calibrator_path_for(model_path)

    artifacts: dict = {}
    if model_path.exists():
        artifacts["model_pt_b64gz"] = base64.b64encode(
            gzip.compress(model_path.read_bytes())).decode()
    if sidecar_path.exists():
        artifacts["sidecar_json"] = sidecar_path.read_text()
    if cal_path.exists():
        artifacts["calibrator_json"] = cal_path.read_text()

    def _iso(v):
        return v.isoformat() if hasattr(v, "isoformat") else (str(v) if v else None)

    entry_dict = {
        "cutoff_date": _iso(entry.cutoff_date),
        "trained_date": _iso(entry.trained_date),
        "artifact_uri": str(entry.artifact_uri),
        "lookahead_days": int(getattr(entry, "lookahead_days", 60)),
        "calibrator_uri": (str(entry.calibrator_uri)
                           if getattr(entry, "calibrator_uri", None) else None),
        "effective_train_cutoff_date": _iso(
            getattr(entry, "effective_train_cutoff_date", None)),
    }

    result = {
        "ok": True,
        "cutoff_date": cutoff,
        "recipe_id": request.get("recipe_id"),
        "image_spec_sha256": request.get("image_spec_sha256"),
        "volume_commit_id": request.get("volume_commit_id"),
        "worker_id": os.environ.get("MODAL_TASK_ID", "unknown"),
        "code_image_id": os.environ.get("MODAL_IMAGE_ID", "unknown"),
        "device": device,
        "started_at": datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - t0,
        "entry": entry_dict,
        "artifacts": artifacts,
    }
    import hashlib  # noqa: PLC0415
    canonical = json.dumps(
        {k: v for k, v in result.items()
         if k not in ("artifacts", "result_checksum")},
        sort_keys=True, default=str)
    result["result_checksum"] = "sha256:" + hashlib.sha256(
        canonical.encode()).hexdigest()[:16]
    # Persist the fold to the Volume too, so a crashed collector can recover it.
    data_volume.commit()
    return json.dumps(result, default=str)
