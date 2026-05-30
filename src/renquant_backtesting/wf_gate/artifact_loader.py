"""Artifact / sidecar loading — first Phase 3 lift out of runner.py.

These three small functions were inline in ``runner.py``. Lifting them here
lets ``LoadArtifactTask`` (in ``pipelines.py``) call the implementation
directly instead of importing the runner module, which:

1. removes the Task → runner coupling that prevented unit-testing Tasks
   without loading the 2525-line runner;
2. gives the multi-repo a clean place to add new artifact kinds (e.g. a future
   ``hf_itransformer`` family) without editing the runner monolith;
3. matches §1c — single-responsibility helper.

Behaviour is byte-for-byte identical to the original ``_load_artifact_payload``
in ``runner.py`` (preserved for the Phase 4 byte-equivalent smoke). The runner
copy currently keeps its own inline copy of the function; Phase 5 flips it to
import from here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def artifact_sidecar_path(path: Path) -> Path | None:
    """Return optional metadata sidecar for non-JSON sequence artifacts.

    Probes three conventional sidecar filenames in priority order:
    ``foo.pt.metadata.json``, ``foo_metadata.json``, ``foo_summary.json``.
    """
    for candidate in (
        path.with_name(path.name + ".metadata.json"),
        path.with_name(path.stem + "_metadata.json"),
        path.with_name(path.stem + "_summary.json"),
    ):
        if candidate.exists():
            return candidate
    return None


def patchtst_params_from_contract(payload: dict) -> dict:
    """Project the model hyperparameters from a PatchTST sidecar's training_contract.

    Used to give the gate a normalised ``params`` field for sequence artifacts
    that matches the shape it expects for panel-LTR artifacts.
    """
    contract = payload.get("training_contract") or {}
    hparams = contract.get("hyperparameters") or {}
    keep = (
        "seq_len", "patch_length", "d_model", "n_heads", "n_layers",
        "lr", "weight_decay", "lr_scheduler", "warmup_ratio",
        "nll_loss_weight", "ranking_margin", "distributional_head",
        "film_regime_cond", "cross_stock_attn", "embargo_days",
    )
    return {k: hparams.get(k) for k in keep if k in hparams}


def load_artifact_payload(path: Path) -> dict:
    """Load a JSON artifact or merge a sequence-checkpoint sidecar.

    * ``.json`` path  → straight ``json.loads``.
    * ``.pt`` path    → load the metadata sidecar (if present); if either
      ``feature_cols`` or ``lookahead_days`` is still missing, lazy-import
      torch and pull them off the checkpoint directly.

    Always returns a dict containing at least ``kind`` (defaults to
    ``"hf_patchtst"`` for non-JSON without an explicit arch) and ``params``
    (projected from training_contract for sequences).
    """
    if path.suffix == ".json":
        return json.loads(path.read_text())
    sidecar = artifact_sidecar_path(path)
    payload: dict[str, Any] = json.loads(sidecar.read_text()) if sidecar else {}
    if path.suffix == ".pt" and (
        "feature_cols" not in payload or "lookahead_days" not in payload
    ):
        try:
            import torch  # noqa: PLC0415
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001
            ckpt = {}
        if isinstance(ckpt, dict):
            payload.setdefault("feature_cols", ckpt.get("feature_cols"))
            payload.setdefault("label_col", ckpt.get("label_col"))
            payload.setdefault("lookahead_days", ckpt.get("lookahead_days"))
            payload.setdefault("training_contract", ckpt.get("training_contract"))
    payload.setdefault("kind", payload.get("arch") or "hf_patchtst")
    payload.setdefault("params", patchtst_params_from_contract(payload))
    return payload


def write_artifact_payload(path: Path, payload: dict) -> Path:
    """Persist gate metadata without corrupting binary sequence checkpoints.

    For a ``.json`` artifact, writes back to the artifact itself.
    For a ``.pt`` or other binary checkpoint, writes to the **sidecar**
    (creating ``foo.pt.metadata.json`` if no sidecar already exists). Returns
    the actual path written.
    """
    out_path = path
    if path.suffix != ".json":
        sidecar = artifact_sidecar_path(path)
        out_path = sidecar or path.with_name(path.name + ".metadata.json")
    out_path.write_text(json.dumps(payload))
    return out_path
