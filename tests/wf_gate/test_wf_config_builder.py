"""Tests for ``build_wf_config_from_prod`` scorer-kind/artifact consistency.

Regression coverage for the active-path WF-gate config-parity bug: when the
prod config read is the PatchTST primary (``kind=hf_patchtst``) but the
candidate under evaluation is a GBDT ``panel-ltr.json``, the derived WF eval
config must carry the kind implied by the candidate artifact (``xgb``), not the
prod kind. Otherwise the prod/WF parity guard
(``wf_config_parity._scorer_kind_artifact_issues``) correctly fails with
"PatchTST scorer kind should not point at a non-PatchTST JSON artifact".
"""
from __future__ import annotations

import json
from pathlib import Path

from renquant_backtesting.wf_gate.wf_config_builder import build_wf_config_from_prod
from renquant_backtesting.wf_gate.wf_config_parity import evaluate_wf_config_parity


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _prod_config(*, kind: str, artifact_path: str) -> dict:
    return {
        "ranking": {
            "panel_scoring": {
                "enabled": True,
                "kind": kind,
                "artifact_path": artifact_path,
                "buy_floor": "adaptive_mean_std",
            },
        },
    }


def _write_manifest(path: Path, artifact_uri: str) -> None:
    _write_json(path, {"retrains": [{"artifact_uri": artifact_uri}]})


def test_gbdt_candidate_sets_xgb_kind_despite_patchtst_prod(tmp_path: Path) -> None:
    """PatchTST prod + GBDT ``panel-ltr.json`` candidate → derived kind == xgb."""
    gbdt_artifact = tmp_path / "artifacts" / "sim" / "panel-ltr.json"
    _write_json(gbdt_artifact, {"kind": "panel_ltr_xgboost", "feature_cols": ["a", "b"]})
    manifest = tmp_path / "artifacts" / "wf" / "manifest.json"
    _write_manifest(manifest, str(gbdt_artifact.relative_to(tmp_path)))

    prod_cfg = _prod_config(kind="hf_patchtst", artifact_path="artifacts/prod/model.pt")
    base_cfg = {"walkforward": {"manifest_path": str(manifest.relative_to(tmp_path))}}

    derived = build_wf_config_from_prod(
        prod_cfg,
        manifest_path=str(manifest.relative_to(tmp_path)),
        base_wf_config=base_cfg,
        strategy_dir=tmp_path,
    )

    panel = derived["ranking"]["panel_scoring"]
    assert panel["kind"] == "xgb"
    assert panel["artifact_path"].endswith("panel-ltr.json")


def test_gbdt_candidate_derived_config_passes_parity(tmp_path: Path) -> None:
    """End-to-end: derived config from a PatchTST prod + GBDT candidate passes."""
    gbdt_artifact = tmp_path / "artifacts" / "sim" / "panel-ltr.json"
    _write_json(gbdt_artifact, {"kind": "panel_ltr_xgboost", "feature_cols": ["a", "b"]})
    manifest = tmp_path / "artifacts" / "wf" / "manifest.json"
    _write_manifest(manifest, str(gbdt_artifact.relative_to(tmp_path)))

    # Prod config read is the PatchTST primary, but the GBDT candidate is what
    # we evaluate. evaluate_wf_config_parity compares against candidate_artifact,
    # so the feature contract matches the GBDT manifest artifact.
    prod_cfg = _prod_config(kind="hf_patchtst", artifact_path="artifacts/prod/model.pt")
    base_cfg = {"walkforward": {"manifest_path": str(manifest.relative_to(tmp_path))}}

    derived = build_wf_config_from_prod(
        prod_cfg,
        manifest_path=str(manifest.relative_to(tmp_path)),
        base_wf_config=base_cfg,
        strategy_dir=tmp_path,
    )

    prod_path = tmp_path / "strategy_config.json"
    wf_path = tmp_path / "wf_config.json"
    # Use the GBDT candidate as the "prod" config for parity (mirrors evaluating
    # a GBDT candidate); the bug fired regardless of which prod kind was read.
    _write_json(prod_path, _prod_config(kind="xgb", artifact_path=str(gbdt_artifact)))
    wf_path.write_text(json.dumps(derived), encoding="utf-8")

    result = evaluate_wf_config_parity(
        prod_path,
        wf_path,
        candidate_artifact=gbdt_artifact,
        strategy_dir=tmp_path,
    )

    assert result["passed"] is True, result["issues"]


def test_patchtst_candidate_keeps_patchtst_kind(tmp_path: Path) -> None:
    """PatchTST ``.pt`` candidate keeps ``kind=hf_patchtst`` and passes parity."""
    pt_artifact = tmp_path / "artifacts" / "sim" / "patchtst_model.pt"
    pt_artifact.parent.mkdir(parents=True, exist_ok=True)
    pt_artifact.write_bytes(b"not-json")
    sidecar = pt_artifact.with_suffix(".pt.json")
    _write_json(sidecar, {"kind": "hf_patchtst", "feature_cols": ["a", "b"]})
    manifest = tmp_path / "artifacts" / "wf" / "manifest.json"
    _write_manifest(manifest, str(pt_artifact.relative_to(tmp_path)))

    prod_cfg = _prod_config(kind="hf_patchtst", artifact_path="artifacts/prod/model.pt")
    base_cfg = {"walkforward": {"manifest_path": str(manifest.relative_to(tmp_path))}}

    derived = build_wf_config_from_prod(
        prod_cfg,
        manifest_path=str(manifest.relative_to(tmp_path)),
        base_wf_config=base_cfg,
        strategy_dir=tmp_path,
    )

    panel = derived["ranking"]["panel_scoring"]
    assert panel["kind"] == "hf_patchtst"
    assert panel["artifact_path"].endswith("patchtst_model.pt")


def test_explicit_candidate_kind_overrides_inference(tmp_path: Path) -> None:
    """An explicit ``candidate_kind`` is normalized and wins over inference."""
    gbdt_artifact = tmp_path / "artifacts" / "sim" / "panel-ltr.json"
    _write_json(gbdt_artifact, {"kind": "panel_ltr_xgboost", "feature_cols": ["a"]})
    manifest = tmp_path / "artifacts" / "wf" / "manifest.json"
    _write_manifest(manifest, str(gbdt_artifact.relative_to(tmp_path)))

    prod_cfg = _prod_config(kind="hf_patchtst", artifact_path="artifacts/prod/model.pt")
    base_cfg = {"walkforward": {"manifest_path": str(manifest.relative_to(tmp_path))}}

    derived = build_wf_config_from_prod(
        prod_cfg,
        manifest_path=str(manifest.relative_to(tmp_path)),
        base_wf_config=base_cfg,
        strategy_dir=tmp_path,
        candidate_kind="panel_ltr_xgboost",
    )

    assert derived["ranking"]["panel_scoring"]["kind"] == "xgb"
