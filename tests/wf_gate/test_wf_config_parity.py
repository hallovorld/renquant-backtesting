from __future__ import annotations

import json
from pathlib import Path

from renquant_backtesting.wf_gate.wf_config_parity import evaluate_wf_config_parity


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _config(*, kind: str, artifact_path: str) -> dict:
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


def test_xgb_kind_cannot_point_at_torch_checkpoint(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    wf_checkpoint = tmp_path / "artifacts" / "sim" / "model.pt"
    _write_json(candidate, {"kind": "panel_ltr_xgboost", "feature_cols": ["a"]})
    wf_checkpoint.parent.mkdir(parents=True)
    wf_checkpoint.write_bytes(b"not-json")

    prod_cfg = tmp_path / "strategy_config.json"
    wf_cfg = tmp_path / "strategy_config.shadow.json"
    _write_json(prod_cfg, _config(kind="xgb", artifact_path=str(candidate)))
    _write_json(wf_cfg, _config(kind="xgb", artifact_path=str(wf_checkpoint)))

    result = evaluate_wf_config_parity(
        prod_cfg,
        wf_cfg,
        candidate_artifact=candidate,
        strategy_dir=tmp_path,
    )

    assert result["passed"] is False
    assert any(
        issue.get("path") == "wf_config.ranking.panel_scoring.artifact_path"
        and "PyTorch checkpoint" in issue.get("reason", "")
        for issue in result["issues"]
    )


def test_xgb_kind_accepts_json_artifact_when_feature_contract_matches(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    wf_artifact = tmp_path / "wf.json"
    _write_json(candidate, {"kind": "panel_ltr_xgboost", "feature_cols": ["a"]})
    _write_json(wf_artifact, {"kind": "panel_ltr_xgboost", "feature_cols": ["a"]})

    prod_cfg = tmp_path / "strategy_config.json"
    wf_cfg = tmp_path / "strategy_config.shadow.json"
    _write_json(prod_cfg, _config(kind="xgb", artifact_path=str(candidate)))
    _write_json(wf_cfg, _config(kind="xgb", artifact_path=str(wf_artifact)))

    result = evaluate_wf_config_parity(
        prod_cfg,
        wf_cfg,
        candidate_artifact=candidate,
        strategy_dir=tmp_path,
    )

    assert result["passed"] is True
