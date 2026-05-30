"""Tests for the lifted artifact_loader — Phase 3a verification.

Pins the behaviour that was previously in runner._load_artifact_payload:
- JSON path: direct read
- .pt path: prefer sidecar; fall back to ckpt if fields missing
- defaults: kind, params
"""
from __future__ import annotations

import json
from pathlib import Path

from renquant_backtesting.wf_gate.artifact_loader import (
    artifact_sidecar_path,
    load_artifact_payload,
    patchtst_params_from_contract,
)


def test_artifact_sidecar_path_finds_metadata_first(tmp_path: Path) -> None:
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"x")
    side = tmp_path / "model.pt.metadata.json"
    side.write_text("{}")
    assert artifact_sidecar_path(pt) == side


def test_artifact_sidecar_path_falls_through_to_summary(tmp_path: Path) -> None:
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"x")
    summary = tmp_path / "model_summary.json"
    summary.write_text("{}")
    assert artifact_sidecar_path(pt) == summary


def test_artifact_sidecar_path_returns_none_when_absent(tmp_path: Path) -> None:
    assert artifact_sidecar_path(tmp_path / "nothing.pt") is None


def test_load_artifact_payload_json_returns_parsed(tmp_path: Path) -> None:
    f = tmp_path / "art.json"
    f.write_text(json.dumps({"kind": "panel_ltr_xgboost", "params": {"eta": 0.05}}))
    out = load_artifact_payload(f)
    assert out["kind"] == "panel_ltr_xgboost"
    assert out["params"]["eta"] == 0.05


def test_load_artifact_payload_pt_uses_sidecar(tmp_path: Path) -> None:
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"\x00")
    side = tmp_path / "model.pt.metadata.json"
    side.write_text(json.dumps({
        "kind": "hf_patchtst",
        "feature_cols": ["a", "b"],
        "lookahead_days": 60,
        "training_contract": {"hyperparameters": {"seq_len": 24, "lr": 1e-4}},
    }))
    out = load_artifact_payload(pt)
    assert out["kind"] == "hf_patchtst"
    assert out["feature_cols"] == ["a", "b"]
    # patchtst_params_from_contract projected hparams
    assert out["params"]["seq_len"] == 24
    assert out["params"]["lr"] == 1e-4


def test_load_artifact_payload_defaults_kind_to_hf_patchtst_for_pt(tmp_path: Path) -> None:
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"\x00")
    side = tmp_path / "model.pt.metadata.json"
    side.write_text(json.dumps({"feature_cols": ["x"], "lookahead_days": 60}))
    out = load_artifact_payload(pt)
    assert out["kind"] == "hf_patchtst"


def test_patchtst_params_filter_keeps_expected_keys() -> None:
    payload = {"training_contract": {"hyperparameters": {
        "seq_len": 24, "lr": 0.0001, "nonsense_param": 999,
        "cross_stock_attn": True, "weight_decay": 0.3,
    }}}
    p = patchtst_params_from_contract(payload)
    assert "seq_len" in p
    assert "lr" in p
    assert "weight_decay" in p
    assert "cross_stock_attn" in p
    assert "nonsense_param" not in p


# ─── write_artifact_payload (Phase 3h) ──────────────────────────────────────

from renquant_backtesting.wf_gate.artifact_loader import write_artifact_payload


def test_write_artifact_payload_json_overwrites_in_place(tmp_path: Path) -> None:
    p = tmp_path / "art.json"
    p.write_text(json.dumps({"kind": "x"}))
    out = write_artifact_payload(p, {"kind": "x", "metadata": {"wf_gate_metadata": {"passed": True}}})
    assert out == p
    written = json.loads(p.read_text())
    assert written["metadata"]["wf_gate_metadata"]["passed"] is True


def test_write_artifact_payload_pt_creates_metadata_sidecar(tmp_path: Path) -> None:
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"\x00")  # binary, do NOT corrupt
    out = write_artifact_payload(pt, {"kind": "hf_patchtst", "test": 1})
    assert out == tmp_path / "model.pt.metadata.json"
    assert pt.read_bytes() == b"\x00"   # binary preserved
    side = json.loads(out.read_text())
    assert side["test"] == 1


def test_write_artifact_payload_pt_uses_existing_sidecar(tmp_path: Path) -> None:
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"\x00")
    existing = tmp_path / "model_summary.json"   # the third probe in priority
    existing.write_text(json.dumps({"kind": "hf_patchtst"}))
    # metadata sidecar takes priority (first probe), so create one
    metadata = tmp_path / "model.pt.metadata.json"
    metadata.write_text("{}")
    out = write_artifact_payload(pt, {"kind": "hf_patchtst", "v": 2})
    assert out == metadata   # priority order respected
