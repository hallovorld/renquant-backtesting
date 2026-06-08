"""P0 promotion-integrity guard (RFC #259): assert_artifact_gated.

Encodes the invariant "no scorer reaches production without a passing
wf_gate_metadata" — the hole the 2026-06-05 PatchTST config-edit promotion fell
through. Reuses the same _check_wf_gate contract promote() enforces.
"""
import json
from datetime import datetime, timezone

import pytest

from renquant_backtesting.forensics.model_acceptance import assert_artifact_gated


def _gated_meta() -> dict:
    """A fully-valid wf_gate_metadata that passes _check_wf_gate."""
    return {
        "wf_gate_metadata": {
            "passed": True,
            "wf_3cut_sharpe_mean": 1.0,
            "run_at": datetime.now(timezone.utc).isoformat(),
            "trade_monotonicity": {
                "passed": True,
                "allow_pass_open": False,
                "regimes": [{"eligible": True, "passed": True}],
            },
            "alpha_economics": {"passed": True},
            "sanity_regime_ic": {"passed": True},
        }
    }


def test_gated_json_artifact_passes(tmp_path):
    p = tmp_path / "panel-ltr.alpha158_fund.json"
    p.write_text(json.dumps(_gated_meta()))
    wf = assert_artifact_gated(p)
    assert wf["passed"] is True


def test_missing_metadata_raises(tmp_path):
    p = tmp_path / "panel-ltr.json"
    p.write_text(json.dumps({"kind": "xgb"}))  # no wf_gate_metadata
    with pytest.raises(ValueError, match="missing wf_gate_metadata"):
        assert_artifact_gated(p)


def test_failed_gate_raises(tmp_path):
    meta = _gated_meta()
    meta["wf_gate_metadata"]["passed"] = False
    p = tmp_path / "panel-ltr.json"
    p.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="passed=False"):
        assert_artifact_gated(p)


def test_sequence_pt_resolves_sidecar(tmp_path):
    # .pt checkpoints carry metadata in <artifact>.metadata.json
    pt = tmp_path / "hf_patchtst_all_seed44_model.pt"
    pt.write_bytes(b"\x80\x02fake-torch")  # not JSON — must use the sidecar
    (tmp_path / "hf_patchtst_all_seed44_model.pt.metadata.json").write_text(
        json.dumps(_gated_meta())
    )
    wf = assert_artifact_gated(pt)
    assert wf["passed"] is True


def test_ungated_pt_sidecar_raises(tmp_path):
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"fake")
    (tmp_path / "model.pt.metadata.json").write_text(json.dumps({"kind": "hf_patchtst"}))
    with pytest.raises(ValueError, match="missing wf_gate_metadata"):
        assert_artifact_gated(pt)


def test_pt_without_sidecar_raises_clear_value_error(tmp_path):
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"\x80\x02fake-torch")
    with pytest.raises(ValueError, match="sidecar metadata not found"):
        assert_artifact_gated(pt)


def test_missing_artifact_raises(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        assert_artifact_gated(tmp_path / "nope.json")
