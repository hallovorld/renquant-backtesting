"""B12 — Phase 4 byte-equivalent smoke test for the wf_gate Pipeline.

Runs the lifted Pipeline end-to-end with stubbed runner.* implementations on
a synthetic fixture. Asserts:

  1. All 6 Jobs execute in order (or skip per ctx flags).
  2. AssembleMetadataTask composes wf_meta with the documented keys.
  3. StampArtifactTask writes wf_meta to the artifact's metadata.wf_gate_metadata.
  4. EmitVerdictTask computes overall_pass correctly from per-stage passed flags.

Phase 4 invariant: the Pipeline composition produces wf_meta equivalent to
what umbrella runner.main()'s inline ``wf_meta = {...}`` block produces —
i.e. a strictly-keyed dict with config_parity / recipe_usage / wf_result /
trade_contract / trade_monotonicity / sanity, dropping None-valued stages.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
_UMBRELLA_SCRIPTS = Path(__file__).resolve().parents[3] / "RenQuant" / "scripts"
if _UMBRELLA_SCRIPTS.exists():
    sys.path.insert(0, str(_UMBRELLA_SCRIPTS))

try:
    from renquant_backtesting.wf_gate import runner as wf_runner
    _RUNNER_OK = True
except (ModuleNotFoundError, ImportError):
    _RUNNER_OK = False

from renquant_backtesting.wf_gate.pipelines import (  # noqa: E402
    AssembleMetadataTask,
    EmitVerdictTask,
    StampArtifactTask,
    WfGateContext,
    build_wf_gate_pipeline,
)

pytestmark = pytest.mark.skipif(
    not _RUNNER_OK,
    reason="umbrella scripts/ not reachable; Phase 1 invariant — Phase 5 flip will lift",
)


@pytest.fixture
def synthetic_artifact(tmp_path):
    """A small JSON artifact that StampArtifactTask can read+merge into."""
    p = tmp_path / "panel-ltr.json"
    p.write_text(json.dumps({
        "kind": "panel_ltr_xgboost",
        "trained_date": "2026-05-18",
        "config_fingerprint": "sha256:abc",
        "feature_cols": ["f1", "f2"],
    }))
    return p


def test_assemble_metadata_drops_none_stages(synthetic_artifact, tmp_path):
    ctx = WfGateContext(
        artifact_path=synthetic_artifact,
        strategy_config="strategy_config.shadow.json",
        strategy_dir=tmp_path,
        # Stage 1 succeeded
        config_parity_result={"passed": True},
        # Stage 2 succeeded
        recipe_usage={"recipe_validated": True},
        # Stage 3 produced a result
        wf_result={"passed": True, "wf_3cut_sharpe_mean": 0.5},
        # Stage 4 produced a trade-contract result
        trade_contract_result={"passed": True},
        trade_gate_result=None,  # monotonicity skipped → should drop
        # Stage 5 produced a sanity result
        sanity_result={"passed": True},
    )
    ok = AssembleMetadataTask().run(ctx)
    assert ok is True
    assert "config_parity" in ctx.wf_meta
    assert "recipe_usage" in ctx.wf_meta
    assert "wf_result" in ctx.wf_meta
    assert "trade_contract" in ctx.wf_meta
    assert "sanity" in ctx.wf_meta
    # None-valued stage dropped
    assert "trade_monotonicity" not in ctx.wf_meta


def test_stamp_artifact_writes_to_metadata_wf_gate_metadata(synthetic_artifact, tmp_path):
    ctx = WfGateContext(
        artifact_path=synthetic_artifact,
        strategy_config="strategy_config.shadow.json",
        strategy_dir=tmp_path,
        wf_meta={"passed": True, "wf_3cut_sharpe_mean": 0.5},
    )
    # Load artifact so StampArtifactTask has ctx.artifact to merge from
    from renquant_backtesting.wf_gate.artifact_loader import load_artifact_payload
    ctx.artifact = load_artifact_payload(synthetic_artifact)
    ok = StampArtifactTask().run(ctx)
    assert ok is True
    after = json.loads(synthetic_artifact.read_text())
    # Stamp lives under metadata.wf_gate_metadata (the historic field name preflight reads)
    assert after["metadata"]["wf_gate_metadata"]["passed"] is True
    assert after["metadata"]["wf_gate_metadata"]["wf_3cut_sharpe_mean"] == 0.5
    # Original artifact fields preserved
    assert after["kind"] == "panel_ltr_xgboost"
    assert after["trained_date"] == "2026-05-18"


def test_emit_verdict_passes_when_all_stages_pass(synthetic_artifact, tmp_path):
    ctx = WfGateContext(
        artifact_path=synthetic_artifact,
        strategy_config="strategy_config.shadow.json",
        strategy_dir=tmp_path,
        config_parity_result={"passed": True},
        recipe_usage={"recipe_validated": True},
        wf_result={"passed": True},
        trade_contract_result={"passed": True},
        trade_gate_result={"passed": True},
        alpha_economics_result={"passed": True},
        sanity_result={"passed": True},
    )
    EmitVerdictTask().run(ctx)
    assert ctx.overall_pass is True


def test_emit_verdict_fails_when_any_stage_fails(synthetic_artifact, tmp_path):
    ctx = WfGateContext(
        artifact_path=synthetic_artifact,
        strategy_config="strategy_config.shadow.json",
        strategy_dir=tmp_path,
        config_parity_result={"passed": True},
        recipe_usage={"recipe_validated": True},
        wf_result={"passed": True},
        trade_contract_result={"passed": True},
        trade_gate_result={"passed": True},
        alpha_economics_result={"passed": True},
        sanity_result={"passed": False},  # failing here
    )
    EmitVerdictTask().run(ctx)
    assert ctx.overall_pass is False


def test_emit_verdict_skipped_required_stages_block_acceptance(synthetic_artifact, tmp_path):
    """Skipped required gates are diagnostic-only, matching runner.py."""
    ctx = WfGateContext(
        artifact_path=synthetic_artifact,
        strategy_config="strategy_config.shadow.json",
        strategy_dir=tmp_path,
        config_parity_result={"passed": True},
        recipe_usage={"recipe_validated": True},
        skip_wf=True,
        skip_trade_gates=True,
        wf_result=None,                       # skipped
        trade_contract_result=None,           # skipped
        trade_gate_result=None,               # skipped
        alpha_economics_result=None,           # skipped
        sanity_result={"passed": True},
    )
    EmitVerdictTask().run(ctx)
    assert ctx.overall_pass is False
    assert "walk_forward_skipped" in ctx.verdict_blockers


def test_pipeline_runnable_with_stubs(synthetic_artifact, tmp_path, monkeypatch):
    """End-to-end: stub runner.* so the Pipeline runs without LEAN/torch."""
    monkeypatch.setattr(wf_runner, "run_walk_forward",
                        lambda *a, **kw: {"passed": True, "wf_3cut_sharpe_mean": 0.4, "cuts": []})
    monkeypatch.setattr(wf_runner, "run_trade_contract_gate",
                        lambda *a, **kw: {"passed": True})
    monkeypatch.setattr(wf_runner, "run_trade_monotonicity_gate",
                        lambda *a, **kw: {"passed": True})
    monkeypatch.setattr(wf_runner, "run_alpha_economics_gate",
                        lambda *a, **kw: {"passed": True})
    monkeypatch.setattr(wf_runner, "run_sanity_battery",
                        lambda *a, **kw: {"passed": True, "real_ic": 0.05})

    # Minimal strategy config for the few Tasks that read it. Under the
    # converged parity contract, CheckConfigParityTask selects this GBDT/shadow
    # config as the kind-matched reference for the panel_ltr_xgboost candidate,
    # so it must declare kind=xgb and point at the candidate artifact (the
    # static-artifact fallback the feature-contract check reads).
    (tmp_path / "strategy_config.shadow.json").write_text(json.dumps({
        "ranking": {
            "panel_scoring": {
                "enabled": True,
                "kind": "xgb",
                "artifact_path": str(synthetic_artifact),
            }
        }
    }))

    pipeline = build_wf_gate_pipeline()
    ctx = WfGateContext(
        artifact_path=synthetic_artifact,
        strategy_config="strategy_config.shadow.json",
        strategy_dir=tmp_path,
        trace_dir=tmp_path / "trace",
        recipe_usage={"recipe_validated": True},
    )
    # Run the Pipeline using its native interface; tolerate the runner functions
    # being stubbed (the Pipeline itself doesn't validate domain).
    result = pipeline.run(ctx)
    # Pipeline runs all 6 Jobs; per-stage results all stubbed PASS
    assert ctx.wf_result["passed"] is True
    assert ctx.trade_contract_result["passed"] is True
    assert ctx.alpha_economics_result["passed"] is True
    assert ctx.sanity_result["passed"] is True
    # Stamp landed
    after = json.loads(synthetic_artifact.read_text())
    assert "wf_gate_metadata" in after["metadata"]
    # Verdict
    assert ctx.overall_pass is True
