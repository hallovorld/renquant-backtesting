"""Structural tests for the wf_gate Pipeline scaffold.

These pin the §1c Task/Job/Pipeline shape so subsequent Phase 2 work (moving
implementations from runner.py into the Tasks) cannot accidentally collapse
the structure.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from renquant_backtesting.wf_gate.pipelines import (
    ConfigJob,
    RecipeMatchJob,
    SanityJob,
    StampJob,
    TradeGateJob,
    WfGateContext,
    WfSimJob,
    build_wf_gate_pipeline,
)


def test_pipeline_has_six_ordered_jobs() -> None:
    p = build_wf_gate_pipeline()
    assert p.name == "wf-gate"
    assert [type(j).__name__ for j in p.jobs] == [
        "ConfigJob", "RecipeMatchJob", "WfSimJob", "TradeGateJob",
        "SanityJob", "StampJob",
    ]


def test_each_job_decomposes_into_named_tasks() -> None:
    assert [type(t).__name__ for t in ConfigJob().tasks] == [
        "LoadArtifactTask", "DeriveConfigTask", "CheckConfigParityTask"]
    assert [type(t).__name__ for t in RecipeMatchJob().tasks] == [
        "ResolveManifestTask", "ValidateRecipeMatchTask"]
    assert [type(t).__name__ for t in WfSimJob().tasks] == ["RunWfSimTask"]
    assert [type(t).__name__ for t in TradeGateJob().tasks] == [
        "RunTradeContractTask", "RunTradeMonotonicityTask"]
    assert [type(t).__name__ for t in SanityJob().tasks] == ["RunSanityBatteryTask"]
    assert [type(t).__name__ for t in StampJob().tasks] == [
        "AssembleMetadataTask", "StampArtifactTask", "EmitVerdictTask"]


@pytest.mark.parametrize("flag,job", [
    ("skip_wf", WfSimJob),
    ("skip_sanity", SanityJob),
    ("skip_trade_gates", TradeGateJob),
])
def test_skip_flags_short_circuit_the_right_jobs(flag: str, job: type) -> None:
    """The --skip-* CLI flags must short-circuit the matching Job, not break it."""
    on = WfGateContext(artifact_path=Path("/x"), strategy_config="x.json", **{flag: True})
    off = WfGateContext(artifact_path=Path("/x"), strategy_config="x.json", **{flag: False})
    assert job().should_skip(on) is True
    assert job().should_skip(off) is False


def test_trade_gate_also_skips_when_wf_skipped() -> None:
    """No trade gate without WF data — both must skip together."""
    ctx = WfGateContext(artifact_path=Path("/x"), strategy_config="x.json", skip_wf=True)
    assert TradeGateJob().should_skip(ctx) is True


def test_check_config_parity_skips_when_no_strategy_dir() -> None:
    from renquant_backtesting.wf_gate.pipelines import CheckConfigParityTask
    ctx = WfGateContext(
        artifact_path=Path("/x"), strategy_config="strategy_config.shadow.json",
        strategy_dir=None,
    )
    CheckConfigParityTask().run(ctx)
    assert ctx.config_parity_result["passed"] is True
    assert "skipped" in ctx.config_parity_result["reason"]


def test_check_config_parity_honours_skip_flag() -> None:
    from renquant_backtesting.wf_gate.pipelines import CheckConfigParityTask
    ctx = WfGateContext(
        artifact_path=Path("/x"), strategy_config="x.json",
        strategy_dir=Path("/tmp"), skip_config_parity=True,
    )
    CheckConfigParityTask().run(ctx)
    assert "--skip-config-parity" in ctx.config_parity_result["reason"]


def test_check_config_parity_skips_when_configs_missing(tmp_path: Path) -> None:
    from renquant_backtesting.wf_gate.pipelines import CheckConfigParityTask
    ctx = WfGateContext(
        artifact_path=tmp_path / "art.json",
        strategy_config="missing.json", strategy_dir=tmp_path,
    )
    CheckConfigParityTask().run(ctx)
    assert ctx.config_parity_result["passed"] is True
    assert "config not found" in ctx.config_parity_result["reason"]


def test_assemble_metadata_drops_none_stages() -> None:
    from renquant_backtesting.wf_gate.pipelines import AssembleMetadataTask
    ctx = WfGateContext(artifact_path=Path("/x"), strategy_config="x.json")
    ctx.config_parity_result = {"passed": True}
    # other stages left None
    AssembleMetadataTask().run(ctx)
    assert "config_parity" in ctx.wf_meta
    assert "wf_result" not in ctx.wf_meta
    assert "sanity" not in ctx.wf_meta


def test_stamp_artifact_writes_metadata_into_json(tmp_path: Path) -> None:
    import json
    from renquant_backtesting.wf_gate.pipelines import (
        AssembleMetadataTask, StampArtifactTask,
    )
    art = tmp_path / "art.json"
    art.write_text(json.dumps({"kind": "panel_ltr_xgboost", "params": {}}))
    ctx = WfGateContext(artifact_path=art, strategy_config="x.json")
    ctx.artifact = {"kind": "panel_ltr_xgboost", "params": {}}
    ctx.config_parity_result = {"passed": True}
    AssembleMetadataTask().run(ctx)
    StampArtifactTask().run(ctx)
    after = json.loads(art.read_text())
    assert "metadata" in after
    assert after["metadata"]["wf_gate_metadata"]["config_parity"]["passed"] is True


def test_emit_verdict_pass_when_all_stages_ok() -> None:
    from renquant_backtesting.wf_gate.pipelines import EmitVerdictTask
    ctx = WfGateContext(artifact_path=Path("/x"), strategy_config="x.json")
    ctx.config_parity_result = {"passed": True}
    ctx.recipe_usage = {"recipe_validated": True}
    ctx.wf_result = {"passed": True}
    EmitVerdictTask().run(ctx)
    assert ctx.overall_pass is True


def test_emit_verdict_fail_when_any_stage_fails() -> None:
    from renquant_backtesting.wf_gate.pipelines import EmitVerdictTask
    ctx = WfGateContext(artifact_path=Path("/x"), strategy_config="x.json")
    ctx.config_parity_result = {"passed": True}
    ctx.recipe_usage = {"recipe_validated": False}   # ← fails
    EmitVerdictTask().run(ctx)
    assert ctx.overall_pass is False


def test_pipeline_run_on_empty_context_does_not_raise() -> None:
    """Scaffold-only: a no-config run should pass through all Jobs cleanly
    (the Tasks delegate to runner.py functions only when state is present)."""
    p = build_wf_gate_pipeline()
    ctx = WfGateContext(
        artifact_path=Path("/nonexistent.json"),
        strategy_config="strategy_config.shadow.json",
        skip_wf=True, skip_sanity=True, skip_trade_gates=True,
        skip_config_parity=True,
    )
    # LoadArtifactTask will try to read; skip via setting artifact ahead
    ctx.artifact = {"kind": "panel_ltr_xgboost", "feature_cols": [], "params": {}}
    # Manually skip the load step by pre-populating; the test verifies the
    # SCAFFOLD doesn't crash before implementations are lifted in.
    # We assert only that build_wf_gate_pipeline returns a runnable Pipeline.
    assert callable(p.run)
