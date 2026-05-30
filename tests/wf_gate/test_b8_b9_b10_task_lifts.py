"""Phase 3e-g (B8/B9/B10) — RunWfSim / TradeContract / TradeMonotonicity /
SanityBattery Tasks now delegate to runner.* instead of returning True stub.

Tests assert each Task's contract:
  * RunWfSimTask                 → ctx.wf_result populated
  * RunTradeContractTask         → ctx.trade_contract_result populated
  * RunTradeMonotonicityTask     → ctx.trade_gate_result populated
  * RunSanityBatteryTask         → ctx.sanity_result populated

Uses monkeypatch to swap the runner.* implementations with stubs (so the
Tasks can run without LEAN/torch/strategy-dir setup). The DELEGATION shape
is the contract; the runner functions themselves are tested in umbrella
``tests/test_wf_gate_cli_contract.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Subrepo src
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
# Umbrella scripts/ — runner.py is a byte-equivalent copy of umbrella
# scripts/run_wf_gate.py and pulls qp_contracts + others from umbrella scripts/.
# Phase 1 invariant: subrepo runner is byte-equivalent, NOT independently
# importable. Tests therefore put umbrella scripts/ on sys.path so the import
# resolves — and soft-skip if the umbrella isn't reachable.
_UMBRELLA_SCRIPTS = Path(__file__).resolve().parents[3] / "RenQuant" / "scripts"
if _UMBRELLA_SCRIPTS.exists():
    sys.path.insert(0, str(_UMBRELLA_SCRIPTS))

from renquant_backtesting.wf_gate.pipelines import (  # noqa: E402
    RunAlphaEconomicsTask,
    RunSanityBatteryTask,
    RunTradeContractTask,
    RunTradeMonotonicityTask,
    RunWfSimTask,
    WfGateContext,
)

try:
    from renquant_backtesting.wf_gate import runner as wf_runner  # noqa: E402
    _RUNNER_OK = True
except (ModuleNotFoundError, ImportError):
    _RUNNER_OK = False

pytestmark = pytest.mark.skipif(
    not _RUNNER_OK,
    reason="umbrella scripts/ not reachable; Phase 1 invariant — Phase 5 flip will lift",
)


@pytest.fixture
def ctx(tmp_path):
    return WfGateContext(
        artifact_path=tmp_path / "fake_artifact.json",
        strategy_config="strategy_config.shadow.json",
        strategy_dir=tmp_path,
        wf_result=None,
        recipe_usage={"recipe_validated": True},
        trace_dir=tmp_path / "trace",
        jobs=1,
    )


def test_run_wf_sim_task_delegates_to_runner(ctx, monkeypatch):
    called = {}

    def fake_run_walk_forward(strategy_config, jobs=1, trace_dir=None):
        called["strategy_config"] = strategy_config
        called["jobs"] = jobs
        return {"passed": True, "wf_3cut_sharpe_mean": 0.5, "cuts": []}

    monkeypatch.setattr(wf_runner, "run_walk_forward", fake_run_walk_forward)
    ok = RunWfSimTask().run(ctx)
    assert ok is True
    assert ctx.wf_result["passed"] is True
    assert called["strategy_config"] == "strategy_config.shadow.json"
    assert called["jobs"] == 1


def test_run_trade_contract_task_skips_without_wf_result(ctx):
    ctx.wf_result = None
    ok = RunTradeContractTask().run(ctx)
    assert ok is True
    assert ctx.trade_contract_result["passed"] is False
    assert ctx.trade_contract_result["reason"] == "no wf_result"


def test_run_trade_contract_task_calls_runner_when_wf_done(ctx, monkeypatch, tmp_path):
    ctx.wf_result = {"passed": True, "cuts": []}
    # provide a minimal strategy_config file
    (tmp_path / "strategy_config.shadow.json").write_text("{}")
    called = {}

    def fake_gate(wf_result, config):
        called["wf_result"] = wf_result
        called["config"] = config
        return {"passed": True}

    monkeypatch.setattr(wf_runner, "run_trade_contract_gate", fake_gate)
    ok = RunTradeContractTask().run(ctx)
    assert ok is True
    assert ctx.trade_contract_result["passed"] is True
    assert called["wf_result"]["passed"] is True


def test_run_trade_monotonicity_task_skips_without_wf_result(ctx):
    ctx.wf_result = None
    ok = RunTradeMonotonicityTask().run(ctx)
    assert ok is True
    assert ctx.trade_gate_result["passed"] is False
    assert ctx.trade_gate_result["reason"] == "no wf_result"


def test_run_trade_monotonicity_task_calls_runner(ctx, monkeypatch):
    ctx.wf_result = {"passed": True, "cuts": []}
    monkeypatch.setattr(wf_runner, "run_trade_monotonicity_gate",
                        lambda wf_result, **kw: {"passed": True, "score_cols": []})
    ok = RunTradeMonotonicityTask().run(ctx)
    assert ok is True
    assert ctx.trade_gate_result["passed"] is True


def test_run_alpha_economics_task_calls_runner(ctx, monkeypatch):
    ctx.wf_result = {"passed": True, "cuts": []}
    monkeypatch.setattr(wf_runner, "run_alpha_economics_gate",
                        lambda wf_result: {"passed": True, "evidence": []})
    ok = RunAlphaEconomicsTask().run(ctx)
    assert ok is True
    assert ctx.alpha_economics_result["passed"] is True


def test_run_sanity_battery_task_delegates_to_runner(ctx, monkeypatch):
    called = {}

    def fake_sanity(artifact_path, artifact_usage=None):
        called["artifact_path"] = artifact_path
        called["artifact_usage"] = artifact_usage
        return {"passed": True, "real_ic": 0.04}

    monkeypatch.setattr(wf_runner, "run_sanity_battery", fake_sanity)
    ok = RunSanityBatteryTask().run(ctx)
    assert ok is True
    assert ctx.sanity_result["passed"] is True
    assert called["artifact_path"] == ctx.artifact_path
    assert called["artifact_usage"] == {"recipe_validated": True}
