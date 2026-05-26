from __future__ import annotations

from pathlib import Path

import pytest

from renquant_backtesting import BacktestContext, BacktestPipeline


def test_backtest_pipeline_runs_with_fingerprinted_manifests(tmp_path: Path) -> None:
    def runner(ctx: BacktestContext):
        return {"apy": 0.1, "sharpe": 1.2, "out": str(ctx.output_dir)}

    ctx = BacktestContext(
        strategy_manifest={"fingerprint": "sha256:strategy"},
        data_manifest={"fingerprint": "sha256:data"},
        artifact_manifest={"fingerprint": "sha256:model"},
        output_dir=tmp_path / "out",
    )
    result = BacktestPipeline(runner).run(ctx)

    assert result.ok is True
    assert ctx.report["sharpe"] == pytest.approx(1.2)
    assert ctx.output_dir.exists()


def test_backtest_pipeline_rejects_unfingerprinted_data(tmp_path: Path) -> None:
    ctx = BacktestContext(
        strategy_manifest={"fingerprint": "sha256:strategy"},
        data_manifest={},
        artifact_manifest={"fingerprint": "sha256:model"},
        output_dir=tmp_path / "out",
    )
    with pytest.raises(ValueError, match="data_manifest missing fingerprint"):
        BacktestPipeline(lambda _: {}).run(ctx)
