"""Backtest/simulation pipeline contract."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from renquant_common import Job, Pipeline, Task


@dataclass
class BacktestContext:
    strategy_manifest: dict[str, Any]
    data_manifest: dict[str, Any]
    artifact_manifest: dict[str, Any]
    output_dir: Path
    report: dict[str, Any] = field(default_factory=dict)


Runner = Callable[[BacktestContext], dict[str, Any]]


class ValidateBacktestInputsTask(Task):
    def run(self, ctx: BacktestContext) -> bool | None:
        for name, manifest in (
            ("strategy_manifest", ctx.strategy_manifest),
            ("data_manifest", ctx.data_manifest),
            ("artifact_manifest", ctx.artifact_manifest),
        ):
            if not manifest.get("fingerprint"):
                raise ValueError(f"{name} missing fingerprint")
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        return True


class RunBacktestTask(Task):
    def __init__(self, runner: Runner) -> None:
        self.runner = runner

    def run(self, ctx: BacktestContext) -> bool | None:
        ctx.report = self.runner(ctx)
        return True


class BacktestJob(Job):
    def __init__(self, runner: Runner) -> None:
        self._tasks = [ValidateBacktestInputsTask(), RunBacktestTask(runner)]

    @property
    def tasks(self) -> list[Task]:
        return self._tasks


class BacktestPipeline(Pipeline):
    def __init__(self, runner: Runner) -> None:
        super().__init__([BacktestJob(runner)], name="backtest")
