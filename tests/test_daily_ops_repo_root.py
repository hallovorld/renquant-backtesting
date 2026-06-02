"""Repo-root contracts for lifted daily ops modules."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from renquant_backtesting.repo_root import resolve_repo_root
from renquant_backtesting.lean_export import export_lean_data, export_lean_watchlist


REPO = Path(__file__).resolve().parent.parent


def _pythonpath() -> str:
    pieces = [
        REPO / "src",
        REPO.parent / "renquant-common" / "src",
        REPO.parent / "renquant-pipeline" / "src",
    ]
    return os.pathsep.join(str(p) for p in pieces)


def test_resolve_repo_root_prefers_explicit_then_env(monkeypatch, tmp_path: Path) -> None:
    env_root = tmp_path / "env-root"
    explicit_root = tmp_path / "explicit-root"
    env_root.mkdir()
    explicit_root.mkdir()
    monkeypatch.setenv("RENQUANT_REPO_ROOT", str(env_root))

    assert resolve_repo_root(explicit_root) == explicit_root.resolve()
    assert resolve_repo_root() == env_root.resolve()


def test_lifted_daily_ops_clis_expose_repo_root() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath()
    modules = [
        "renquant_backtesting.analysis.smoke_test_model",
        "renquant_backtesting.analysis.compute_portfolio_metrics",
        "renquant_backtesting.lean_export.export_lean_data",
        "renquant_backtesting.lean_export.export_lean_watchlist",
    ]

    for module in modules:
        proc = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            check=True,
            text=True,
            capture_output=True,
            env=env,
        )
        assert "--repo-root" in proc.stdout
        assert "RENQUANT_REPO_ROOT" in proc.stdout


def test_lean_watchlist_reads_strategy_from_explicit_repo_root(tmp_path: Path) -> None:
    strategy_dir = tmp_path / "backtesting" / "renquant_104"
    strategy_dir.mkdir(parents=True)
    (strategy_dir / "strategy_config.json").write_text(
        json.dumps({"watchlist": ["AAPL"], "benchmark": "SPY", "stock_symbol": "NVDA"}),
        encoding="utf-8",
    )

    assert export_lean_watchlist.get_watchlist("renquant_104", tmp_path) == [
        "AAPL",
        "SPY",
        "NVDA",
    ]


def test_export_symbol_missing_parquet_error_uses_explicit_repo_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as exc:
        export_lean_data.export_symbol("XYZ", repo_root=tmp_path)

    assert str(tmp_path / "data" / "ohlcv" / "XYZ" / "1d.parquet") in str(exc.value)

