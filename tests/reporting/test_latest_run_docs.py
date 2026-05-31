from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from renquant_backtesting.reporting.latest_run_docs import (
    find_latest_run,
    generate_latest_run_docs,
)


def test_generate_latest_run_docs_from_wf_gate_metadata(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    artifact = root / "panel-ltr.json"
    artifact.write_text(json.dumps({
        "metadata": {
            "wf_gate_metadata": {
                "passed": True,
                "wf_3cut_sharpe_mean": 0.8,
                "spy_sharpe_mean": 0.5,
                "strategy_minus_spy_sharpe_mean": 0.3,
                "wf_3cut_apy_mean": 0.12,
                "spy_apy_mean": 0.07,
                "trade_buy_source_counts_total": {"PanelScoringJob": 4},
                "cuts": [{
                    "start": "2025-01-01",
                    "end": "2025-12-31",
                    "annual_net_sharpe": 0.8,
                    "annual_net_apy": 0.12,
                    "market_context": {"spy_sharpe": 0.5, "spy_apy": 0.07},
                    "trade_trace_summary": {"n_buys": 4, "n_sells": 2},
                }],
                "benchmark_by_dominant_regime": {
                    "BULL": {
                        "mean_sharpe": 0.8,
                        "mean_spy_sharpe": 0.5,
                        "mean_apy": 0.12,
                        "mean_spy_apy": 0.07,
                    },
                },
            },
        },
    }), encoding="utf-8")

    out = generate_latest_run_docs(
        search_roots=[root],
        docs_dir=tmp_path / "docs",
        now=datetime(2026, 5, 30, tzinfo=timezone.utc),
    )

    text = out.read_text(encoding="utf-8")
    assert "Latest Simulation Run" in text
    assert "`wf_3cut_sharpe_mean` | 0.800" in text
    assert "2025-01-01 to 2025-12-31" in text
    assert (tmp_path / "docs" / "latest-run-assets" / "summary.svg").exists()
    assert (tmp_path / "docs" / "latest-run-assets" / "cuts.svg").exists()


def test_latest_run_prefers_informative_trade_run_over_newer_empty_equity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    old = root / "20260530T150815Z"
    new = root / "20260530T220958Z"
    old.mkdir(parents=True)
    new.mkdir(parents=True)

    old_equity = old / "2025.equity.json"
    old_equity.write_text(json.dumps({
        "start": "2025-01-01",
        "end": "2025-12-31",
        "apy": 0.10,
        "sharpe": 0.70,
        "total_return": 0.10,
        "final_value": 110000.0,
        "equity": {"2025-01-01": 100000.0, "2025-12-31": 110000.0},
    }), encoding="utf-8")
    (old / "2025.trades.json").write_text(json.dumps([
        {"action": "buy", "source_job": "JointPortfolioQPJob", "regime": "BULL"},
        {"action": "sell", "exit_reason": "trailing_stop", "regime": "BULL"},
    ]), encoding="utf-8")

    new_equity = new / "2026.equity.json"
    new_equity.write_text(json.dumps({
        "start": "2026-01-01",
        "end": "2026-12-31",
        "apy": 0.0,
        "total_return": 0.0,
        "final_value": 100000.0,
        "equity": {"2026-01-01": 100000.0, "2026-12-31": 100000.0},
    }), encoding="utf-8")
    (new / "2026.trades.json").write_text("[]", encoding="utf-8")

    os.utime(old_equity, (100, 100))
    os.utime(new_equity, (200, 200))

    latest = find_latest_run([root])
    assert latest is not None
    assert latest.source == old_equity
    assert latest.metrics["n_buys"] == 1
    assert latest.metrics["n_sells"] == 1
    assert "Newer lower-information artifact ignored" in latest.warnings[0]

    out = generate_latest_run_docs(
        search_roots=[root],
        docs_dir=tmp_path / "docs",
        now=datetime(2026, 5, 30, tzinfo=timezone.utc),
    )
    text = out.read_text(encoding="utf-8")
    assert "2025.equity.json" in text
    assert "`n_buys` | 1.000" in text
    assert "`trade_buy_source_counts_total.JointPortfolioQPJob` | 1.000" in text
    assert "2026.equity.json" in text
