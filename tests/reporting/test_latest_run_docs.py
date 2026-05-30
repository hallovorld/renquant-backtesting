from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from renquant_backtesting.reporting.latest_run_docs import generate_latest_run_docs


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
