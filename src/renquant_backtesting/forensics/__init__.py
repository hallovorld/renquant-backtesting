"""Backtest forensics / risk-metric modules lifted from the umbrella.

Per RFC §"Backfill Plan" functional-lift (copy-not-move), copied verbatim
from `backtesting/renquant_104/kernel/` and verified import-clean.

* ``sim_smoke``               — sim smoke-check helpers
* ``trade_score_diagnostics`` — per-trade score attribution
* ``risk_metrics``            — Sharpe / drawdown / risk statistics
"""
from __future__ import annotations

__all__: list[str] = []
