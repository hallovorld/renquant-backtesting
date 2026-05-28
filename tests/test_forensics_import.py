"""Smoke-import tests for lifted forensics / risk-metric modules."""
from __future__ import annotations

import importlib

import pytest

LIFTED_MODULES = [
    "renquant_backtesting.forensics.sim_smoke",
    "renquant_backtesting.forensics.trade_score_diagnostics",
    "renquant_backtesting.forensics.risk_metrics",
]


@pytest.mark.parametrize("module_name", LIFTED_MODULES)
def test_lifted_module_imports(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None
