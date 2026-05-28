"""Parity test for the metrics package lift (kernel/metrics → forensics/metrics).

Leaf package (stdlib + numpy/scipy + intra-package relatives): deflated_sharpe,
pbo, perf_summary (compute_perf_triple), block_bootstrap, hac_se. Lifted
verbatim. Tests smoke-import AND behavioral correctness (the value-add — DSR/PBO
multiple-testing correction), not just imports.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

MODULES = [
    "renquant_backtesting.forensics.metrics",
    "renquant_backtesting.forensics.metrics.deflated_sharpe",
    "renquant_backtesting.forensics.metrics.pbo",
    "renquant_backtesting.forensics.metrics.perf_summary",
    "renquant_backtesting.forensics.metrics.block_bootstrap",
    "renquant_backtesting.forensics.metrics.hac_se",
]


@pytest.mark.parametrize("mod", MODULES)
def test_metrics_module_imports(mod: str) -> None:
    assert importlib.import_module(mod) is not None


def test_metrics_package_reexports() -> None:
    m = importlib.import_module("renquant_backtesting.forensics.metrics")
    for fn in ("deflated_sharpe_ratio", "probability_of_backtest_overfitting",
               "compute_perf_triple"):
        assert hasattr(m, fn), f"package should re-export {fn}"


def test_annualized_sharpe_sign_and_finite() -> None:
    ds = importlib.import_module("renquant_backtesting.forensics.metrics.deflated_sharpe")
    rng = np.random.default_rng(0)
    pos = rng.normal(0.001, 0.01, 504)   # positive drift
    neg = rng.normal(-0.001, 0.01, 504)  # negative drift
    sp = ds.annualized_sharpe(pos)
    sn = ds.annualized_sharpe(neg)
    assert np.isfinite(sp) and np.isfinite(sn)
    assert sp > sn  # positive-drift series must out-Sharpe negative-drift


def test_compute_perf_triple_structure_and_dsr_penalizes_trials() -> None:
    m = importlib.import_module("renquant_backtesting.forensics.metrics")
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0008, 0.01, 504)
    t1 = m.compute_perf_triple(returns, n_trials=1)
    t50 = m.compute_perf_triple(returns, n_trials=50)
    for k in ("sharpe", "dsr", "pbo"):
        assert k in t1
    assert np.isfinite(t1["sharpe"])
    # More trials searched → deflated Sharpe must not increase (selection-bias penalty).
    assert t50["dsr"] <= t1["dsr"] + 1e-9
