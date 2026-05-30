"""Smoke test for the metrics/ lift (Track C2 chunk 1).

Verifies the byte-identical copy at ``renquant_backtesting.metrics`` is
importable end-to-end and exposes the same callable surface as the umbrella's
``kernel.metrics``. Behavioural tests for the underlying functions stay in
umbrella `tests/test_evaluation_protocol.py` until Phase 5 flips callers.
"""
from __future__ import annotations


def test_deflated_sharpe_module_imports() -> None:
    from renquant_backtesting.metrics import deflated_sharpe
    # The two canonical entry points used by the gate
    assert callable(getattr(deflated_sharpe, "deflated_sharpe_ratio", None)) \
        or callable(getattr(deflated_sharpe, "deflated_sharpe", None))


def test_pbo_module_imports() -> None:
    from renquant_backtesting.metrics import pbo
    assert hasattr(pbo, "__file__")


def test_block_bootstrap_imports() -> None:
    from renquant_backtesting.metrics import block_bootstrap
    assert hasattr(block_bootstrap, "__file__")


def test_hac_se_imports_canonical_callables() -> None:
    from renquant_backtesting.metrics import hac_se
    # Per umbrella tests/test_evaluation_protocol.py:
    assert callable(getattr(hac_se, "andrews_optimal_lag", None))
    assert callable(getattr(hac_se, "newey_west_se", None))
    assert callable(getattr(hac_se, "hac_t_stat", None))


def test_perf_summary_imports() -> None:
    from renquant_backtesting.metrics import perf_summary
    assert hasattr(perf_summary, "__file__")


def test_package_init_exposes_namespace() -> None:
    import renquant_backtesting.metrics as m
    assert m.__doc__  # the __init__.py docstring carries over


def test_byte_equivalent_to_umbrella() -> None:
    """Phase-1 invariant: lift is byte-for-byte (no transformations yet)."""
    import hashlib
    from pathlib import Path
    bt = Path(__file__).resolve().parents[2] / "src" / "renquant_backtesting" / "metrics"
    umbrella = Path(__file__).resolve().parents[3] / "RenQuant" / "backtesting" / \
               "renquant_104" / "kernel" / "metrics"
    if not umbrella.exists():
        # Running outside the multi-repo layout — soft-skip
        import pytest
        pytest.skip(f"umbrella not at {umbrella}")
    for f in sorted(bt.glob("*.py")):
        u = umbrella / f.name
        if not u.exists():
            continue
        assert hashlib.md5(f.read_bytes()).hexdigest() == hashlib.md5(u.read_bytes()).hexdigest(), \
            f"byte-mismatch with umbrella: {f.name}"
