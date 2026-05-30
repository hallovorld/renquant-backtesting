"""Smoke test for the reconciliation/ lift (Track C2 chunk 3).

Phase 1 invariant: byte-equivalent + file-presence; soft-skip on kernel.* dep.
Same shape as walk_forward/test_import_lift.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


_BT_PKG = Path(__file__).resolve().parents[2] / "src" / "renquant_backtesting" / "reconciliation"
_UMBRELLA = Path(__file__).resolve().parents[3] / "RenQuant" / "backtesting" / \
            "renquant_104" / "kernel" / "reconciliation"


def test_byte_equivalent_to_umbrella() -> None:
    if not _UMBRELLA.exists():
        pytest.skip(f"umbrella not at {_UMBRELLA}")
    seen = 0
    for f in sorted(_BT_PKG.glob("*.py")):
        u = _UMBRELLA / f.name
        if not u.exists():
            continue
        assert hashlib.md5(f.read_bytes()).hexdigest() == hashlib.md5(u.read_bytes()).hexdigest(), \
            f"byte-mismatch: {f.name}"
        seen += 1
    assert seen >= 1


def test_expected_files_present() -> None:
    expected = {"__init__.py", "live_sim_reconcile.py"}
    present = {f.name for f in _BT_PKG.glob("*.py")}
    missing = expected - present
    assert not missing, f"missing: {missing}"


def test_submodule_import_or_known_kernel_dep() -> None:
    """Either imports cleanly OR ModuleNotFoundError('kernel') (acknowledged)."""
    try:
        import renquant_backtesting.reconciliation.live_sim_reconcile  # noqa: F401
    except ModuleNotFoundError as exc:
        if "kernel" not in str(exc):
            raise
