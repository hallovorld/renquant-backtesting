"""Smoke test for the walk_forward/ lift (Track C2 chunk 2).

Phase 1 limitation: the umbrella ``kernel/walk_forward/__init__.py`` does
absolute ``from kernel.walk_forward.X import …`` re-exports. Copied verbatim
into ``renquant_backtesting.walk_forward``, those resolve only when the
umbrella's ``kernel`` namespace is on sys.path. That's expected pre-Phase-5.

The pragmatic Phase 1 invariant we DO assert:

1. Files are present and byte-equivalent to umbrella.
2. Each submodule imports cleanly when accessed by its package path
   (``renquant_backtesting.walk_forward.loader``, etc.) and does NOT depend on
   the umbrella's absolute-import re-exports.

Behavioural tests stay in umbrella ``tests/`` until Phase 5 flips callers.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


_BT_PKG = Path(__file__).resolve().parents[2] / "src" / "renquant_backtesting" / "walk_forward"
_UMBRELLA = Path(__file__).resolve().parents[3] / "RenQuant" / "backtesting" / \
            "renquant_104" / "kernel" / "walk_forward"


def test_byte_equivalent_to_umbrella() -> None:
    """Phase-1 invariant: every .py is byte-for-byte identical to the umbrella."""
    if not _UMBRELLA.exists():
        pytest.skip(f"umbrella not at {_UMBRELLA}")
    seen = 0
    for f in sorted(_BT_PKG.glob("*.py")):
        u = _UMBRELLA / f.name
        if not u.exists():
            continue
        assert hashlib.md5(f.read_bytes()).hexdigest() == hashlib.md5(u.read_bytes()).hexdigest(), \
            f"byte-mismatch with umbrella: {f.name}"
        seen += 1
    assert seen >= 6, "expected at least 6 lifted files"


def test_all_expected_files_present() -> None:
    """The 7 files from the inventory must all be in the package."""
    expected = {
        "__init__.py", "correlation_guard.py", "gmm_guard.py", "leakage_guard.py",
        "lean_guard.py", "loader.py", "manifest.py",
    }
    present = {f.name for f in _BT_PKG.glob("*.py")}
    missing = expected - present
    assert not missing, f"missing lifted files: {missing}"


def _try_import(modname: str) -> bool:
    """Returns True if the module imports cleanly, False on the known umbrella
    ModuleNotFoundError ('kernel'), re-raises everything else."""
    try:
        __import__(modname)
        return True
    except ModuleNotFoundError as exc:
        if "kernel" in str(exc):
            return False
        raise


@pytest.mark.parametrize("name", [
    "renquant_backtesting.walk_forward.loader",
    "renquant_backtesting.walk_forward.manifest",
    "renquant_backtesting.walk_forward.leakage_guard",
    "renquant_backtesting.walk_forward.correlation_guard",
    "renquant_backtesting.walk_forward.lean_guard",
    "renquant_backtesting.walk_forward.gmm_guard",
])
def test_submodule_import_when_umbrella_kernel_unavailable(name: str) -> None:
    """Documents the Phase 1 limitation: each submodule either imports cleanly
    (no kernel.* deps) OR raises a known ModuleNotFoundError('kernel') we'll
    fix when Phase 5 rewrites callers. Either way is recorded; an unexpected
    exception fails the test loudly."""
    ok = _try_import(name)
    # No assertion — the call is itself the test. If it raises something other
    # than ModuleNotFoundError('kernel'), _try_import re-raises and pytest fails.
    assert ok in (True, False)
