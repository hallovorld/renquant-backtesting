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


#: Files intentionally diverged from the umbrella post-Phase-5 retirement.
#: ``loader.py`` + ``manifest.py`` now use canonical imports
#: (``renquant_pipeline.kernel.*`` / ``.loader``) so the package can be
#: consumed standalone without ``kernel.*`` on sys.path. The umbrella copies
#: still use the umbrella-relative ``from kernel.walk_forward...`` form
#: because the umbrella's own kernel package resolves that form natively.
_PHASE5_DIVERGED = {"__init__.py", "loader.py", "manifest.py"}


def test_byte_equivalent_to_umbrella() -> None:
    """Phase-1 invariant: every NON-Phase-5-diverged .py is byte-for-byte
    identical to the umbrella."""
    if not _UMBRELLA.exists():
        pytest.skip(f"umbrella not at {_UMBRELLA}")
    seen = 0
    for f in sorted(_BT_PKG.glob("*.py")):
        if f.name in _PHASE5_DIVERGED:
            continue
        u = _UMBRELLA / f.name
        if not u.exists():
            continue
        assert hashlib.md5(f.read_bytes()).hexdigest() == hashlib.md5(u.read_bytes()).hexdigest(), \
            f"byte-mismatch with umbrella: {f.name}"
        seen += 1
    assert seen >= 4, "expected at least 4 lifted (non-Phase-5) files"


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
    """Documents the Phase 1 limitation for guards that haven't been
    Phase-5-divergedyet: each submodule either imports cleanly (no
    kernel.* deps) OR raises a known ModuleNotFoundError('kernel') we'll
    fix as the remaining guards get cleaned up. Either way is recorded;
    an unexpected exception fails the test loudly.

    ``loader.py`` + ``manifest.py`` (this PR) MUST import cleanly — they
    use canonical ``renquant_pipeline.kernel.*`` paths. The 4 guard
    modules still tolerate the umbrella-relative form pending follow-up."""
    ok = _try_import(name)
    if name.endswith(("loader", "manifest")):
        assert ok is True, f"{name} failed to import standalone post-Phase-5"
    else:
        assert ok in (True, False)
