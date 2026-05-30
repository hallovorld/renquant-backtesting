"""Smoke test for the B1 forensics chunk lift (Track C2.6).

5 more forensics files lifted into ``renquant_backtesting.forensics``:
acceptance_entry_ic, artifact_snapshot, challenger, model_acceptance,
model_acceptance_short. All Sim / forensics per kernel-inventory.md B1.

Phase 1 invariant: byte-equivalent; soft-skip on kernel.* dep.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


_BT_PKG = Path(__file__).resolve().parents[2] / "src" / "renquant_backtesting" / "forensics"
_UMBRELLA = Path(__file__).resolve().parents[3] / "RenQuant" / "backtesting" / \
            "renquant_104" / "kernel"

_LIFTED = (
    "acceptance_entry_ic.py",
    "artifact_snapshot.py",
    "challenger.py",
    "model_acceptance.py",
    "model_acceptance_short.py",
)


def test_byte_equivalent_to_umbrella() -> None:
    if not _UMBRELLA.exists():
        pytest.skip(f"umbrella not at {_UMBRELLA}")
    seen = 0
    for name in _LIFTED:
        bt = _BT_PKG / name
        um = _UMBRELLA / name
        if not bt.exists() or not um.exists():
            continue
        assert hashlib.md5(bt.read_bytes()).hexdigest() == hashlib.md5(um.read_bytes()).hexdigest(), \
            f"byte-mismatch: {name}"
        seen += 1
    assert seen == len(_LIFTED), f"expected all {len(_LIFTED)} lifted, saw {seen}"


def test_expected_files_present() -> None:
    present = {f.name for f in _BT_PKG.glob("*.py")}
    missing = set(_LIFTED) - present
    assert not missing, f"missing: {missing}"


def _try_import(modname: str) -> bool:
    try:
        __import__(modname)
        return True
    except ModuleNotFoundError as exc:
        if "kernel" in str(exc):
            return False
        raise


@pytest.mark.parametrize("name", [
    "renquant_backtesting.forensics.acceptance_entry_ic",
    "renquant_backtesting.forensics.artifact_snapshot",
    "renquant_backtesting.forensics.challenger",
    "renquant_backtesting.forensics.model_acceptance",
    "renquant_backtesting.forensics.model_acceptance_short",
])
def test_submodule_import_or_known_kernel_dep(name: str) -> None:
    ok = _try_import(name)
    assert ok in (True, False)
