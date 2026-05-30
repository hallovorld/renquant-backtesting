"""Smoke test for the meta_label/ lift (Track C2.4).

Phase 1 invariant: byte-equivalent + file-presence; soft-skip on kernel.* dep.
Same shape as walk_forward/test_import_lift, reconciliation/test_import_lift.

Track C2.4: meta-labeling / triple-barrier (López de Prado AFML ch.20).
9 files: snapshot logger + labeler + predictor + triple-barrier + purged-kfold
+ pipeline Tasks (veto, snapshot) + Job (log).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


_BT_PKG = Path(__file__).resolve().parents[2] / "src" / "renquant_backtesting" / "meta_label"
_UMBRELLA = Path(__file__).resolve().parents[3] / "RenQuant" / "backtesting" / \
            "renquant_104" / "kernel" / "meta_label"


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
    assert seen >= 8, f"expected at least 8 lifted files, saw {seen}"


def test_expected_files_present() -> None:
    expected = {
        "__init__.py",
        "job_meta_label_log.py",
        "labeler.py",
        "predictor.py",
        "purged_kfold.py",
        "snapshot.py",
        "task_meta_label_veto.py",
        "task_snapshot.py",
        "triple_barrier.py",
    }
    present = {f.name for f in _BT_PKG.glob("*.py")}
    missing = expected - present
    assert not missing, f"missing: {missing}"


def _try_import(modname: str) -> bool:
    """True if module imports cleanly, False on known ModuleNotFoundError('kernel')."""
    try:
        __import__(modname)
        return True
    except ModuleNotFoundError as exc:
        if "kernel" in str(exc):
            return False
        raise


@pytest.mark.parametrize("name", [
    "renquant_backtesting.meta_label.snapshot",
    "renquant_backtesting.meta_label.labeler",
    "renquant_backtesting.meta_label.predictor",
    "renquant_backtesting.meta_label.purged_kfold",
    "renquant_backtesting.meta_label.triple_barrier",
])
def test_submodule_import_or_known_kernel_dep(name: str) -> None:
    """Either imports cleanly (no kernel.* deps) OR ModuleNotFoundError('kernel')."""
    ok = _try_import(name)
    assert ok in (True, False)
