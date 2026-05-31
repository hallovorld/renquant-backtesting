"""Smoke test for the top-level triple_barrier lift (Track C2.10).

Lifts kernel/triple_barrier.py (LABELS — used by training_panel) into
renquant_backtesting.labels.triple_barrier.

Distinct from renquant_backtesting.meta_label.triple_barrier which is the
faithful AFML §3.4 algorithm port (C2.4). The two paths exist because
training_panel uses the former for label construction; meta_label/ is the
canonical reference. Consolidation is a §5.13.5 followup.

Phase 1 invariant: byte-equivalent + clean import.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


_BT = Path(__file__).resolve().parents[2] / "src" / "renquant_backtesting" / "labels" / "triple_barrier.py"
_UM = Path(__file__).resolve().parents[3] / "RenQuant" / "backtesting" / "renquant_104" / "kernel" / "triple_barrier.py"


def test_byte_equivalent_to_umbrella() -> None:
    if not _UM.exists():
        pytest.skip(f"umbrella not at {_UM}")
    assert _BT.exists()
    assert hashlib.md5(_BT.read_bytes()).hexdigest() == hashlib.md5(_UM.read_bytes()).hexdigest()


def test_imports_cleanly() -> None:
    import renquant_backtesting.labels.triple_barrier  # noqa: F401
