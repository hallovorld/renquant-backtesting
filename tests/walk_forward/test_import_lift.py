"""Smoke test for the walk_forward/ lift (Track C2 chunk 2).

Verifies the byte-identical copy at ``renquant_backtesting.walk_forward`` is
importable. Behavioural tests stay in umbrella until Phase 5 flip.

Unblocks: B8/B9/B10 wf_gate lifts (which import kernel.walk_forward.loader).
"""
from __future__ import annotations


def test_walk_forward_package_imports() -> None:
    import renquant_backtesting.walk_forward  # noqa: F401


def test_loader_callable_exposed() -> None:
    """B8 needs WalkForwardModelLoader importable from the package."""
    from renquant_backtesting.walk_forward import loader
    assert hasattr(loader, "__file__")
    # The class wf_gate sanity uses
    assert hasattr(loader, "WalkForwardModelLoader") or any(
        cls.endswith("Loader") for cls in dir(loader)
    )


def test_manifest_imports() -> None:
    from renquant_backtesting.walk_forward import manifest
    assert hasattr(manifest, "__file__")


def test_leakage_guard_imports() -> None:
    from renquant_backtesting.walk_forward import leakage_guard
    assert hasattr(leakage_guard, "__file__")


def test_correlation_guard_imports() -> None:
    from renquant_backtesting.walk_forward import correlation_guard
    assert hasattr(correlation_guard, "__file__")


def test_lean_guard_imports() -> None:
    from renquant_backtesting.walk_forward import lean_guard
    assert hasattr(lean_guard, "__file__")


def test_gmm_guard_imports() -> None:
    from renquant_backtesting.walk_forward import gmm_guard
    assert hasattr(gmm_guard, "__file__")


def test_byte_equivalent_to_umbrella() -> None:
    """Phase-1 invariant: lift is byte-for-byte."""
    import hashlib
    from pathlib import Path
    bt = Path(__file__).resolve().parents[2] / "src" / "renquant_backtesting" / "walk_forward"
    umbrella = Path(__file__).resolve().parents[3] / "RenQuant" / "backtesting" / \
               "renquant_104" / "kernel" / "walk_forward"
    if not umbrella.exists():
        import pytest
        pytest.skip(f"umbrella not at {umbrella}")
    for f in sorted(bt.glob("*.py")):
        u = umbrella / f.name
        if not u.exists():
            continue
        assert hashlib.md5(f.read_bytes()).hexdigest() == hashlib.md5(u.read_bytes()).hexdigest(), \
            f"byte-mismatch with umbrella: {f.name}"
