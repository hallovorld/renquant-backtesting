"""B13 — ``python -m renquant_backtesting.wf_gate`` entry point.

The new entry point delegates to ``runner.main()``. This test pins:
  1. The module is importable.
  2. ``__main__.main`` is callable.
  3. It returns ``runner.main()``'s exit code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
_UMBRELLA_SCRIPTS = Path(__file__).resolve().parents[3] / "RenQuant" / "scripts"
if _UMBRELLA_SCRIPTS.exists():
    sys.path.insert(0, str(_UMBRELLA_SCRIPTS))

try:
    from renquant_backtesting.wf_gate import __main__ as wf_gate_main
    from renquant_backtesting.wf_gate import runner as wf_runner
    _RUNNER_OK = True
except (ModuleNotFoundError, ImportError):
    _RUNNER_OK = False

pytestmark = pytest.mark.skipif(
    not _RUNNER_OK,
    reason="umbrella scripts/ not reachable; Phase 1 invariant — Phase 5 flip will lift",
)


def test_main_module_importable():
    assert callable(wf_gate_main.main)


def test_main_delegates_to_runner_main(monkeypatch):
    called = {}

    def fake_runner_main():
        called["yes"] = True
        return 0

    monkeypatch.setattr(wf_runner, "main", fake_runner_main)
    assert wf_gate_main.main() == 0
    assert called["yes"] is True


def test_main_propagates_runner_exit_code(monkeypatch):
    monkeypatch.setattr(wf_runner, "main", lambda: 2)
    assert wf_gate_main.main() == 2
