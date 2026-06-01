from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]


def _subrepo_pythonpath() -> str:
    paths = [REPO / "src"]
    for base in (REPO.parent, Path("/Users/renhao/git/github")):
        for name in (
            "renquant-common",
            "renquant-pipeline",
            "renquant-base-data",
            "renquant-artifacts",
        ):
            src = base / name / "src"
            if src.exists() and src not in paths:
                paths.append(src)
    return os.pathsep.join(str(path) for path in paths)


def _write_umbrella_helper_stubs(umbrella: Path) -> None:
    scripts = umbrella / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "qp_contracts.py").write_text(
        "def validate_qp_contract_config(*args, **kwargs):\n"
        "    return None\n",
        encoding="utf-8",
    )
    (scripts / "trade_contracts.py").write_text(
        "def evaluate_trade_contract(*args, **kwargs):\n"
        "    return {}\n",
        encoding="utf-8",
    )
    (scripts / "trade_monotonicity.py").write_text(
        "def evaluate_trade_monotonicity(*args, **kwargs):\n"
        "    return {}\n",
        encoding="utf-8",
    )
    (scripts / "wf_config_parity.py").write_text(
        "def evaluate_wf_config_parity(*args, **kwargs):\n"
        "    return {}\n",
        encoding="utf-8",
    )


def test_wf_gate_module_help_uses_package_imports_and_repo_root(tmp_path: Path) -> None:
    pytest.importorskip("pydantic")
    umbrella = tmp_path / "RenQuant"
    (umbrella / "backtesting" / "renquant_104").mkdir(parents=True)
    _write_umbrella_helper_stubs(umbrella)
    env = dict(os.environ)
    env["PYTHONPATH"] = _subrepo_pythonpath()
    env["RENQUANT_REPO_ROOT"] = str(umbrella)

    proc = subprocess.run(
        [sys.executable, "-m", "renquant_backtesting.wf_gate", "--help"],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    assert "--artifact" in proc.stdout
    assert "--derive-config-from-prod" in proc.stdout


def test_runner_resolves_renquant_repo_root_from_env(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("pydantic")
    umbrella = tmp_path / "RenQuant"
    strategy_dir = umbrella / "backtesting" / "renquant_104"
    strategy_dir.mkdir(parents=True)
    _write_umbrella_helper_stubs(umbrella)
    monkeypatch.setenv("RENQUANT_REPO_ROOT", str(umbrella))
    for path in reversed(_subrepo_pythonpath().split(os.pathsep)):
        if path and path not in sys.path:
            sys.path.insert(0, path)

    from renquant_backtesting.wf_gate import runner  # noqa: PLC0415

    assert runner._resolve_repo_root() == umbrella.resolve()
