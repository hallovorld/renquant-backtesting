from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from renquant_backtesting.wf_gate import stamp_walkforward_fingerprints as stamp


def test_stamp_cli_exposes_repo_root() -> None:
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    pythonpath = [
        str(repo / "src"),
        str(repo.parent / "renquant-pipeline" / "src"),
        str(repo.parent / "renquant-common" / "src"),
    ]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "renquant_backtesting.wf_gate.stamp_walkforward_fingerprints",
            "--help",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    assert "--repo-root" in proc.stdout
    assert "RENQUANT_REPO_ROOT" in proc.stdout


def test_configure_repo_root_resolves_strategy_paths(tmp_path: Path) -> None:
    stamp._configure_repo_root(tmp_path)

    assert stamp.REPO == tmp_path.resolve()
    assert stamp.STRATEGY_DIR == tmp_path / "backtesting" / "renquant_104"
    assert (
        stamp._resolve_strategy_path("artifacts/model.json")
        == tmp_path / "backtesting" / "renquant_104" / "artifacts" / "model.json"
    )
