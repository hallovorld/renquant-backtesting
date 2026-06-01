from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


def test_sim_driver_cli_exposes_repo_root() -> None:
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src")
    proc = subprocess.run(
        [sys.executable, "-m", "renquant_backtesting.wf_gate.sim_driver", "--help"],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    assert "--repo-root" in proc.stdout
    assert "RENQUANT_REPO_ROOT" in proc.stdout
