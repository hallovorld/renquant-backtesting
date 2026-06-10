from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_qp_contract_resolves_from_package_without_umbrella_path(tmp_path: Path) -> None:
    """Regression: 2026-06-08 weekly_wf_promote lost all 3 sim cuts to
    ``ModuleNotFoundError: qp_contracts`` because sim_driver used a bare
    top-level import that only resolves when the umbrella ``scripts/`` dir
    happens to be on sys.path. The validator must come from
    ``renquant_backtesting.wf_gate.qp_contracts`` so the lifted gate works
    from any checkout (.subrepo_runtime included).

    The fixture config enables QP with a broken μ contract, so reaching
    exit code 3 ("QP contract failed") proves the import resolved and the
    validator ran — under an env where the old bare import cannot resolve.
    """
    repo = Path(__file__).resolve().parents[2]
    strategy_dir = tmp_path / "backtesting" / "renquant_104"
    strategy_dir.mkdir(parents=True)
    (strategy_dir / "strategy_config.json").write_text(json.dumps({
        "rotation": {"joint_actions": {"enabled": True, "solver": "qp"}},
    }))

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src")

    proc = subprocess.run(
        [
            sys.executable, "-m", "renquant_backtesting.wf_gate.sim_driver",
            "--repo-root", str(tmp_path),
        ],
        text=True,
        capture_output=True,
        env=env,
        cwd=str(tmp_path),
    )

    assert "ModuleNotFoundError" not in proc.stderr
    assert proc.returncode == 3, proc.stderr
    assert "QP contract failed" in proc.stderr
