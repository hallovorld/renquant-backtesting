from __future__ import annotations


def test_sim_cut_uses_package_sim_driver() -> None:
    from renquant_backtesting.wf_gate import runner

    cmd = runner._sim_driver_cmd(
        "strategy_config.sim_candidate.json",
        "2024-01-02",
        "2024-06-30",
    )

    assert cmd[:3] == [
        runner.PYTHON,
        "-m",
        "renquant_backtesting.wf_gate.sim_driver",
    ]
    assert "--repo-root" in cmd
    assert cmd[cmd.index("--repo-root") + 1] == str(runner.REPO)
    assert "scripts/run_sim_104.py" not in " ".join(cmd)
    assert "--strategy-config-name" in cmd
    assert cmd[cmd.index("--strategy-config-name") + 1] == "strategy_config.sim_candidate.json"
