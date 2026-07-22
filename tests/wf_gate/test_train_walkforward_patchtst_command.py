"""Regression guard for the WF PatchTST driver command construction.

The driver used to shell out to ``<repo>/scripts/patchtst_hf.py`` — a script
that only ever existed in the umbrella working tree, never in the
renquant-backtesting checkout — so every fold died with
``can't open file '.../scripts/patchtst_hf.py': No such file or directory``
when run from a clean/pinned checkout. The fix invokes the renquant-model
training/calibration code *as a module* (repo-boundary correct) and resolves
the umbrella data/artifact root via ``--repo-root``.
"""
from __future__ import annotations

import os

import pandas as pd

from renquant_backtesting.wf_gate import train_walkforward_patchtst as twp


def _args(tmp_path, **extra):
    argv = [
        "--start-date", "2024-01-02",
        "--end-date", "2024-01-02",
        "--repo-root", str(tmp_path),
    ]
    for k, v in extra.items():
        argv += [f"--{k}", str(v)]
    return twp.parse_args(argv)


def test_train_cmd_invokes_model_module_not_missing_script(tmp_path) -> None:
    args = _args(tmp_path)
    cutoff = pd.Timestamp("2024-01-02")
    out_dir = twp.artifact_dir(args, cutoff)
    cmd = twp.train_cmd(args, cutoff, out_dir)

    assert cmd[:3] == [twp.sys.executable, "-m", "renquant_model_patchtst.hf_trainer"]
    # The old, broken script-path invocation must never come back.
    joined = " ".join(cmd)
    assert "scripts/patchtst_hf.py" not in joined
    assert "--save-model" in cmd
    assert cmd[cmd.index("--output-dir") + 1] == str(out_dir)
    assert cmd[cmd.index("--train-cutoff") + 1] == "2024-01-02"


def test_calibrator_cmd_invokes_model_module_with_panel_args(tmp_path) -> None:
    args = _args(tmp_path)
    cutoff = pd.Timestamp("2024-01-02")
    model_path = twp.model_path_for(twp.artifact_dir(args, cutoff), int(args.seed))
    cal_path = twp.calibrator_path_for(model_path)
    cmd = twp.calibrator_cmd(args, cutoff, model_path, cal_path)

    assert cmd[:3] == [twp.sys.executable, "-m", "renquant_model_patchtst.fit_calibrator"]
    assert "scripts/fit_hf_patchtst_calibrator.py" not in " ".join(cmd)
    for flag in ("--scorer-artifact", "--panel", "--raw-label-panel",
                 "--label-col", "--data-end", "--min-rows"):
        assert flag in cmd, flag


def test_constants_point_at_model_repo() -> None:
    assert twp.TRAIN_MODULE == "renquant_model_patchtst.hf_trainer"
    assert twp.CALIBRATOR_MODULE == "renquant_model_patchtst.fit_calibrator"
    # renquant-model src must be a wiring candidate so the subprocess can
    # import the training internals regardless of cwd.
    assert any(p.name == "src" and p.parent.name == "renquant-model"
               for p in twp.MULTIREPO_SRC_PATHS)


def test_subprocess_env_exports_existing_sibling_src(tmp_path) -> None:
    env = twp.subprocess_env()
    existing = [str(p) for p in twp.MULTIREPO_SRC_PATHS if p.exists()]
    if existing:  # in a normal multi-repo checkout the siblings are present
        pythonpath = env.get("PYTHONPATH", "").split(os.pathsep)
        for p in existing:
            assert p in pythonpath


def test_artifact_and_manifest_paths_follow_repo_root(tmp_path) -> None:
    args = _args(tmp_path)
    cutoff = pd.Timestamp("2024-01-02")
    out_dir = twp.artifact_dir(args, cutoff)
    expected_prefix = tmp_path / "backtesting" / "renquant_104" / "artifacts"
    assert str(out_dir).startswith(str(expected_prefix))
    assert twp.default_manifest_output(args) == str(
        expected_prefix / "walkforward_patchtst_manifest.json"
    )
