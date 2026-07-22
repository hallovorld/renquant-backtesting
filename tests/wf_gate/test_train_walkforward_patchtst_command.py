"""Regression + provenance guards for the WF PatchTST driver.

The driver used to shell out to ``<repo>/scripts/patchtst_hf.py`` — a script
that only ever existed in the umbrella working tree, never in the
renquant-backtesting checkout — so every fold died with
``can't open file '.../scripts/patchtst_hf.py': No such file or directory``
when run from a clean/pinned checkout. The fix invokes the renquant-model
training/calibration code *as a module* against a single, pinned subrepo
assembly (fail closed — never an ad-hoc dev checkout) and resolves the umbrella
data/artifact root via ``--repo-root``.

Test tiers:
  * command-shape + resolver (always run, pure);
  * a pinned-assembly subprocess import smoke test (needs renquant-model in the
    resolved assembly);
  * a non-skip-calibrators fold test exercising the calibration/provenance path
    (opt-in via $RENQUANT_WF_TEST_DATASET / $RENQUANT_WF_TEST_RAW_LABEL).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

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


# ── command shape (the exact bug) ───────────────────────────────────────────

def test_train_cmd_invokes_model_module_not_missing_script(tmp_path) -> None:
    args = _args(tmp_path)
    cutoff = pd.Timestamp("2024-01-02")
    out_dir = twp.artifact_dir(args, cutoff)
    cmd = twp.train_cmd(args, cutoff, out_dir)

    assert cmd[:3] == [sys.executable, "-m", "renquant_model_patchtst.hf_trainer"]
    # The old, broken script-path invocation must never come back.
    assert "scripts/patchtst_hf.py" not in " ".join(cmd)
    assert "--save-model" in cmd
    assert cmd[cmd.index("--output-dir") + 1] == str(out_dir)
    assert cmd[cmd.index("--train-cutoff") + 1] == "2024-01-02"


def test_calibrator_cmd_invokes_model_module_with_panel_args(tmp_path) -> None:
    args = _args(tmp_path)
    cutoff = pd.Timestamp("2024-01-02")
    model_path = twp.model_path_for(twp.artifact_dir(args, cutoff), int(args.seed))
    cal_path = twp.calibrator_path_for(model_path)
    cmd = twp.calibrator_cmd(args, cutoff, model_path, cal_path)

    assert cmd[:3] == [sys.executable, "-m", "renquant_model_patchtst.fit_calibrator"]
    assert "scripts/fit_hf_patchtst_calibrator.py" not in " ".join(cmd)
    for flag in ("--scorer-artifact", "--panel", "--raw-label-panel",
                 "--label-col", "--data-end", "--min-rows"):
        assert flag in cmd, flag


def test_constants_point_at_model_repo() -> None:
    assert twp.TRAIN_MODULE == "renquant_model_patchtst.hf_trainer"
    assert twp.CALIBRATOR_MODULE == "renquant_model_patchtst.fit_calibrator"
    assert "renquant-model" in twp.REQUIRED_SUBREPOS


def test_artifact_and_manifest_paths_follow_repo_root(tmp_path) -> None:
    args = _args(tmp_path)
    cutoff = pd.Timestamp("2024-01-02")
    out_dir = twp.artifact_dir(args, cutoff)
    expected_prefix = tmp_path / "backtesting" / "renquant_104" / "artifacts"
    assert str(out_dir).startswith(str(expected_prefix))
    assert twp.default_manifest_output(args) == str(
        expected_prefix / "walkforward_patchtst_manifest.json"
    )


# ── pinned-assembly resolver: fail closed, no dev-checkout leakage ───────────

def test_subrepo_root_honors_env_and_has_no_home_fallback(tmp_path, monkeypatch) -> None:
    # Build a fake but complete assembly and point the env var at it.
    for repo in twp.REQUIRED_SUBREPOS:
        (tmp_path / repo / "src").mkdir(parents=True)
    monkeypatch.setenv("RENQUANT_SUBREPO_ROOT", str(tmp_path))
    root = twp.resolve_subrepo_root()
    assert root == tmp_path.resolve()
    srcs = twp.required_subrepo_src_paths()
    assert srcs == [tmp_path.resolve() / r / "src" for r in twp.REQUIRED_SUBREPOS]
    # PYTHONPATH is pinned to exactly the injected assembly — nothing from
    # ~/git/github or a loose sibling scan.
    pythonpath = twp.subprocess_env()["PYTHONPATH"].split(os.pathsep)
    home_github = str(Path.home() / "git" / "github")
    assert all(str(p).startswith(str(tmp_path.resolve())) for p in srcs)
    assert not any(p.startswith(home_github) for p in pythonpath[:len(srcs)])


def test_required_src_paths_fail_closed_on_incomplete_assembly(tmp_path, monkeypatch) -> None:
    # Assembly missing renquant-model → must raise, not fall through.
    for repo in twp.REQUIRED_SUBREPOS:
        if repo != "renquant-model":
            (tmp_path / repo / "src").mkdir(parents=True)
    monkeypatch.setenv("RENQUANT_SUBREPO_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="renquant-model"):
        twp.required_subrepo_src_paths()
    with pytest.raises(RuntimeError):
        twp.subprocess_env()


# ── pinned-assembly subprocess import smoke (codex point 2) ─────────────────

def _model_in_assembly() -> bool:
    root = twp.resolve_subrepo_root()
    return (root / "renquant-model" / "src" / "renquant_model_patchtst").is_dir()


@pytest.mark.skipif(not _model_in_assembly(),
                    reason="renquant-model not present in resolved subrepo assembly")
def test_subprocess_imports_model_pkg_against_pinned_assembly() -> None:
    """The subprocess env must let a fresh interpreter import BOTH model
    entrypoints from the pinned assembly (proves the invocation the driver
    actually launches will resolve)."""
    env = twp.subprocess_env()
    proc = subprocess.run(
        [sys.executable, "-c",
         "import renquant_model_patchtst.hf_trainer as t; "
         "import renquant_model_patchtst.fit_calibrator as c; "
         "assert hasattr(t, 'build_parser') and hasattr(c, 'build_parser'); "
         "print('import-ok')"],
        env=env, capture_output=True, text=True, cwd=str(Path.cwd()),
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "import-ok" in proc.stdout


# ── non-skip-calibrators fold: calibration + provenance (codex point 3) ─────

def _wf_data_env() -> tuple[Path, Path] | None:
    ds = os.environ.get("RENQUANT_WF_TEST_DATASET")
    raw = os.environ.get("RENQUANT_WF_TEST_RAW_LABEL")
    if not ds or not raw:
        return None
    ds_p, raw_p = Path(ds), Path(raw)
    if not ds_p.is_file() or not raw_p.is_file():
        return None
    return ds_p, raw_p


@pytest.mark.skipif(not _model_in_assembly(),
                    reason="renquant-model not present in resolved subrepo assembly")
@pytest.mark.skipif(_wf_data_env() is None,
                    reason="set RENQUANT_WF_TEST_DATASET + RENQUANT_WF_TEST_RAW_LABEL "
                           "(feature panel + raw-label panel) to run the calibrator leg")
def test_calibrator_leg_produces_artifacts_and_provenance(tmp_path) -> None:
    """Run ONE real fold WITHOUT --skip-calibrators and assert the production
    calibration/provenance path: per-fold model .pt + model sidecar + calibrator
    sidecar, and a manifest entry carrying BOTH the calibrator_uri and the model
    sidecar's trained/effective-cutoff dates."""
    dataset, raw_label = _wf_data_env()
    manifest_out = tmp_path / "manifest.json"
    cmd = [
        sys.executable, "-m",
        "renquant_backtesting.wf_gate.train_walkforward_patchtst",
        "--start-date", "2023-06-01", "--end-date", "2023-06-01",
        "--cadence-days", "45",
        "--repo-root", str(tmp_path),
        "--dataset", str(dataset), "--raw-label-panel", str(raw_label),
        "--manifest-output", str(manifest_out),
        "--device", "cpu", "--epochs", "1",
        "--seq-len", "16", "--d-model", "32", "--n-heads", "2", "--n-layers", "1",
        "--calibrator-min-rows", "50",
    ]
    env = os.environ.copy()
    # <repo>/src so ``python -m renquant_backtesting...`` resolves the package.
    src_dir = str(Path(twp.__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(
        [src_dir] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr[-2500:]

    fold = tmp_path / "backtesting" / "renquant_104" / "artifacts" / \
        "walkforward_patchtst" / "2023-06-01"
    model_pt = fold / "hf_patchtst_all_seed44_model.pt"
    model_sidecar = fold / "hf_patchtst_all_seed44_model.pt.metadata.json"
    calibrator = fold / "hf_patchtst-calibration.json"
    assert model_pt.is_file() and model_pt.stat().st_size > 0
    assert model_sidecar.is_file()
    assert calibrator.is_file(), "calibrator sidecar was not produced"

    manifest = json.loads(manifest_out.read_text())
    assert len(manifest["retrains"]) == 1
    entry = manifest["retrains"][0]
    # provenance read back from BOTH sidecars
    assert entry["calibrator_uri"] and entry["calibrator_uri"].endswith(
        "hf_patchtst-calibration.json")
    assert Path(entry["calibrator_uri"]).is_file()
    assert entry["trained_date"] and entry["effective_train_cutoff_date"]
    assert entry["artifact_uri"].endswith("hf_patchtst_all_seed44_model.pt")
