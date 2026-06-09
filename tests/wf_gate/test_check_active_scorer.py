"""RFC #259 P0b — active-config scorer gate check (canonical subrepo home).

Exercises check_config end-to-end through the REAL assert_artifact_gated (same
package), building a temp umbrella layout (backtesting/<strategy>/ + config +
artifact). Verifies config→artifact resolution (incl. `../..` paths) and the
ok / violation / error status mapping.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from renquant_backtesting.wf_gate.check_active_scorer import check_config, main


def _gated_meta() -> dict:
    """A fully-valid wf_gate_metadata that passes _check_wf_gate (mirrors #44)."""
    return {
        "wf_gate_metadata": {
            "passed": True,
            "wf_3cut_sharpe_mean": 1.0,
            "run_at": datetime.now(timezone.utc).isoformat(),
            "trade_monotonicity": {
                "passed": True,
                "allow_pass_open": False,
                "regimes": [{"eligible": True, "passed": True}],
            },
            "alpha_economics": {"passed": True},
            "sanity_regime_ic": {"passed": True},
        }
    }


def _layout(tmp_path: Path, strategy="renquant_104"):
    sdir = tmp_path / "backtesting" / strategy
    sdir.mkdir(parents=True)
    return tmp_path, sdir


def _write_config(sdir: Path, name: str, artifact_rel: str, kind="xgb"):
    (sdir / name).write_text(json.dumps({
        "ranking": {"panel_scoring": {"kind": kind, "artifact_path": artifact_rel}}
    }))


def test_gated_json_scorer_ok(tmp_path):
    root, sdir = _layout(tmp_path)
    (sdir / "model.json").write_text(json.dumps(_gated_meta()))
    _write_config(sdir, "strategy_config.json", "model.json")
    status, detail = check_config(root, "renquant_104", "strategy_config.json")
    assert status == "ok", detail


def test_ungated_scorer_violation(tmp_path):
    root, sdir = _layout(tmp_path)
    (sdir / "model.json").write_text(json.dumps({"kind": "xgb"}))  # no metadata
    _write_config(sdir, "strategy_config.json", "model.json")
    status, detail = check_config(root, "renquant_104", "strategy_config.json")
    assert status == "violation"
    assert "missing wf_gate_metadata" in detail


def test_failed_gate_violation(tmp_path):
    root, sdir = _layout(tmp_path)
    meta = _gated_meta()
    meta["wf_gate_metadata"]["passed"] = False
    (sdir / "model.json").write_text(json.dumps(meta))
    _write_config(sdir, "strategy_config.json", "model.json")
    status, detail = check_config(root, "renquant_104", "strategy_config.json")
    assert status == "violation"
    assert "passed=False" in detail


def test_parent_relative_artifact_path_resolves(tmp_path):
    """`../../artifacts/...` resolves like the live PatchTST config."""
    root, sdir = _layout(tmp_path)
    art = root / "artifacts" / "m.json"
    art.parent.mkdir(parents=True)
    art.write_text(json.dumps(_gated_meta()))
    _write_config(sdir, "strategy_config.json", "../../artifacts/m.json", kind="hf_patchtst")
    status, detail = check_config(root, "renquant_104", "strategy_config.json")
    assert status == "ok", detail


def test_missing_config_error(tmp_path):
    root, _ = _layout(tmp_path)
    status, detail = check_config(root, "renquant_104", "nope.json")
    assert status == "error"
    assert "not found" in detail


def test_missing_artifact_path_key_error(tmp_path):
    root, sdir = _layout(tmp_path)
    (sdir / "strategy_config.json").write_text(json.dumps({"ranking": {"panel_scoring": {"kind": "xgb"}}}))
    status, detail = check_config(root, "renquant_104", "strategy_config.json")
    assert status == "error"


def test_main_exit_codes(tmp_path):
    root, sdir = _layout(tmp_path)
    (sdir / "good.json").write_text(json.dumps(_gated_meta()))
    (sdir / "bad.json").write_text(json.dumps({"kind": "xgb"}))
    _write_config(sdir, "strategy_config.json", "good.json")
    _write_config(sdir, "strategy_config.shadow.json", "bad.json")
    # all gated → 0
    assert main(["--repo-root", str(root), "--config", "strategy_config.json"]) == 0
    # one ungated → 1
    assert main([
        "--repo-root", str(root),
        "--config", "strategy_config.json",
        "--config", "strategy_config.shadow.json",
    ]) == 1
    # missing config → 2
    assert main(["--repo-root", str(root), "--config", "nope.json"]) == 2
