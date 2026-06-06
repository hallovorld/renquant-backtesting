from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from renquant_backtesting.analysis.assemble_track_b_verdict import (
    assemble_track_b_verdict,
)


def test_assembles_track_b_verdict_from_split_evidence() -> None:
    placebo = {
        "by_regime": {
            "BULL_CALM": {
                "mean_ic": 0.031,
                "n_dates": 42,
                "n_raw_rows": 1200,
                "hit_rate": 0.62,
            }
        },
        "shift_diagnostics": [
            {
                "shift_days": 120,
                "model_placebo_ic": 0.006,
                "aligned_real_ic": 0.030,
                "label_autocorr_ic": 0.010,
                "n_dates": 40,
                "n_rows": 1100,
            }
        ],
    }
    aa_shuffle = {
        "aa_mean": 0.029,
        "aa_std": 0.003,
        "aa_seeds": [0.027, 0.029, 0.031],
        "shuffle_ic": 0.001,
    }

    verdict = assemble_track_b_verdict([
        ("placebo.json", placebo),
        ("aa_shuffle.json", aa_shuffle),
    ])

    assert verdict["promotion_verdict"]["passed"] is True
    evidence = verdict["required_evidence"]
    assert evidence["bull_calm_per_regime_ic"]["mean_ic"] == 0.031
    assert evidence["shuffle"]["ic"] == 0.001
    assert evidence["aa"]["mean_ic"] == 0.029
    assert evidence["time_shift_placebo_120d"]["placebo_ic"] == 0.006
    assert evidence["time_shift_placebo_120d"]["threshold"] == 0.015


def test_assembler_handles_wf_gate_metadata_shape() -> None:
    payload = {
        "wf_gate_metadata": {
            "sanity_shuffled_ic": -0.002,
            "aa_mean": 0.026,
            "sanity_regime_ic": {
                "regimes": {
                    "BULL_CALM": {
                        "mean_ic": 0.024,
                        "n_dates": 55,
                        "eligible": True,
                    }
                }
            },
            "placebo_shift_diagnostics": [
                {
                    "shift_days": 120,
                    "ic": -0.004,
                    "aligned_real_ic": 0.020,
                    "n_dates": 50,
                    "n_rows": 1400,
                }
            ],
        }
    }

    verdict = assemble_track_b_verdict([("artifact.json", payload)])

    assert verdict["promotion_verdict"]["passed"] is True
    assert verdict["required_evidence"]["bull_calm_per_regime_ic"]["source"].endswith(
        "sanity_regime_ic.regimes.BULL_CALM"
    )
    assert verdict["required_evidence"]["time_shift_placebo_120d"]["source"].endswith(
        "#root1"
    )


def test_assembler_blocks_missing_and_failing_required_fields() -> None:
    payload = {
        "by_regime": {"BULL_CALM": {"mean_ic": 0.010}},
        "shuffle_ic": 0.012,
        "shift_diagnostics": [{"shift_days": 120, "model_placebo_ic": 0.009}],
    }

    verdict = assemble_track_b_verdict([("bad.json", payload)])

    assert verdict["promotion_verdict"]["passed"] is False
    reasons = verdict["promotion_verdict"]["blocked_reasons"]
    assert any("BULL_CALM per-regime IC" in reason for reason in reasons)
    assert any("shuffle IC" in reason for reason in reasons)
    assert any("A/A mean IC missing" in reason for reason in reasons)
    assert any("+120d time-shift placebo IC" in reason for reason in reasons)


def test_track_b_verdict_cli_writes_output(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({
        "by_regime": {"BULL_CALM": {"mean_ic": 0.030, "n_dates": 40}},
        "shuffle_ic": 0.0,
        "aa_mean": 0.025,
        "shift_diagnostics": [
            {"shift_days": 120, "model_placebo_ic": 0.002, "aligned_real_ic": 0.020}
        ],
    }))
    out = tmp_path / "verdict.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "renquant_backtesting.analysis.assemble_track_b_verdict",
            "--evidence",
            str(evidence),
            "--output",
            str(out),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
    )

    payload = json.loads(out.read_text())
    stdout_payload = json.loads(result.stdout)
    assert payload["promotion_verdict"]["recommendation"] == "PROMOTE"
    assert stdout_payload == payload
