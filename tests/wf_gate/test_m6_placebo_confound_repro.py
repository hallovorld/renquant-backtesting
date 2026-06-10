"""M6 reproduction tests — the placebo↔autocorr discriminator.

Pure-logic coverage for ``repro_m6_placebo_confound`` so the verdict's decisive
statistic (``corr(placebo_ic, label_autocorr_ic)`` across regimes) is pinned. The
heavy autocorr-from-parquet path is exercised by the documented reproduction
command, not in CI (it needs the multi-hundred-MB panel).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from renquant_backtesting.analysis.repro_m6_placebo_confound import (
    cross_sectional_autocorr,
    run_regime,
)


def _stamped_artifact(per_regime: dict[str, dict]) -> dict:
    """Minimal artifact shaped like the gate's stamped Layer-1a metadata."""
    return {
        "metadata": {
            "wf_gate_metadata": {
                "model_placebo_profile": {
                    "label_col": "fwd_60d_excess",
                    "pooled": {
                        "2x": {
                            "aligned_real_ic": 0.0570,
                            "placebo_ic": 0.0359,
                            "genuine_ic": 0.0211,
                            "label_autocorr_ic": 0.0390,
                            "n_dates": 388,
                        }
                    },
                    "per_regime": per_regime,
                }
            }
        }
    }


def test_confound_signature_correlates_near_plus_one(tmp_path):
    """The real failing artifact's shape: placebo tracks label autocorr (r≈+1)."""
    per_regime = {
        "BEAR": {"2x": {"placebo_ic": -0.0206, "label_autocorr_ic": 0.0223,
                        "aligned_real_ic": 0.2719, "genuine_ic": 0.2925, "n_dates": 49}},
        "BULL_CALM": {"2x": {"placebo_ic": 0.0413, "label_autocorr_ic": 0.0422,
                             "aligned_real_ic": 0.0302, "genuine_ic": -0.0112, "n_dates": 302}},
        "CHOPPY": {"2x": {"placebo_ic": 0.0433, "label_autocorr_ic": 0.0403,
                          "aligned_real_ic": -0.0097, "genuine_ic": -0.0530, "n_dates": 26}},
    }
    art = tmp_path / "staging.json"
    art.write_text(json.dumps(_stamped_artifact(per_regime)))
    result = run_regime(art)
    # Confound: placebo is regime-for-regime explained by the label's autocorr.
    assert result["corr_placebo_autocorr"] > 0.95
    assert {r["regime"] for r in result["regimes"]} == {"BEAR", "BULL_CALM", "CHOPPY"}


def test_leakage_signature_would_anticorrelate(tmp_path):
    """A genuine leak shows placebo HIGH where label autocorr is LOW (r<0)."""
    per_regime = {
        "A": {"2x": {"placebo_ic": 0.08, "label_autocorr_ic": 0.01,
                     "aligned_real_ic": 0.05, "genuine_ic": -0.03, "n_dates": 50}},
        "B": {"2x": {"placebo_ic": 0.05, "label_autocorr_ic": 0.04,
                     "aligned_real_ic": 0.05, "genuine_ic": 0.0, "n_dates": 50}},
        "C": {"2x": {"placebo_ic": 0.02, "label_autocorr_ic": 0.08,
                     "aligned_real_ic": 0.05, "genuine_ic": 0.03, "n_dates": 50}},
    }
    art = tmp_path / "leak.json"
    art.write_text(json.dumps(_stamped_artifact(per_regime)))
    result = run_regime(art)
    assert result["corr_placebo_autocorr"] < 0.0


def test_low_date_regimes_are_excluded(tmp_path):
    per_regime = {
        "BEAR": {"2x": {"placebo_ic": -0.02, "label_autocorr_ic": 0.02,
                        "aligned_real_ic": 0.27, "genuine_ic": 0.29, "n_dates": 49}},
        "THIN": {"2x": {"placebo_ic": 0.12, "label_autocorr_ic": 0.02,
                        "aligned_real_ic": -0.005, "genuine_ic": -0.13, "n_dates": 11}},
    }
    art = tmp_path / "thin.json"
    art.write_text(json.dumps(_stamped_artifact(per_regime)))
    result = run_regime(art, min_dates=25)
    regimes = {r["regime"] for r in result["regimes"]}
    assert regimes == {"BEAR"}  # THIN (11 dates) excluded


def test_cross_sectional_autocorr_recovers_known_persistence():
    """Synthetic AR(1)-ish panel: positive serial corr at lag-1, ~0 at lag-10."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2026-01-01", periods=80, freq="D")
    tickers = [f"T{i}" for i in range(40)]
    base = {t: rng.normal(0, 1) for t in tickers}
    rows = []
    prev = {t: rng.normal() for t in tickers}
    for d in dates:
        for t in tickers:
            # persistent cross-sectional ordering with small daily noise
            val = 0.9 * prev[t] + 0.1 * base[t] + 0.05 * rng.normal()
            prev[t] = val
            rows.append({"date": d, "ticker": t, "lab": val})
    df = pd.DataFrame(rows)
    ac1, n1 = cross_sectional_autocorr(df, "lab", 1, min_names=20)
    ac10, _ = cross_sectional_autocorr(df, "lab", 10, min_names=20)
    assert n1 > 0
    assert ac1 > 0.5  # strong lag-1 persistence
    assert ac1 > ac10  # decays with lag
    assert math.isfinite(ac10)
