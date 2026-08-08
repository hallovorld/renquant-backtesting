"""orch#905 wiring: the WF replay persists its test cross-sections.

Synthetic panel, one tiny fold — the point is the emission contract, not the
model. Every test builds its own frame; nothing touches real data dirs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from renquant_backtesting.wf_gate import wf_sanity_paired as wsp


def _tiny_panel(n_tickers=30, seed=0) -> tuple[pd.DataFrame, list[str]]:
    """Train dates 2016-2018 (enough rows), test dates early 2019."""
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    tr_dates = pd.bdate_range("2016-01-04", periods=40)
    te_dates = pd.bdate_range("2019-02-04", periods=5)
    rows = []
    for d in list(tr_dates) + list(te_dates):
        for t in tickers:
            f1, f2 = rng.normal(), rng.normal()
            rows.append({"ticker": t, "date": d, "f1": f1, "f2": f2,
                         wsp.LABEL: f1 * 0.1 + rng.normal() * 0.5})
    return pd.DataFrame(rows), ["f1", "f2"]


_CUT = [("2016-01-01", "2018-12-31", "2019-02-01", "2019-12-31")]


def test_default_behaviour_is_unchanged_without_a_sink():
    panel, feats = _tiny_panel()
    a = wsp.run_wf(panel, feats, cuts=_CUT, seed=42)
    b = wsp.run_wf(panel, feats, cuts=_CUT, seed=42)
    assert a == b and len(a) == 1 and not np.isnan(a[0])


def test_emission_writes_one_pair_per_test_date(tmp_path):
    panel, feats = _tiny_panel()
    sink = wsp.ServedMatrixEmission(tmp_path, run_id="wfrun-1")
    ics = wsp.run_wf(panel, feats, cuts=_CUT, seed=42, served_sink=sink)

    day_dirs = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert len(day_dirs) == 5  # the 5 synthetic test dates
    assert sink.n_files == 5
    one = tmp_path / day_dirs[0]
    parquet = list(one.glob("*.parquet"))
    sidecar = list(one.glob("*.json"))
    assert len(parquet) == 1 and len(sidecar) == 1
    assert parquet[0].stem == "wf_replay_panel__wfrun-1"

    frame = pd.read_parquet(parquet[0])
    assert sorted(frame.columns) == ["score", "ticker"]
    assert len(frame) == 30  # full cross-section, every ticker

    manifest = json.loads(sidecar[0].read_text())
    assert manifest["lane"] == "wf_replay_panel"
    assert manifest["run_id"] == "wfrun-1"
    assert manifest["as_of_date"] == day_dirs[0]
    assert manifest["replay"]["fold_train_end"] == "2018-12-31"
    assert manifest["replay"]["kind"] == "wf_sanity_paired.run_wf"

    # emission must not perturb the returned ICs
    assert ics == wsp.run_wf(panel, feats, cuts=_CUT, seed=42)


def test_emitted_scores_match_the_scored_cross_section(tmp_path):
    """The persisted rows ARE the predictions cs_ic consumed — per ticker."""
    panel, feats = _tiny_panel()
    sink = wsp.ServedMatrixEmission(tmp_path, run_id="wfrun-2")
    wsp.run_wf(panel, feats, cuts=_CUT, seed=42, served_sink=sink)
    files = sorted(tmp_path.rglob("*.parquet"))
    total = sum(len(pd.read_parquet(f)) for f in files)
    te_rows = panel[(panel["date"] >= "2019-02-01") & (panel["date"] <= "2019-12-31")]
    assert total == len(te_rows.dropna(subset=[wsp.LABEL]))


def test_perturbed_arms_refuse_a_sink(tmp_path):
    panel, feats = _tiny_panel()
    sink = wsp.ServedMatrixEmission(tmp_path, run_id="wfrun-3")
    with pytest.raises(ValueError, match="placebo"):
        wsp.run_wf(panel, feats, cuts=_CUT, shuffle=True, served_sink=sink)
    with pytest.raises(ValueError, match="placebo"):
        wsp.run_wf(panel, feats, cuts=_CUT, shift_days=60, served_sink=sink)
    assert list(tmp_path.iterdir()) == []  # nothing was written


def test_lane_must_carry_the_replay_prefix(tmp_path):
    with pytest.raises(ValueError, match="wf_replay"):
        wsp.ServedMatrixEmission(tmp_path, run_id="x", lane="alpaca")
    # a replay row must never be mistakable for a served-live row
    wsp.ServedMatrixEmission(tmp_path, run_id="x", lane="wf_replay_momentum")
