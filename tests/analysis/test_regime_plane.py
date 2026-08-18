"""Tests for the ONE production regime-label plane (orch#985 ranked item 1).

Hermetic: synthetic SPY + synthetic GMM artifact + minimal strategy config.
No production data is read; the committed-corpus test reads only files
committed in THIS repo and their manifest.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from renquant_backtesting.analysis import analyze_manifest_sanity_placebo as amsp
from renquant_backtesting.analysis import regime_plane as rp

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── fixtures ────────────────────────────────────────────────────────────────

def _synthetic_spy() -> pd.DataFrame:
    """400 calm-bull days, 30 crash days (−3%/d), 60 recovery days.

    Deterministic (no RNG): calm returns alternate +0.08%/+0.02%.
    """
    rets: list[float] = []
    for i in range(400):
        rets.append(0.0008 if i % 2 == 0 else 0.0002)
    rets += [-0.03] * 30
    rets += [0.004] * 60
    dates = pd.bdate_range("2016-01-04", periods=len(rets))
    close = 100.0
    rows = []
    for d, r in zip(dates, rets):
        close *= 1.0 + r
        rows.append({
            "date": d,
            "open": close * 0.999,
            "high": close * 1.004,
            "low": close * 0.996,
            "close": close,
            "volume": 1_000_000,
        })
    return pd.DataFrame(rows).set_index("date")


def _synthetic_gmm() -> dict:
    ident = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    return {
        # feature vector: [r10d, vol20_annualized, spy_adx, r_autocorr]
        "scaler_mean": [0.0, 0.0, 25.0, 0.0],
        "scaler_scale": [1.0, 1.0, 10.0, 1.0],
        "means": [
            [0.005, 0.10, 0.0, 0.0],   # BULL_CALM
            [0.000, 0.25, 0.0, 0.0],   # BULL_VOLATILE
            [-0.15, 0.45, 0.0, 0.0],   # BEAR
        ],
        "covariances": [ident, ident, ident],
        "weights": [1 / 3, 1 / 3, 1 / 3],
        "cluster_labels": ["BULL_CALM", "BULL_VOLATILE", "BEAR"],
    }


@pytest.fixture
def fixture_strategy_dir(tmp_path: Path) -> Path:
    sdir = tmp_path / "strategy"
    (sdir / "artifacts").mkdir(parents=True)
    (sdir / "artifacts" / "gmm.json").write_text(json.dumps(_synthetic_gmm()))
    (sdir / "strategy_config.json").write_text(json.dumps(
        {"regime": {"gmm_artifact": "gmm.json"}}
    ))
    return sdir


@pytest.fixture
def fixture_spy(tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    spy = _synthetic_spy()
    spy_dir = tmp_path / "data" / "ohlcv" / "SPY"
    spy_dir.mkdir(parents=True)
    spy_path = spy_dir / "1d.parquet"
    spy.to_parquet(spy_path)
    return spy, spy_path


# ── plane resolution ────────────────────────────────────────────────────────

def test_resolve_regime_plane_default_is_production(monkeypatch):
    monkeypatch.delenv(rp.REGIME_PLANE_ENV, raising=False)
    assert rp.resolve_regime_plane() == rp.PLANE_PRODUCTION


def test_resolve_regime_plane_escape_hatch(monkeypatch):
    monkeypatch.setenv(rp.REGIME_PLANE_ENV, rp.PLANE_LEGACY_STATELESS)
    assert rp.resolve_regime_plane() == rp.PLANE_LEGACY_STATELESS


def test_resolve_regime_plane_rejects_typo(monkeypatch):
    monkeypatch.setenv(rp.REGIME_PLANE_ENV, "prod")
    with pytest.raises(ValueError, match="RENQUANT_REGIME_PLANE"):
        rp.resolve_regime_plane()


# ── the accessor ────────────────────────────────────────────────────────────

def test_accessor_deterministic_and_bear_on_crash(fixture_strategy_dir, fixture_spy):
    spy, _ = fixture_spy
    dates = spy.index[380:]  # tail: calm end + crash + recovery
    a = rp.production_regime_labels(dates, spy, strategy_dir=fixture_strategy_dir)
    b = rp.production_regime_labels(dates, spy, strategy_dir=fixture_strategy_dir)
    pd.testing.assert_frame_equal(a, b)
    assert not a.empty
    # the −3%/day crash segment must be labeled BEAR by the hard override
    crash_dates = spy.index[405:425]
    crash = a[a["date"].isin(crash_dates)]
    assert (crash["regime"] == "BEAR").all(), crash[["date", "regime"]]


def test_wf_sanity_wrapper_is_the_accessor(
    monkeypatch, tmp_path, fixture_strategy_dir, fixture_spy,
):
    """build_regime_series (the WF sanity leg's entry) must emit labels
    IDENTICAL to production_regime_labels — one code path, one plane."""
    from renquant_backtesting.wf_gate import runner as runner_mod

    spy, _ = fixture_spy
    monkeypatch.setattr(runner_mod, "REPO", tmp_path)
    dates = spy.index[390:440]
    via_wrapper = amsp.build_regime_series(dates, strategy_dir=fixture_strategy_dir)
    direct = rp.production_regime_labels(
        dates, spy, strategy_dir=fixture_strategy_dir,
    )
    pd.testing.assert_frame_equal(via_wrapper, direct)


# ── the publisher ───────────────────────────────────────────────────────────

def _run_publisher(out_dir: Path, spy_path: Path, strategy_dir: Path,
                   name: str) -> tuple[Path, Path]:
    out_csv = out_dir / f"{name}.csv"
    out_manifest = out_dir / f"{name}.manifest.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")],
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "publish_production_regime_labels.py"),
            "--spy", str(spy_path),
            "--strategy-dir", str(strategy_dir),
            "--start", "2017-06-01",
            "--out-csv", str(out_csv),
            "--out-manifest", str(out_manifest),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return out_csv, out_manifest


def test_publisher_two_runs_identical(tmp_path, fixture_strategy_dir, fixture_spy):
    _, spy_path = fixture_spy
    csv1, man1 = _run_publisher(tmp_path, spy_path, fixture_strategy_dir, "run1")
    csv2, man2 = _run_publisher(tmp_path, spy_path, fixture_strategy_dir, "run2")
    assert csv1.read_bytes() == csv2.read_bytes()
    m1 = json.loads(man1.read_text())
    m2 = json.loads(man2.read_text())
    assert m1["series_sha256"] == m2["series_sha256"]
    assert m1["row_count"] == m2["row_count"] > 0
    # fixture window does not cover the memo replica span
    assert m1["memo_985_crosscheck"]["evaluated"] is False


def test_committed_corpus_matches_manifest():
    corpus = rp.repo_corpus_path()
    manifest_path = rp.repo_manifest_path()
    assert corpus.exists(), "committed corpus missing — run the publisher"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert rp.sha256_bytes_of(corpus) == manifest["series_sha256"]
    df = rp.load_production_regime_corpus(corpus)
    assert len(df) == manifest["row_count"]
    assert str(df["date"].min().date()) == manifest["date_start"]
    assert str(df["date"].max().date()) == manifest["date_end"]
    # the publish-time replica validation against the #985 memo facts
    cc = manifest["memo_985_crosscheck"]
    assert cc["evaluated"] is True
    assert cc["exact_match"] is True
    assert cc["expected"] == {
        "n_days": 2459, "bear_days": 413, "bear_episodes": 57,
    }


# ── site (i): cut_market_context ────────────────────────────────────────────

def _patch_runner_tree(monkeypatch, tmp_path, fixture_strategy_dir, spy_path):
    from renquant_backtesting.wf_gate import runner as runner_mod

    # REPO/data/ohlcv/SPY/1d.parquet must exist under the patched REPO
    repo = spy_path.parents[3]
    monkeypatch.setattr(runner_mod, "REPO", repo)
    monkeypatch.setattr(runner_mod, "STRATEGY_DIR", fixture_strategy_dir)
    return runner_mod


def test_cut_market_context_defaults_to_production_plane(
    monkeypatch, tmp_path, fixture_strategy_dir, fixture_spy,
):
    spy, spy_path = fixture_spy
    runner_mod = _patch_runner_tree(monkeypatch, tmp_path, fixture_strategy_dir, spy_path)
    monkeypatch.delenv(rp.REGIME_PLANE_ENV, raising=False)
    start, end = str(spy.index[400].date()), str(spy.index[460].date())
    ctx = runner_mod.cut_market_context(start, end)
    assert ctx["regime_plane"] == rp.PLANE_PRODUCTION
    assert sum(ctx["regime_counts"].values()) > 0
    assert "BEAR" in ctx["regime_counts"]  # crash inside the window
    # legacy stateless keys are NOT emitted on the production plane
    assert "hmm_regime_counts" not in ctx
    assert "spy_grid_regime_counts" not in ctx


def test_cut_market_context_legacy_escape_hatch(
    monkeypatch, tmp_path, fixture_strategy_dir, fixture_spy,
):
    spy, spy_path = fixture_spy
    runner_mod = _patch_runner_tree(monkeypatch, tmp_path, fixture_strategy_dir, spy_path)
    monkeypatch.setenv(rp.REGIME_PLANE_ENV, rp.PLANE_LEGACY_STATELESS)
    start, end = str(spy.index[400].date()), str(spy.index[460].date())
    ctx = runner_mod.cut_market_context(start, end)
    assert ctx["regime_plane"] == rp.PLANE_LEGACY_STATELESS
    # pre-consolidation keys are reproduced verbatim under the hatch
    assert "hmm_regime_counts" in ctx
    assert "spy_grid_regime_counts" in ctx
    assert ctx["regime_counts"] == ctx["hmm_regime_counts"]
