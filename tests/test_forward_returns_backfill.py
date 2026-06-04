"""Forward-return backfill contracts for the lifted analysis CLI."""
from __future__ import annotations

import datetime as dt
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from renquant_backtesting.analysis import backfill_forward_returns as backfill


def _seed_parquet(cache_root: Path, ticker: str, closes: list[float]) -> None:
    import pandas as pd  # noqa: PLC0415

    dates = pd.bdate_range(start=dt.date(2026, 4, 1), periods=len(closes))
    df = pd.DataFrame(
        {
            "close": closes,
            "open": closes,
            "high": closes,
            "low": closes,
            "volume": [1_000_000] * len(closes),
        },
        index=dates,
    )
    df.index.name = "Date"
    out = cache_root / ticker / "1d.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)


def _seed_candidate_db(db_path: Path) -> None:
    from renquant_pipeline.kernel.persistence import (  # noqa: PLC0415
        get_connection,
        record_candidate_scores,
        record_pipeline_run,
    )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(
        {"persistence": {"enabled": True, "db_path": str(db_path)}},
    )
    assert conn is not None
    run_id = record_pipeline_run(
        conn,
        run_type="live",
        run_date=dt.date(2026, 4, 1),
        strategy="renquant_104",
    )
    candidate = SimpleNamespace(
        ticker="NVDA",
        raw_score=5.0,
        rank_score=0.5,
        rs_score=0.0,
        panel_score=0.5,
        mu=None,
        sigma=None,
    )
    record_candidate_scores(conn, run_id, [candidate], {}, selected_tickers={"NVDA"})
    conn.commit()
    conn.close()


def test_benchmark_pairs_helper_returns_missing_only(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "runs.db")
    conn.executescript(
        """
        CREATE TABLE pipeline_runs (run_id TEXT, run_date TEXT);
        CREATE TABLE ticker_forward_returns (
            as_of_date TEXT, ticker TEXT,
            fwd_1d REAL, fwd_5d REAL, fwd_10d REAL, fwd_20d REAL, fwd_60d REAL,
            PRIMARY KEY (as_of_date, ticker)
        );
        INSERT INTO pipeline_runs VALUES ('r1', '2025-01-02');
        INSERT INTO pipeline_runs VALUES ('r2', '2025-01-03');
        INSERT INTO ticker_forward_returns VALUES
            ('2025-01-02','SPY', 0.01, 0.02, 0.03, 0.04, 0.05);
        """
    )

    assert backfill._benchmark_pairs(conn, ["SPY"], None) == [
        ("2025-01-03", "SPY"),
    ]


def test_repo_root_backfill_computes_forward_returns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "umbrella"
    cache_root = repo_root / "data" / "ohlcv"
    db_path = repo_root / "data" / "runs.db"
    (repo_root / "backtesting" / "renquant_104").mkdir(parents=True)
    _seed_parquet(cache_root, "NVDA", [100.0 + i for i in range(65)])
    _seed_candidate_db(db_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backfill_forward_returns",
            "--repo-root",
            str(repo_root),
            "--db",
            "data/runs.db",
            "--cache-root",
            "data/ohlcv",
            "--benchmarks",
            "",
        ],
    )
    backfill.main()

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        """
        SELECT close_price, fwd_1d, fwd_5d, fwd_10d, fwd_20d, fwd_60d
          FROM ticker_forward_returns
        """
    ).fetchone()
    assert row is not None
    close, fwd_1d, fwd_5d, fwd_10d, fwd_20d, fwd_60d = row
    assert close == 100.0
    assert fwd_1d == pytest.approx(0.01)
    assert fwd_5d == pytest.approx(0.05)
    assert fwd_10d == pytest.approx(0.10)
    assert fwd_20d == pytest.approx(0.20)
    assert fwd_60d == pytest.approx(0.60)


def test_rows_needing_backfill_covers_score_distribution_mu(tmp_path: Path) -> None:
    """#204 B2: mu/sigma live in score_distribution, NOT candidate_scores.

    The QP Step-4 A/B replay loader joins score_distribution.(date,ticker)
    to ticker_forward_returns. Before this fix, _rows_needing_backfill only
    emitted candidate_scores (run_date,ticker), so the score_distribution
    mu/sigma rows on sim-run dates the backfill never visited had no
    forward return -> the loader returned 0 bars. This pins that the
    score_distribution (date,ticker) with mu populated is now in the
    backfill worklist.
    """
    import sqlite3

    from renquant_backtesting.analysis import backfill_forward_returns as backfill

    db = tmp_path / "sim.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE ticker_forward_returns (as_of_date DATE, ticker TEXT, "
        "fwd_1d REAL, fwd_5d REAL, fwd_10d REAL, fwd_20d REAL, fwd_60d REAL, "
        "PRIMARY KEY(as_of_date, ticker))"
    )
    conn.execute("CREATE TABLE candidate_scores (run_id TEXT, ticker TEXT)")
    conn.execute("CREATE TABLE pipeline_runs (run_id TEXT, run_date DATE)")
    conn.execute(
        "CREATE TABLE score_distribution (run_id TEXT, date TEXT, ticker TEXT, "
        "mu REAL, sigma REAL)"
    )
    # A score_distribution row with mu populated, on a date the backfill
    # would otherwise never visit (no candidate_scores entry for it).
    conn.execute(
        "INSERT INTO score_distribution VALUES ('r1', '2024-04-05', 'NVDA', 0.013, 0.069)"
    )
    # A score_distribution row with NULL mu must NOT be pulled in.
    conn.execute(
        "INSERT INTO score_distribution VALUES ('r2', '2024-04-06', 'AAPL', NULL, NULL)"
    )
    conn.commit()

    assert backfill._has_score_distribution_mu(conn) is True
    rows = backfill._rows_needing_backfill(conn, None)
    assert ("2024-04-05", "NVDA") in rows
    assert ("2024-04-06", "AAPL") not in rows  # NULL mu excluded
    conn.close()


def test_rows_needing_backfill_guards_missing_score_distribution(tmp_path: Path) -> None:
    """No score_distribution table -> UNION is skipped, no crash."""
    import sqlite3

    from renquant_backtesting.analysis import backfill_forward_returns as backfill

    db = tmp_path / "sim.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE ticker_forward_returns (as_of_date DATE, ticker TEXT, "
        "fwd_1d REAL, fwd_20d REAL, PRIMARY KEY(as_of_date, ticker))"
    )
    conn.execute("CREATE TABLE candidate_scores (run_id TEXT, ticker TEXT)")
    conn.execute("CREATE TABLE pipeline_runs (run_id TEXT, run_date DATE)")
    conn.commit()
    assert backfill._has_score_distribution_mu(conn) is False
    # must not raise
    rows = backfill._rows_needing_backfill(conn, None)
    assert rows == []
    conn.close()
