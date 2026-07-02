"""Session-deduplication contracts for weekend/holiday as-of forward returns.

Codex review on #60: a Friday/Saturday/Sunday decision-date trio all resolve
to the same Friday base close and forward-return path — one real market
realization, not three independent observations. These tests prove the
deduplication machinery in `session_resolution.py` collapses such a trio to
exactly one row, end to end through the real backfill + analysis CLI path,
not just at the unit level.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from renquant_backtesting.analysis import backfill_forward_returns as backfill
from renquant_backtesting.analysis.session_resolution import (
    annotate_base_sessions,
    dedupe_by_session,
    is_non_session_run,
    resolve_base_session_date,
)


def _seed_parquet(cache_root: Path, ticker: str, closes: list[float]) -> None:
    dates = pd.bdate_range(start=dt.date(2026, 4, 1), periods=len(closes))
    df = pd.DataFrame(
        {
            "close": closes, "open": closes, "high": closes, "low": closes,
            "volume": [1_000_000] * len(closes),
        },
        index=dates,
    )
    df.index.name = "Date"
    out = cache_root / ticker / "1d.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)


def _seed_candidate_db(db_path: Path, run_dates: list[dt.date], ticker: str = "NVDA") -> None:
    from renquant_pipeline.kernel.persistence import (
        get_connection, record_candidate_scores, record_pipeline_run,
    )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection({"persistence": {"enabled": True, "db_path": str(db_path)}})
    assert conn is not None
    for run_date in run_dates:
        run_id = record_pipeline_run(
            conn, run_type="live", run_date=run_date, strategy="renquant_104",
        )
        candidate = SimpleNamespace(
            ticker=ticker, raw_score=5.0, rank_score=0.5, rs_score=0.0,
            panel_score=0.5, mu=None, sigma=None,
        )
        record_candidate_scores(conn, run_id, [candidate], {}, selected_tickers={ticker})
    conn.commit()
    conn.close()


# ------------------------------------------------------------- unit level


def test_resolve_base_session_date_weekend_trio_shares_one_base() -> None:
    """Friday, Saturday, Sunday all resolve to the same Friday session."""
    closes = [100.0 + i for i in range(65)]
    dates = pd.bdate_range(start=dt.date(2026, 4, 1), periods=len(closes))
    df = pd.DataFrame({"close": closes}, index=dates)

    friday = resolve_base_session_date(dt.date(2026, 4, 3), df)
    saturday = resolve_base_session_date(dt.date(2026, 4, 4), df)
    sunday = resolve_base_session_date(dt.date(2026, 4, 5), df)

    assert friday == saturday == sunday == dt.date(2026, 4, 3)
    assert is_non_session_run(dt.date(2026, 4, 3), friday) is False  # own session
    assert is_non_session_run(dt.date(2026, 4, 4), saturday) is True
    assert is_non_session_run(dt.date(2026, 4, 5), sunday) is True


def test_dedupe_by_session_collapses_weekend_trio() -> None:
    """A Fri/Sat/Sun trio for one ticker collapses to a single row."""
    df = pd.DataFrame({
        "run_date": pd.to_datetime(["2026-04-03", "2026-04-04", "2026-04-05"]),
        "ticker": ["NVDA", "NVDA", "NVDA"],
        "base_session_date": [dt.date(2026, 4, 3)] * 3,
        "fwd_20d": [0.20, 0.20, 0.20],
    })
    deduped = dedupe_by_session(df, "base_session_date", ["ticker"])
    assert len(deduped) == 1
    assert deduped.iloc[0]["run_date"] == pd.Timestamp("2026-04-03")


def test_dedupe_by_session_does_not_collapse_distinct_tickers_or_sessions() -> None:
    """Distinct tickers, and distinct genuine sessions, must NOT collapse."""
    df = pd.DataFrame({
        "run_date": pd.to_datetime(
            ["2026-04-03", "2026-04-04", "2026-04-03", "2026-04-10"]
        ),
        "ticker": ["NVDA", "NVDA", "AAPL", "NVDA"],
        "base_session_date": [
            dt.date(2026, 4, 3), dt.date(2026, 4, 3),
            dt.date(2026, 4, 3), dt.date(2026, 4, 10),
        ],
    })
    deduped = dedupe_by_session(df, "base_session_date", ["ticker"])
    # NVDA@04-03 collapses with NVDA@04-04 (same session); AAPL@04-03 and
    # NVDA@04-10 are each their own distinct (session, ticker) key.
    assert len(deduped) == 3


def test_annotate_base_sessions_uncachable_ticker_fails_open_to_own_date(
    tmp_path: Path,
) -> None:
    """A ticker with no cached parquet dedupes only with itself, never
    silently collides with another uncachable row under a shared null key."""
    df = pd.DataFrame({
        "run_date": pd.to_datetime(["2026-04-03", "2026-04-04"]),
        "ticker": ["GHOST", "GHOST"],
    })
    out = annotate_base_sessions(
        df, date_col="run_date", ticker_col="ticker", cache_root=tmp_path,
    )
    assert out["non_session_run"].tolist() == [False, False]
    # Each row's base_session_date is its OWN date, not a shared sentinel —
    # so they do not collapse together under dedupe_by_session.
    assert out["base_session_date"].tolist() == list(
        pd.to_datetime(["2026-04-03", "2026-04-04"])
    )


# ------------------------------------------------------------- end to end


def test_friday_weekend_trio_collapses_to_one_admissible_observation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The exact case Codex's review names: a Fri + Sat + Sun decision-date
    trio must produce THREE raw storage rows but exactly ONE unique-session
    admissible observation once deduplicated — proving downstream sample
    counts do not inflate, not merely that the weekend rows join."""
    repo_root = tmp_path / "umbrella"
    cache_root = repo_root / "data" / "ohlcv"
    db_path = repo_root / "data" / "runs.db"
    (repo_root / "backtesting" / "renquant_104").mkdir(parents=True)
    _seed_parquet(cache_root, "NVDA", [100.0 + i for i in range(65)])
    _seed_candidate_db(
        db_path,
        run_dates=[dt.date(2026, 4, 3), dt.date(2026, 4, 4), dt.date(2026, 4, 5)],
    )

    monkeypatch.setattr(
        sys, "argv",
        ["backfill_forward_returns", "--repo-root", str(repo_root),
         "--db", "data/runs.db", "--cache-root", "data/ohlcv", "--benchmarks", ""],
    )
    backfill.main()

    conn = sqlite3.connect(db_path)
    raw = pd.read_sql(
        """
        SELECT ps.run_date, cs.ticker, tfr.fwd_20d
          FROM candidate_scores cs
          JOIN pipeline_runs ps ON ps.run_id = cs.run_id
          JOIN ticker_forward_returns tfr
            ON tfr.as_of_date = ps.run_date AND tfr.ticker = cs.ticker
        """,
        conn,
    )
    conn.close()

    # Storage coverage: all 3 decision dates joined a forward outcome.
    assert len(raw) == 3
    # All 3 share the identical Friday-based forward return — proving they
    # are the same market realization, not independent draws.
    assert raw["fwd_20d"].nunique() == 1

    raw["run_date"] = pd.to_datetime(raw["run_date"])
    annotated = annotate_base_sessions(
        raw, date_col="run_date", ticker_col="ticker", cache_root=cache_root,
    )
    assert annotated["non_session_run"].sum() == 2  # Sat + Sun are non-session

    admissible = dedupe_by_session(annotated, "base_session_date", ["ticker"])
    assert len(admissible) == 1, (
        "a Fri/Sat/Sun decision-date trio must collapse to exactly one "
        "statistically admissible observation"
    )


def test_friday_only_cohort_is_unaffected_by_dedup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A cohort of genuine, distinct trading-day decisions must NOT be
    collapsed — dedup only removes true weekend/holiday duplicates."""
    repo_root = tmp_path / "umbrella"
    cache_root = repo_root / "data" / "ohlcv"
    db_path = repo_root / "data" / "runs.db"
    (repo_root / "backtesting" / "renquant_104").mkdir(parents=True)
    _seed_parquet(cache_root, "NVDA", [100.0 + i for i in range(65)])
    _seed_candidate_db(
        db_path,
        run_dates=[dt.date(2026, 4, 1), dt.date(2026, 4, 2), dt.date(2026, 4, 3)],
    )

    monkeypatch.setattr(
        sys, "argv",
        ["backfill_forward_returns", "--repo-root", str(repo_root),
         "--db", "data/runs.db", "--cache-root", "data/ohlcv", "--benchmarks", ""],
    )
    backfill.main()

    conn = sqlite3.connect(db_path)
    raw = pd.read_sql(
        """
        SELECT ps.run_date, cs.ticker
          FROM candidate_scores cs
          JOIN pipeline_runs ps ON ps.run_id = cs.run_id
          JOIN ticker_forward_returns tfr
            ON tfr.as_of_date = ps.run_date AND tfr.ticker = cs.ticker
        """,
        conn,
    )
    conn.close()
    raw["run_date"] = pd.to_datetime(raw["run_date"])

    annotated = annotate_base_sessions(
        raw, date_col="run_date", ticker_col="ticker", cache_root=cache_root,
    )
    assert annotated["non_session_run"].sum() == 0

    admissible = dedupe_by_session(annotated, "base_session_date", ["ticker"])
    assert len(admissible) == 3, "3 genuine trading-day decisions must remain 3 rows"
