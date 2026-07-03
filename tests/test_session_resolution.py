"""Session-resolution contracts for weekend/holiday as-of forward returns.

#60 review rounds 2-3: a Friday/Saturday/Sunday decision-date trio all
resolve to the same Friday base close and forward-return path — one real
market realization, not three independent observations. These tests prove,
end to end through the real backfill + consumer paths: exact re-recordings
of one decision collapse; genuinely DIFFERENT decisions sharing a session
are retained and weighted so the shared realization counts once (not
arbitrarily discarded); exchange holidays are distinguished from weekends
via the shared NYSE calendar; session identity is a pure date function
(OHLCV cache coverage cannot change it) and is marked unresolved when the
calendar is unavailable; and a truncated parquet cannot fabricate a
weeks-stale as-of base.
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
from renquant_backtesting.analysis import session_resolution
from renquant_backtesting.analysis.session_resolution import (
    add_session_weights,
    annotate_base_sessions,
    classify_date,
    collapse_rerecorded_decisions,
    dedupe_by_session,
    nyse_sessions,
    session_key,
)

# Hand-built session calendar (deterministic, no calendar package needed):
# plain weekdays over Q2 2026. Weekend logic is identical to the real NYSE
# calendar; real-holiday behavior is covered by the importorskip tests below.
WEEKDAY_SESSIONS = pd.bdate_range(start="2026-04-01", end="2026-06-30")


def _seed_parquet(
    cache_root: Path,
    ticker: str,
    closes: list[float],
    index: "pd.DatetimeIndex | None" = None,
) -> None:
    dates = index if index is not None else pd.bdate_range(
        start=dt.date(2026, 4, 1), periods=len(closes),
    )
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


def _seed_candidate_db(
    db_path: Path,
    run_dates: list[dt.date],
    ticker: str = "NVDA",
    rank_scores: list[float] | None = None,
) -> None:
    from renquant_pipeline.kernel.persistence import (  # noqa: PLC0415
        get_connection, record_candidate_scores, record_pipeline_run,
    )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection({"persistence": {"enabled": True, "db_path": str(db_path)}})
    assert conn is not None
    for i, run_date in enumerate(run_dates):
        run_id = record_pipeline_run(
            conn, run_type="live", run_date=run_date, strategy="renquant_104",
        )
        candidate = SimpleNamespace(
            ticker=ticker, raw_score=5.0,
            rank_score=rank_scores[i] if rank_scores else 0.5,
            rs_score=0.0, panel_score=0.5, mu=None, sigma=None,
        )
        record_candidate_scores(conn, run_id, [candidate], {}, selected_tickers={ticker})
    conn.commit()
    conn.close()


def _run_backfill(monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> None:
    monkeypatch.setattr(
        sys, "argv",
        ["backfill_forward_returns", "--repo-root", str(repo_root),
         "--db", "data/runs.db", "--cache-root", "data/ohlcv", "--benchmarks", ""],
    )
    backfill.main()


# ------------------------------------------------------------- unit level


def test_session_key_weekend_trio_shares_one_session() -> None:
    """Friday, Saturday, Sunday all key to the same Friday session — with an
    explicit calendar AND under the weekday fallback (sessions=None)."""
    for sessions in (WEEKDAY_SESSIONS, None):
        assert session_key(dt.date(2026, 4, 10), sessions) == dt.date(2026, 4, 10)
        assert session_key(dt.date(2026, 4, 11), sessions) == dt.date(2026, 4, 10)
        assert session_key(dt.date(2026, 4, 12), sessions) == dt.date(2026, 4, 10)


def test_classify_date_weekend_vs_session() -> None:
    assert classify_date(dt.date(2026, 4, 10), WEEKDAY_SESSIONS) == "session"
    assert classify_date(dt.date(2026, 4, 11), WEEKDAY_SESSIONS) == "weekend"
    assert classify_date(dt.date(2026, 4, 12), WEEKDAY_SESSIONS) == "weekend"
    # Weekends are detected even without any calendar (degraded fallback).
    assert classify_date(dt.date(2026, 4, 11), None) == "weekend"


def test_classify_and_key_weekday_holiday_with_explicit_calendar() -> None:
    """A weekday the calendar excludes is a holiday and keys to the prior
    session — hand-built calendar variant (no calendar package needed)."""
    sessions = WEEKDAY_SESSIONS.drop(pd.Timestamp("2026-05-25"))  # Memorial Day
    assert classify_date(dt.date(2026, 5, 25), sessions) == "holiday"
    assert session_key(dt.date(2026, 5, 25), sessions) == dt.date(2026, 5, 22)
    # Without a calendar, a weekday holiday is NOT detectable — degraded to
    # 'session' (and the backfill then fails closed, writing no as-of row).
    assert classify_date(dt.date(2026, 5, 25), None) == "session"
    assert session_key(dt.date(2026, 5, 25), None) == dt.date(2026, 5, 25)


def test_real_nyse_calendar_distinguishes_holidays() -> None:
    """The shared NYSE calendar (pandas_market_calendars, as used by
    renquant-pipeline) marks real 2026 exchange holidays."""
    pytest.importorskip("pandas_market_calendars")
    sessions = nyse_sessions(dt.date(2026, 4, 1), dt.date(2026, 6, 1))
    assert sessions is not None
    assert classify_date(dt.date(2026, 4, 3), sessions) == "holiday"   # Good Friday
    assert classify_date(dt.date(2026, 5, 25), sessions) == "holiday"  # Memorial Day
    assert classify_date(dt.date(2026, 4, 25), sessions) == "weekend"
    assert classify_date(dt.date(2026, 4, 24), sessions) == "session"
    # Saturday after Good Friday keys back to Thursday — two days, one hop.
    assert session_key(dt.date(2026, 4, 4), sessions) == dt.date(2026, 4, 2)


def test_annotate_base_sessions_adds_key_flag_and_kind() -> None:
    sessions = WEEKDAY_SESSIONS.drop(pd.Timestamp("2026-05-25"))
    df = pd.DataFrame({
        "run_date": pd.to_datetime(
            ["2026-04-10", "2026-04-11", "2026-05-25", "2026-04-13"]
        ),
        "ticker": ["NVDA"] * 4,
    })
    out = annotate_base_sessions(df, date_col="run_date", sessions=sessions)
    assert out["base_session_date"].tolist() == list(
        pd.to_datetime(["2026-04-10", "2026-04-10", "2026-05-22", "2026-04-13"])
    )
    assert out["non_session_run"].tolist() == [False, True, True, False]
    assert out["non_session_kind"].tolist() == [None, "weekend", "holiday", None]


def test_dedupe_by_session_collapses_weekend_trio_keeping_session_row() -> None:
    """A Fri/Sat/Sun trio for one ticker collapses to a single row, and the
    kept row is deterministically the session-dated (Friday) one."""
    df = pd.DataFrame({
        "run_date": pd.to_datetime(["2026-04-12", "2026-04-11", "2026-04-10"]),
        "ticker": ["NVDA", "NVDA", "NVDA"],
        "base_session_date": [pd.Timestamp("2026-04-10")] * 3,
        "non_session_run": [True, True, False],
        "fwd_20d": [0.20, 0.20, 0.20],
    })
    deduped = dedupe_by_session(df, "base_session_date", ["ticker"])
    assert len(deduped) == 1
    assert deduped.iloc[0]["run_date"] == pd.Timestamp("2026-04-10")


def test_dedupe_by_session_does_not_collapse_distinct_tickers_or_sessions() -> None:
    """Distinct tickers, and distinct genuine sessions, must NOT collapse."""
    df = pd.DataFrame({
        "run_date": pd.to_datetime(
            ["2026-04-10", "2026-04-11", "2026-04-10", "2026-04-17"]
        ),
        "ticker": ["NVDA", "NVDA", "AAPL", "NVDA"],
        "base_session_date": pd.to_datetime(
            ["2026-04-10", "2026-04-10", "2026-04-10", "2026-04-17"]
        ),
    })
    deduped = dedupe_by_session(df, "base_session_date", ["ticker"])
    # NVDA@04-10 collapses with NVDA@04-11 (same session); AAPL@04-10 and
    # NVDA@04-17 are each their own distinct (session, ticker) key.
    assert len(deduped) == 3


def test_compute_row_staleness_guard_rejects_truncated_parquet() -> None:
    """A truncated parquet (de-watchlisted ticker whose feed stopped) must
    NOT fabricate a weeks-stale as-of base — the base bar must be at least
    the canonical previous NYSE session. Pre-fix behavior was fail-closed
    (no row); the guard restores that for stale data."""
    closes = [100.0 + i for i in range(23)]
    dates = pd.bdate_range(start=dt.date(2026, 4, 1), periods=len(closes))
    df = pd.DataFrame({"close": closes}, index=dates)  # ends Fri 2026-05-01

    for sessions in (WEEKDAY_SESSIONS, None):
        # Weekday decision weeks after the last bar: no fabricated row.
        assert backfill._compute_row(dt.date(2026, 6, 5), "DOCU", df, sessions) is None
        # Weekend decision weeks after the last bar: also refused.
        assert backfill._compute_row(dt.date(2026, 6, 6), "DOCU", df, sessions) is None
        # A weekend directly after the last bar (base = that Friday bar)
        # still resolves as-of; one MISSING session in between would not.
        row = backfill._compute_row(dt.date(2026, 5, 2), "DOCU", df, sessions)
        assert row is not None and row["close_price"] == closes[-1]
        assert backfill._compute_row(dt.date(2026, 5, 9), "DOCU", df, sessions) is None


def test_distinct_same_session_decisions_are_retained_and_weighted() -> None:
    """#60 review round 3: equal outcome identity does not imply equal
    decision identity. Exact re-recordings of one decision collapse;
    genuinely different decisions sharing a session are RETAINED, each
    carrying 1/cluster-size weight so the shared realization contributes
    exactly one observation in aggregate."""
    df = pd.DataFrame({
        "run_date": pd.to_datetime(
            ["2026-04-10", "2026-04-10", "2026-04-11", "2026-04-10"]
        ),
        "ticker": ["NVDA", "NVDA", "NVDA", "AAPL"],
        # rows 0+2: same decision content re-recorded Fri + Sat -> collapse;
        # row 1: a same-day re-run with a DIFFERENT score -> retained.
        "rank_score": [0.7, 0.5, 0.7, 0.9],
        "base_session_date": pd.to_datetime(
            ["2026-04-10", "2026-04-10", "2026-04-10", "2026-04-10"]
        ),
        "non_session_run": [False, False, True, False],
    })
    collapsed = collapse_rerecorded_decisions(
        df, "base_session_date", ["ticker"], ["rank_score"],
    )
    assert len(collapsed) == 3  # 0.7-decision once, 0.5-decision, AAPL
    assert sorted(collapsed[collapsed.ticker == "NVDA"]["rank_score"]) == [0.5, 0.7]
    # The kept 0.7 row is the session-dated Friday one, not the Saturday copy.
    kept_07 = collapsed[(collapsed.ticker == "NVDA") & (collapsed.rank_score == 0.7)]
    assert kept_07.iloc[0]["run_date"] == pd.Timestamp("2026-04-10")

    weighted = add_session_weights(collapsed, "base_session_date", ["ticker"])
    nvda = weighted[weighted.ticker == "NVDA"]
    assert nvda["session_weight"].tolist() == [0.5, 0.5]
    assert float(nvda["session_weight"].sum()) == 1.0  # one realization total
    assert weighted[weighted.ticker == "AAPL"]["session_weight"].tolist() == [1.0]


def test_session_key_is_pure_date_function_no_cache_dependence() -> None:
    """#60 review round 3: rows whose ticker has NO OHLCV parquet must not
    fail open to a per-row pseudo-session. The session key is a pure
    function of the date, so a cache-less ticker resolves identically to a
    cached one and weekend duplicates still cluster."""
    df = pd.DataFrame({
        "run_date": pd.to_datetime(["2026-04-10", "2026-04-11"]),
        "ticker": ["GHOST", "GHOST"],  # no parquet exists anywhere
    })
    out = annotate_base_sessions(df, date_col="run_date", sessions=WEEKDAY_SESSIONS)
    assert out["base_session_date"].tolist() == [pd.Timestamp("2026-04-10")] * 2
    assert out["non_session_run"].tolist() == [False, True]
    assert out["session_resolved"].all()


def test_unresolved_sessions_are_marked_inadmissible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no NYSE calendar can be built, every row must be MARKED
    session_resolved=False (weekday holidays are undetectable) rather than
    silently treated as independent."""
    monkeypatch.setattr(session_resolution, "nyse_sessions", lambda *a, **k: None)
    df = pd.DataFrame({
        "run_date": pd.to_datetime(["2026-04-10", "2026-04-11"]),
        "ticker": ["NVDA", "NVDA"],
    })
    out = session_resolution.annotate_base_sessions(df, date_col="run_date")
    assert not out["session_resolved"].any()
    # Weekend rolling still works in the fallback; the flag communicates the
    # residual holiday ambiguity.
    assert out["base_session_date"].tolist() == [pd.Timestamp("2026-04-10")] * 2


# ------------------------------------------------------------- end to end


def test_friday_weekend_trio_collapses_to_one_admissible_observation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The exact case the #60 review names: a Fri + Sat + Sun decision-date
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
        run_dates=[dt.date(2026, 4, 10), dt.date(2026, 4, 11), dt.date(2026, 4, 12)],
    )
    _run_backfill(monkeypatch, repo_root)

    conn = sqlite3.connect(db_path)
    raw = pd.read_sql(
        """
        SELECT ps.run_date, cs.ticker, cs.rank_score, tfr.close_price, tfr.fwd_20d
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
    # All 3 share the identical Friday-based close + forward return —
    # proving they are one market realization, not independent draws.
    assert raw["close_price"].nunique() == 1
    assert raw["fwd_20d"].nunique() == 1

    raw["run_date"] = pd.to_datetime(raw["run_date"])
    annotated = annotate_base_sessions(
        raw, date_col="run_date", sessions=WEEKDAY_SESSIONS,
    )
    assert annotated["non_session_run"].sum() == 2  # Sat + Sun are non-session
    assert set(annotated["non_session_kind"].dropna()) == {"weekend"}

    # The trio is an exact re-recording of one decision (identical content),
    # so provenance collapse alone reduces it to the session-dated Friday
    # row with full weight — downstream sample counts do not inflate.
    collapsed = collapse_rerecorded_decisions(
        annotated, "base_session_date", ["ticker"], ["rank_score"],
    )
    weighted = add_session_weights(collapsed, "base_session_date", ["ticker"])
    assert len(weighted) == 1, (
        "a re-recorded Fri/Sat/Sun decision-date trio must collapse to "
        "exactly one statistically admissible observation"
    )
    assert weighted.iloc[0]["run_date"] == pd.Timestamp("2026-04-10")
    assert weighted.iloc[0]["session_weight"] == 1.0
    # The unique-session COVERAGE view agrees.
    assert len(dedupe_by_session(annotated, "base_session_date", ["ticker"])) == 1


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
        run_dates=[dt.date(2026, 4, 6), dt.date(2026, 4, 7), dt.date(2026, 4, 8)],
    )
    _run_backfill(monkeypatch, repo_root)

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
        raw, date_col="run_date", sessions=WEEKDAY_SESSIONS,
    )
    assert annotated["non_session_run"].sum() == 0

    collapsed = collapse_rerecorded_decisions(
        annotated, "base_session_date", ["ticker"], [],
    )
    weighted = add_session_weights(collapsed, "base_session_date", ["ticker"])
    assert len(weighted) == 3, "3 genuine trading-day decisions must remain 3 rows"
    assert weighted["session_weight"].tolist() == [1.0, 1.0, 1.0]


def test_holiday_dated_run_resolves_and_classifies_as_holiday(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A run recorded on a real exchange holiday (Memorial Day Monday
    2026-05-25) resolves as-of Friday's close, and is classified 'holiday'
    (NOT 'weekend') via the shared NYSE calendar — distinguishing the two
    non-session run kinds end to end."""
    pytest.importorskip("pandas_market_calendars")
    repo_root = tmp_path / "umbrella"
    cache_root = repo_root / "data" / "ohlcv"
    db_path = repo_root / "data" / "runs.db"
    (repo_root / "backtesting" / "renquant_104").mkdir(parents=True)
    # Real-shaped bars: weekdays WITHOUT the Memorial Day holiday bar.
    index = pd.bdate_range(start="2026-04-27", periods=66)
    index = index.drop(pd.Timestamp("2026-05-25"))
    closes = [100.0 + i for i in range(len(index))]
    _seed_parquet(cache_root, "NVDA", closes, index=index)
    _seed_candidate_db(db_path, run_dates=[dt.date(2026, 5, 25)])
    _run_backfill(monkeypatch, repo_root)

    friday_close = float(closes[list(index).index(pd.Timestamp("2026-05-22"))])
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT close_price, fwd_20d FROM ticker_forward_returns "
        "WHERE as_of_date = '2026-05-25' AND ticker = 'NVDA'",
    ).fetchone()
    conn.close()
    assert row is not None, "holiday-dated decision must join a forward outcome"
    assert row[0] == friday_close
    assert row[1] is not None

    sessions = nyse_sessions(dt.date(2026, 4, 20), dt.date(2026, 6, 30))
    df = pd.DataFrame({"run_date": pd.to_datetime(["2026-05-25", "2026-05-22"]),
                       "ticker": ["NVDA", "NVDA"]})
    annotated = annotate_base_sessions(df, date_col="run_date", sessions=sessions)
    assert annotated["non_session_kind"].tolist() == ["holiday", None]
    # ... and the holiday row dedupes against Friday's genuine session row.
    assert len(dedupe_by_session(annotated, "base_session_date", ["ticker"])) == 1


def test_rolling_ic_loader_dedupes_weekend_duplicates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The live-sim reconcile rolling-IC loader must count a Fri+Sat+Sun
    duplicate cohort as ONE (rank, fwd) observation — distinct decisions
    averaged against the shared realization."""
    from renquant_backtesting.reconciliation.live_sim_reconcile import (  # noqa: PLC0415
        _load_score_realized_pairs,
    )

    repo_root = tmp_path / "umbrella"
    cache_root = repo_root / "data" / "ohlcv"
    db_path = repo_root / "data" / "runs.db"
    (repo_root / "backtesting" / "renquant_104").mkdir(parents=True)
    _seed_parquet(cache_root, "NVDA", [100.0 + i for i in range(65)])
    _seed_candidate_db(
        db_path,
        run_dates=[dt.date(2026, 4, 10), dt.date(2026, 4, 11), dt.date(2026, 4, 12)],
        rank_scores=[0.7, 0.5, 0.3],  # Friday's score differs from the dupes
    )
    _run_backfill(monkeypatch, repo_root)

    pairs = _load_score_realized_pairs(
        db_path, "2026-04-01", "2026-04-30", horizon_days=20,
    )
    assert len(pairs) == 1, (
        "a Fri/Sat/Sun duplicate cohort must contribute exactly one IC pair"
    )
    # Three genuinely different decisions share the realization: the cluster
    # contributes their MEAN score (retained + averaged, not one arbitrarily
    # kept and two discarded).
    assert pairs[0][0] == pytest.approx((0.7 + 0.5 + 0.3) / 3)
