"""Campaign B5 lockstep tests: ``analysis/session_resolution`` sources its
calendar and at-or-before session-key semantics from the canonical
``renquant_common.market_calendar`` (audit #296 §4.1 row 4 / XC-3 — the
orchestrator KPI scorecard used to hand-copy THIS module's semantics; both
now delegate to the same canonical, so they can never disagree again)."""
from __future__ import annotations

import datetime as dt

import pandas as pd

from renquant_backtesting.analysis.session_resolution import (
    classify_date,
    nyse_sessions,
    session_key,
)
from renquant_common.market_calendar import (
    session_key as canonical_session_key,
    sessions_between,
)


def test_nyse_sessions_is_the_canonical_index() -> None:
    start, end = dt.date(2026, 6, 1), dt.date(2026, 7, 31)
    ours = nyse_sessions(start, end)
    canonical = sessions_between(start, end)
    assert ours is not None
    assert ours.equals(canonical)
    assert ours.tz is None


def test_session_key_lockstep_with_canonical() -> None:
    sessions = sessions_between(dt.date(2026, 6, 1), dt.date(2026, 7, 31))
    for day in ("2026-06-26", "2026-06-27", "2026-06-28", "2026-07-03", "2026-06-30"):
        assert session_key(day, sessions) == canonical_session_key(day, sessions)


def test_session_key_weekday_fallback_preserved_outside_window() -> None:
    # A date preceding the sessions window falls through to the weekday
    # logic — the pre-B5 degenerate-edge behavior, kept byte-identical.
    sessions = sessions_between(dt.date(2026, 6, 1), dt.date(2026, 6, 30))
    assert session_key("2026-05-02", sessions) == dt.date(2026, 5, 1)  # Sat -> Fri
    assert session_key("2026-05-06", sessions) == dt.date(2026, 5, 6)  # weekday self


def test_classify_date_against_canonical_sessions() -> None:
    sessions = nyse_sessions(dt.date(2026, 6, 25), dt.date(2026, 7, 10))
    assert classify_date("2026-06-30", sessions) == "session"
    assert classify_date("2026-06-28", sessions) == "weekend"
    assert classify_date("2026-07-03", sessions) == "holiday"  # Jul-4 observed
