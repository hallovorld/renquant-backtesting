"""Shared session resolution — single source of truth for "which real NYSE
trading session does this decision date's data actually come from."

`backfill_forward_returns.py` resolves a decision date lacking its own OHLCV
bar (weekend or exchange holiday) to the last bar at or before that date —
correct for filling `ticker_forward_returns`, since fwd_h then counts trading
bars from that same base. But `ticker_forward_returns` is keyed by
(as_of_date, ticker), one row per as_of_date, so a Friday/Saturday/Sunday
decision-date trio resolving to the same Friday base close produces THREE
ROWS with identical close_price and fwd_*d values — one real market
realization, not three independent ones. Any consumer joining candidate
rows to ticker_forward_returns and treating (run_date, ticker) as an
independent observation silently overweights that realization 2-3x.

`ticker_forward_returns`'s schema is owned by renquant-pipeline, not this
repo — persisting a base_session_date/non_session_run column from here would
mean this repo's script and renquant-pipeline's own CREATE TABLE silently
diverging on schema, the exact cross-repo drift this fix exists to prevent.
Instead the canonical mapping is computed on read. It is a pure function of
the DATE against the NYSE session calendar (the same shared calendar
renquant-pipeline already uses: ``pandas_market_calendars`` in
``kernel/data.py``, ``kernel/execution/t2_settlement.py``), so every
consumer — including ones without access to the OHLCV parquet cache, like
the orchestrator KPI scorecard — can derive the identical key. Backfill and
every in-repo consumer call the SAME functions here, so they can never
disagree.

Degraded fallback: when ``pandas_market_calendars`` is unavailable,
Saturday/Sunday dates roll back to the preceding Friday and weekdays map to
themselves. Exchange holidays are then NOT detected (a holiday-dated run
keys to itself and its as-of forward-return row is not written) — same
fail-closed posture as the pre-fix exact-hit behavior. All observed
non-session live run dates to date are weekends, so the fallback covers the
real data; the calendar path additionally covers holiday-dated runs.
"""
from __future__ import annotations

import datetime
import logging

log = logging.getLogger("session-resolution")

_FALLBACK_WARNED = False


def nyse_sessions(
    start: datetime.date,
    end: datetime.date,
) -> "pd.DatetimeIndex | None":
    """Real NYSE sessions in [start, end], tz-naive and normalized.

    Returns None when ``pandas_market_calendars`` is unavailable — callers
    then degrade to the weekday fallback in :func:`session_key` /
    :func:`classify_date` (weekends handled, holidays not).
    """
    global _FALLBACK_WARNED  # noqa: PLW0603
    import pandas as pd  # noqa: PLC0415
    try:
        import pandas_market_calendars as mcal  # noqa: PLC0415
    except Exception:  # pragma: no cover — mcal is in the umbrella .venv
        if not _FALLBACK_WARNED:
            log.warning(
                "pandas_market_calendars unavailable — session resolution "
                "degrades to weekday-only (weekends roll to Friday; NYSE "
                "holidays NOT detected)."
            )
            _FALLBACK_WARNED = True
        return None
    cal = mcal.get_calendar("NYSE")
    days = pd.DatetimeIndex(cal.valid_days(start_date=start, end_date=end))
    if days.tz is not None:
        days = days.tz_localize(None)
    return days.normalize()


def session_key(
    date: datetime.date | str,
    sessions: "pd.DatetimeIndex | None" = None,
) -> datetime.date:
    """Canonical ``decision_session_date``: the last NYSE session at or
    before ``date``. A pure function of the date — NOT of any per-ticker
    parquet — so every consumer derives the identical dedup key.

    With ``sessions=None`` (calendar unavailable): Saturday/Sunday roll back
    to the preceding Friday; weekdays map to themselves (holidays undetected
    — degraded, fail-closed elsewhere).
    """
    import pandas as pd  # noqa: PLC0415
    ts = pd.Timestamp(date).normalize()
    if sessions is not None and len(sessions):
        idx = int(sessions.searchsorted(ts, side="right")) - 1
        if idx >= 0:
            return sessions[idx].date()
        # date precedes the calendar window — fall through to weekday logic
    d = ts.date()
    if d.weekday() >= 5:  # Sat=5 / Sun=6 → preceding Friday
        return d - datetime.timedelta(days=d.weekday() - 4)
    return d


def classify_date(
    date: datetime.date | str,
    sessions: "pd.DatetimeIndex | None" = None,
) -> str:
    """Classify a decision date: ``'session'`` | ``'weekend'`` | ``'holiday'``.

    Distinguishing weekends from exchange holidays requires the NYSE
    calendar; without it (``sessions=None``) only weekends are detected and
    weekday holidays report as ``'session'`` (degraded).
    """
    import pandas as pd  # noqa: PLC0415
    ts = pd.Timestamp(date).normalize()
    d = ts.date()
    if d.weekday() >= 5:
        return "weekend"
    if sessions is not None and len(sessions) and sessions[0] <= ts <= sessions[-1]:
        return "session" if ts in sessions else "holiday"
    return "session"


def annotate_base_sessions(
    df: "pd.DataFrame",
    date_col: str,
    sessions: "pd.DatetimeIndex | None" = None,
) -> "pd.DataFrame":
    """Add ``base_session_date`` (canonical session key, Timestamp),
    ``non_session_run`` (bool), ``non_session_kind`` (None | 'weekend' |
    'holiday') and ``session_resolved`` (bool) columns, resolved date-level
    against the NYSE calendar.

    ``session_resolved`` is False for every row when no calendar is
    available (weekday-fallback keys cannot rule out holidays) — consumers
    must surface that as "session identity unresolved", NOT silently treat
    the rows as independent. There is deliberately NO per-ticker failure
    mode: the key is a pure function of the date, so OHLCV cache coverage
    cannot change it.

    When ``sessions`` is None the calendar is built automatically over the
    frame's date range (padded); pass an explicit index for determinism in
    tests or to amortize across calls.
    """
    import pandas as pd  # noqa: PLC0415
    out = df.copy()
    if out.empty:
        out["base_session_date"] = pd.Series(dtype="datetime64[ns]")
        out["non_session_run"] = pd.Series(dtype=bool)
        out["non_session_kind"] = pd.Series(dtype=object)
        out["session_resolved"] = pd.Series(dtype=bool)
        return out
    dates = pd.to_datetime(out[date_col]).dt.normalize()
    if sessions is None:
        sessions = nyse_sessions(
            (dates.min() - pd.Timedelta(days=14)).date(),
            (dates.max() + pd.Timedelta(days=7)).date(),
        )
    uniq = {
        ts: (
            pd.Timestamp(session_key(ts, sessions)),
            classify_date(ts, sessions),
        )
        for ts in dates.unique()
    }
    out["base_session_date"] = [uniq[ts][0] for ts in dates]
    kinds = [uniq[ts][1] for ts in dates]
    out["non_session_run"] = [k != "session" for k in kinds]
    out["non_session_kind"] = [None if k == "session" else k for k in kinds]
    out["session_resolved"] = sessions is not None and len(sessions) > 0
    return out


def collapse_rerecorded_decisions(
    df: "pd.DataFrame",
    session_col: str,
    identity_cols: list[str],
    decision_cols: list[str],
    date_col: str = "run_date",
) -> "pd.DataFrame":
    """Drop exact re-recordings of ONE decision: rows identical on
    (session, *identity_cols, *decision_cols). A weekend/holiday re-stamp of
    Friday's decision, or a same-day re-run persisting the identical scores,
    is the same decision recorded twice — keeping one (session-dated first,
    then earliest ``date_col``) loses nothing.

    Rows with the same session key but DIFFERENT decision content are
    genuinely different decisions sharing one realized return; they are
    retained (equal outcome identity does not imply equal decision
    identity) — weight them with :func:`add_session_weights` so inference
    does not overcount the shared realization.
    """
    key_cols = [session_col, *identity_cols, *decision_cols]
    sort_cols = list(key_cols)
    if "non_session_run" in df.columns:
        sort_cols.append("non_session_run")  # False (session-dated) first
    if date_col in df.columns and date_col not in sort_cols:
        sort_cols.append(date_col)
    return (
        df.sort_values(sort_cols, kind="mergesort")
        .drop_duplicates(subset=key_cols, keep="first")
    )


def add_session_weights(
    df: "pd.DataFrame",
    session_col: str,
    identity_cols: list[str],
) -> "pd.DataFrame":
    """Add ``session_weight`` = 1 / (number of retained decisions sharing
    this (session, *identity_cols) cluster). Distinct decisions sharing one
    realized return each carry a fraction of one observation, so a cluster
    contributes exactly ONE observation of its market realization to any
    weighted statistic — retained, not arbitrarily discarded. Run
    :func:`collapse_rerecorded_decisions` first so exact re-recordings do
    not dilute the weights of genuinely different decisions.
    """
    out = df.copy()
    key_cols = [session_col, *identity_cols]
    out["session_weight"] = 1.0 / out.groupby(key_cols)[session_col].transform("size")
    return out


def dedupe_by_session(
    df: "pd.DataFrame",
    session_col: str,
    identity_cols: list[str],
    date_col: str = "run_date",
) -> "pd.DataFrame":
    """Collapse rows sharing (session_col, *identity_cols) to ONE row each —
    the unique-session view, for COVERAGE COUNTING (how many independent
    market realizations exist), not for decision-level inference: it keeps
    one representative row per cluster and so discards genuinely different
    decisions that share a realization. For inference over decisions use
    :func:`collapse_rerecorded_decisions` (drop exact re-recordings) +
    :func:`add_session_weights` (retain distinct decisions, weight the
    shared realization) instead.

    The kept row is deterministic and canonical: a session-dated row is
    preferred over its weekend/holiday duplicates (when a
    ``non_session_run`` column is present), then the earliest ``date_col``.
    """
    key_cols = [session_col, *identity_cols]
    sort_cols = list(key_cols)
    if "non_session_run" in df.columns:
        sort_cols.append("non_session_run")  # False (session-dated) first
    if date_col in df.columns and date_col not in sort_cols:
        sort_cols.append(date_col)
    return (
        df.sort_values(sort_cols, kind="mergesort")
        .drop_duplicates(subset=key_cols, keep="first")
    )
