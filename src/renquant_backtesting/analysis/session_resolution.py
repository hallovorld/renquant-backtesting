"""Shared as-of session resolution — single source of truth for "which real
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
repo — adding a persisted base_session_date/non_session_run column here
would mean this repo's script and renquant-pipeline's own CREATE TABLE
silently diverging on schema, the exact cross-repo drift this fix exists to
prevent. Instead: resolve the same session mapping independently, on read,
from the OHLCV cache both scripts already have access to. Backfill and every
in-repo consumer call the SAME function here, so they can never disagree.
"""
from __future__ import annotations

import datetime
from pathlib import Path


def resolve_base_session_date(
    date: datetime.date | str,
    df: "pd.DataFrame",
) -> datetime.date | None:
    """Return the actual trading-session date `date`'s forward-return data
    is based on: `date` itself if it has its own bar, otherwise the last bar
    at or before it (the same as-of rule backfill_forward_returns.py uses).
    Returns None if `date` precedes the first cached bar (no base exists).

    `df` must be a per-ticker OHLCV frame with a sorted DatetimeIndex, as
    produced by `backfill_forward_returns._load_ohlcv`.
    """
    import pandas as pd  # noqa: PLC0415

    ts = pd.Timestamp(date)
    idx = int(df.index.searchsorted(ts, side="right")) - 1
    if idx < 0:
        return None
    return df.index[idx].date()


def is_non_session_run(date: datetime.date | str, base_session_date: datetime.date | None) -> bool:
    """True iff `date` itself is not the trading session its data is based on
    (i.e. it's a weekend/holiday-dated decision resolving as-of to an earlier
    real session)."""
    if base_session_date is None:
        return False
    import pandas as pd  # noqa: PLC0415

    return pd.Timestamp(date).date() != base_session_date


def annotate_base_sessions(
    df: "pd.DataFrame",
    date_col: str,
    ticker_col: str,
    cache_root: Path,
) -> "pd.DataFrame":
    """Add `base_session_date` and `non_session_run` columns to `df` by
    resolving each (date, ticker) row's actual trading-session basis against
    the per-ticker OHLCV cache under `cache_root`. Rows whose ticker has no
    cached parquet get `base_session_date=NaT`, `non_session_run=False`
    (fail open to "treat as its own session" only when we cannot determine
    otherwise — these rows already can't be deduplicated meaningfully without
    the cache, and are rare in practice since the same cache backfill used
    to populate ticker_forward_returns in the first place)."""
    import pandas as pd  # noqa: PLC0415

    from renquant_backtesting.analysis.backfill_forward_returns import _load_ohlcv

    base_dates: list = []
    non_session: list = []
    cache: dict = {}
    for date, ticker in zip(df[date_col], df[ticker_col]):
        if ticker not in cache:
            cache[ticker] = _load_ohlcv(ticker, cache_root)
        tdf = cache[ticker]
        if tdf is None:
            # Uncachable ticker — fail open to "own date is its own session"
            # so this row dedupes only with itself, never silently collides
            # with another uncachable row under a shared null key.
            base_dates.append(pd.Timestamp(date))
            non_session.append(False)
            continue
        base = resolve_base_session_date(date, tdf)
        if base is None:
            base_dates.append(pd.Timestamp(date))
            non_session.append(False)
            continue
        base_dates.append(pd.Timestamp(base))
        non_session.append(is_non_session_run(date, base))
    out = df.copy()
    out["base_session_date"] = base_dates
    out["non_session_run"] = non_session
    return out


def dedupe_by_session(
    df: "pd.DataFrame",
    session_col: str,
    identity_cols: list[str],
) -> "pd.DataFrame":
    """Collapse rows sharing (base_session_date, *identity_cols) to one row
    each — the admissible, statistically-independent view. `identity_cols`
    should include the ticker and whatever else distinguishes a genuine
    decision (e.g. ticker alone for factor analysis; ticker + role for
    selection-level analyses). Assumes `session_col` has already been
    populated by `annotate_base_sessions` (never null — uncachable rows fall
    back to their own date there, so they dedupe only with themselves)."""
    key_cols = [session_col] + identity_cols
    return df.sort_values(key_cols).drop_duplicates(subset=key_cols, keep="first")
