# Re-point to the canonical NYSE market calendar (campaign B5)

Date: 2026-07-04
PR: fix(calendar): re-point to the common canonical

## What

`analysis/session_resolution.py` (audit #296 §4.1 row 4 / XC-3) now sources
its calendar from the canonical `renquant_common.market_calendar`:

- `nyse_sessions` delegates to `sessions_between` (hand-rolled
  `pandas_market_calendars` acquisition deleted); keeps this module's
  documented lenient contract — backend unavailable => `None` + one warning,
  callers degrade to the weekday fallback.
- `session_key` delegates its at-or-before semantics to the canonical
  `session_key`; the date-precedes-window edge still falls through to the
  weekday logic (byte-identical pre-B5 behavior).
- `classify_date` / `annotate_base_sessions` / weights and dedup helpers are
  backtesting-specific and unchanged — they consume the canonical index.

The orchestrator KPI scorecard, which hand-copied this module's semantics
(docstring-admitted), re-points to the same canonical in its own PR — the
two consumers can never disagree again by construction.

## Semantics / behavior

- Equivalence-proven on a 10-year daily fixture (2016-01-01..2026-12-31):
  session index, per-date session keys, and classifications identical to the
  pre-B5 implementation.
- Dependency floor bumped: `renquant-common>=0.10,<1.0`.
- Stale-deploy safety: a renquant_common predating `market_calendar`
  degrades to the same weekday fallback as a missing
  `pandas_market_calendars` did before.

## Merge order

renquant-common `feat(calendar): canonical NYSE market calendar (campaign
B5)` merges FIRST; this PR's CI is red until it lands (CI checks out
common@main).
