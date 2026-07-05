# WF Gate Incumbent Regression Check

**Date:** 2026-07-05
**PR:** TBD
**Status:** Design RFC — awaiting review before implementation

## What

Replace the WF gate's hard SPY benchmark requirement with an incumbent regression
check: candidate must not be significantly worse than the current production model.
SPY comparison becomes advisory-only (logged, not gated).

## Why

In bull markets (SPY Sharpe >1.0), the stock-selection strategy structurally
cannot beat SPY — every promote requires operator override. The right question is
"is the new model at least as good as the old one?" not "does it beat the index?"

## Key decisions

- `absolute_ok` (mean Sharpe >= 0.40, 2/3 cuts positive): **unchanged, hard gate**
- `benchmark_ok` (beat SPY): **demoted to advisory** (still logged)
- `regime_ok` (beat SPY per regime): **demoted to advisory** (still logged)
- `incumbent_ok` (new): **hard gate** — candidate_sharpe >= incumbent - 0.15,
  candidate_apy >= incumbent - 5pp. Auto-pass if no incumbent.

## Validation

The 2026-07-05 retrain (Sharpe 0.776) would PASS under this gate: incumbent
Sharpe = 0.697, delta = +0.079 (improvement, not regression).
