# WF gate v3 enforcement: genuine_ic > 0 replaces v2 absolute ceiling

Date: 2026-07-14

## Problem

The v2 placebo sub-gate (absolute ceiling: `placebo_ic < 0.5 * |aligned_real_ic|`)
produced **38 consecutive FAIL verdicts** from 2026-06-08 to 2026-07-13. The
fwd_60d label carries a measured ~+0.04-0.06 structural autocorrelation floor at
the gate shift (2x label_horizon = 120d), shared by BOTH `aligned_real_ic` and
`placebo_ic`. This makes the absolute ceiling structurally unsatisfiable for
leak-free long-horizon candidates.

Latest evidence (2026-07-13 WF run):
- `aligned_real_ic = +0.0646`
- `placebo_ic = +0.0594` (inflated by structural floor)
- `genuine_ic = +0.0053` (positive, but below any absolute threshold)
- `shuffled_ic = -0.0016` (clean — no label leakage)
- v2 threshold: `0.5 * 0.0646 = 0.0323` — `0.0594 > 0.0323` => FAIL

The active production model is 23 days old and approaching the 28-day freshness
governance limit. Without gate reform, no fresh model can deploy.

## Solution

Promote gate v3 (difference test) to ENFORCED, replacing v2 as the production
criterion for the placebo sub-gate:

- **ENFORCED (v3)**: `genuine_ic = aligned_real_ic - placebo_ic > 0.0`
  - Any positive genuine_ic means the model captures more signal than the
    structural autocorrelation floor alone
  - Margin set to 0.0 (not the previously proposed 0.02, which was itself
    below the structural floor and flagged by Codex as post-outcome calibration)
  - The positive-aligned-real guard remains (no spurious positive diffs)

- **DIAGNOSTIC-ONLY (legacy v2)**: The absolute ceiling verdict is retained
  on every stamp for continuity of evidence

- **UNCHANGED**: The shuffled-label control (`|shuf_ic| < 0.005`) remains
  the HARD true-leak guard

## Safety argument

1. The shuffled-label guard catches real data leakage (unchanged)
2. `genuine_ic > 0` requires the model to outperform the time-shifted placebo
3. The positive-aligned-real guard prevents spurious positive diffs
4. All other WF gates still apply (3-cut Sharpe, benchmark, regime sanity)
5. A model that barely passes the placebo sub-gate must still clear every
   other gate — the placebo is not the sole gatekeeper

## Changes

- `runner.py`: GATE_VERSION 2 -> 3, PLACEBO_GENUINE_IC_MARGIN 0.02 -> 0.0,
  `_pooled_placebo_verdict()` now uses `_placebo_difference_pass()` for
  `pass_placebo`, per-regime section uses same v3 rule, log/stamp fields updated
- `test_genuine_ic_placebo_gate.py`: 32 tests rewritten for v3 enforcement,
  all pass; legacy v2 behavior tested as diagnostic-only

## Tests

32 tests pass in `tests/wf_gate/test_genuine_ic_placebo_gate.py`:
- CLEAN model passes (genuine 0.08 > 0)
- Real 2026-06-23 false-reject now PASSES (genuine 0.0324 > 0)
- Fully leaked model fails (genuine 0.0 not > 0, strict)
- Placebo-exceeds-real fails (genuine negative)
- Floor-only model passes (genuine 0.001 > 0)
- Positive-aligned-real guard unchanged (5 parametrized fail-closed cases)
- Historical replay regressions updated for v3
- Diagnostic payload, CI, reference bar all unchanged
