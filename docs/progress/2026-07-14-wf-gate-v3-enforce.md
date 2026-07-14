# WF gate v3 pre-registration protocol (revised from enforcement attempt)

Date: 2026-07-14

## Problem

The v2 placebo sub-gate (absolute ceiling: `placebo_ic < 0.5 * |aligned_real_ic|`)
produced **38 consecutive FAIL verdicts** from 2026-06-08 to 2026-07-14. The
fwd_60d label carries a measured ~+0.04-0.06 structural autocorrelation floor at
the gate shift (2x label_horizon = 120d), shared by BOTH `aligned_real_ic` and
`placebo_ic`. This makes the absolute ceiling structurally unsatisfiable for
leak-free long-horizon candidates.

## Approach (revised after codex review)

Initial attempt: enforce v3 directly (`genuine_ic > 0`). Codex correctly
identified this as an unvalidated relaxation — `genuine_ic > 0` on one noisy,
overlapping-label estimate has no controlled false-positive rate. Under a null
centered difference, a strict zero threshold passes roughly half of noisy
estimates.

Revised approach:
1. **v2 remains ENFORCED** — no change to production gate behavior
2. **v3 stays SHADOW-ONLY** — continues accumulating diagnostic evidence
3. **Pre-registration protocol added** (`docs/research/2026-07-14-wf-gate-v3-prereg.md`)
   specifying what must be validated before v3 can activate:
   - Dependence-aware confidence bound (not point estimate)
   - Minimum economic effect size (set before examining candidates)
   - Independent model vintages (not rolling-window replays)
   - Family-wise error control for per-regime sub-tests
   - Synthetic null false-positive demonstration
   - Prospective held-out validation

## Changes

- `docs/research/2026-07-14-wf-gate-v3-prereg.md`: new pre-registration protocol
- No code changes to `runner.py` or test files

## Model Freshness Risk

The active production model is approaching the 28-day freshness governance limit.
With v2 structurally unsatisfiable, options documented in the prereg protocol:
operator override, manual promotion (bypassing placebo sub-gate only), or
emergency v2 threshold adjustment (requires its own prereg).
