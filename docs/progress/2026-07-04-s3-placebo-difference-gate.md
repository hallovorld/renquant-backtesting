# S3: Switch WF-gate placebo criterion to difference test

**Date:** 2026-07-04
**Supersedes:** PR #67 (feat/s3-placebo-difference-gate)
**System:** 107 (WF-gate)
**Status:** PR ready for review

## What changed

The WF-gate sanity battery's pooled placebo criterion switches from the legacy
absolute-ceiling rule (gate v2) to the pre-registered difference test (gate v3,
S3):

- **Before (v2):** `|placebo_ic| < 0.5 × |aligned_real_ic|` (floored at 0.005)
- **After (v3):** `genuine_ic = aligned_real_ic − placebo_ic >= 0.02`

The per-regime placebo leg retains the absolute-ceiling rule (no family-wise
error control across regimes for the difference test yet).

## Pre-registration justification

### Why the difference test?

The 60d forward label has ~30d embargo overlap at the 2× label-horizon gate
shift. This inflates **both** `aligned_real_ic` and `placebo_ic` by a measured
~+0.04 structural floor (PRs #52/#53, N=477 dates, `label_autocorr_ic` pooled
mean = +0.040 ± 0.006).

The ratio test conflates this floor with real signal: a leak-free model with
`real_ic=+0.06`, `placebo=+0.04` (genuine_ic=+0.02) **FAILS** because
`0.04 > 0.5×0.06 = 0.03`. The difference test cancels the shared floor.

### Why 0.02?

The margin was frozen 2026-07-02 in the unified 107 master plan §S3 row
**before** any specific candidate evaluation. It sits at half the measured
structural floor (~0.04), meaning the gate requires genuine predictive content
roughly equal to the floor itself — conservative enough that the floor alone
cannot pass a no-edge model, while not being so tight that floating-point
noise in the ~+0.04 cancellation rejects clean models.

### Why >= (not >)?

The original v3 shadow implementation used strict `>`. The S3 enforcement uses
`>=` because at exactly the margin, the model has demonstrated the required
minimum genuine content. The margin itself was the pre-registered threshold.

### What held-out evidence shows the difference test separates clean vs leaky?

The shadow evaluation corpus (stamped since gate v3 candidate, PRs #52/#53):
- **Clean models** (genuine_ic >> 0.02): correctly pass both tests
- **Leaked models** (genuine_ic ~ 0): correctly fail both tests
- **False-reject zone** (genuine_ic 0.02–0.04): the difference test correctly
  passes these while the ratio test incorrectly fails them — this is exactly
  the structural unfairness the fix addresses
- **Floor-only models** (genuine_ic ~ 0.001): correctly fail both tests

The 2× label-horizon shift is the standard placebo shift per AFML §7.4, not a
tuned parameter.

## Files changed

- `src/renquant_backtesting/wf_gate/runner.py` — flip `_pooled_placebo_verdict`
  to enforce difference test, legacy absolute rule becomes diagnostic
- `tests/wf_gate/test_placebo_difference_gate.py` — 11 new tests for S3
- `tests/wf_gate/test_genuine_ic_placebo_gate.py` — updated to match v3
  enforcement (was pinning v2-enforced / v3-shadow-only behavior)
- `docs/progress/2026-07-04-s3-placebo-difference-gate.md` — this doc

## Test results

186 passed, 2 failed (pre-existing sim_driver env issues, unrelated).
