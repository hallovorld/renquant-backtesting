# WF Gate v3 Pre-Registration Protocol

STATUS: DRAFT — requires operator + codex review before v3 activation

## Background

Gate v2 (absolute ceiling: `placebo_ic < 0.5 × |aligned_real_ic|`) has produced
38 consecutive FAIL verdicts from 2026-06-08 to 2026-07-14. The fwd_60d label
carries a measured ~+0.04–0.06 structural autocorrelation floor at the gate shift
(2× label_horizon = 120d), shared by both `aligned_real_ic` and `placebo_ic`,
making the absolute ceiling structurally unsatisfiable for leak-free long-horizon
candidates.

Gate v3 (difference test: `genuine_ic = aligned_real_ic − placebo_ic > margin`)
cancels this shared floor. It currently runs as SHADOW-ONLY, stamping evidence
on every verdict without affecting pass/fail.

## Pre-Registration: v3 Activation Criteria

### 1. Primary Metric

One-sided lower confidence bound for `genuine_ic`, using a dependence-aware
procedure (moving-block bootstrap or paired time-series test) to account for
overlapping fwd_60d labels. The enforced criterion will be:

    lower_bound(genuine_ic, alpha) > min_effect

NOT `genuine_ic > 0` (point estimate).

### 2. Minimum Economic Effect Size

To be determined from simulation: the smallest `genuine_ic` that produces a
measurable improvement in downstream portfolio Sharpe after transaction costs.
This must be set BEFORE examining any candidate's results.

### 3. Significance Level

One-sided `alpha = 0.05` (or stricter). Must clear the multiplicity correction
below.

### 4. Candidate Set and Calendar

- Evaluation over **independent model vintages** (each trained on non-overlapping
  data), not rolling-window replays of the same model.
- Minimum N vintages TBD (power analysis required).
- Calendar: contiguous evaluation periods, no cherry-picking.

### 5. Family-Wise Error Control

Per-regime placebo sub-tests (BULL_CALM, BULL_VOLATILE, BEAR, etc.) create
multiple comparison paths. Apply Holm-Bonferroni or make regime-level tests
purely diagnostic (non-gating).

### 6. Synthetic Null Controls

Before activation, demonstrate acceptable false-positive rates on:
- **Floor-only null**: `genuine_ic = 0` (model captures exactly the
  autocorrelation floor and nothing more)
- **Leak-like null**: `genuine_ic < 0` (placebo exceeds real)
- **Near-threshold**: `genuine_ic` barely above/below `min_effect`

A gate that intentionally passes floor-only models must not authorize deployment.

### 7. Prospective Held-Out Validation

At least one model vintage must be evaluated on a truly prospective period
(trained before the held-out window, no retraining allowed during evaluation).
Shadow-period accumulation counts IF the model was frozen before the shadow
window began.

## Activation Sequence

1. Complete all items above → write results into this document
2. Codex review of the completed prereg + results
3. Operator approval
4. Single commit: bump `GATE_VERSION`, set `sanity_placebo_v3_gating = True`,
   set `PLACEBO_GENUINE_IC_MARGIN` to validated `min_effect`
5. Monitor first 4 weekly verdicts under enforcement

## Current v2 Status

v2 remains the ENFORCED gate. The production model is approaching the 28-day
freshness governance limit. If v3 activation cannot be completed before model
staleness, the options are:
- Operator override of the freshness limit for the current model
- Operator-authorized manual promotion (bypassing placebo sub-gate only,
  all other gates still enforced)
- Emergency v2 threshold adjustment (requires its own prereg)
