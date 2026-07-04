# S3: Placebo difference test gate

**Date**: 2026-07-04
**PR**: (this PR)
**Status**: Implementation complete

## What changed

The WF-gate sanity battery's placebo check was structurally unfair: the raw
ratio test (`|placebo_ic| < 0.5 × |real_ic|`) did not account for the ~+0.04
embargo floor caused by ~30d overlap in the 60d label (confirmed in PRs #52/#53).

This PR switches the gate criterion to the **placebo difference test**:
`genuine_ic = real_ic − placebo_ic ≥ 0.02` (margin per master plan §S3).

The Layer-1a diagnostic profile (RFC #259, already merged) computes
`genuine_ic` at {1×, 2×, 3×}×horizon. The gate now reads the 2× entry (the
gate shift) and uses it as the primary pass/fail test. Falls back to the
legacy ratio test if the profile is unavailable (pre-Layer-1a artifacts).

## Files changed

- `src/renquant_backtesting/wf_gate/runner.py`: `PLACEBO_DIFF_MARGIN = 0.02`,
  gate criterion at `pass_placebo`, reason messages, return dict fields
- `tests/wf_gate/test_placebo_difference_gate.py`: 9 tests (clean pass,
  leaked fail, boundary, fallback, end-to-end from assembly)

## Acceptance criteria (master plan)

- [x] Gate uses difference test (genuine_ic ≥ margin) instead of raw ratio
- [x] Margin frozen at 0.02 vs measured +0.04 embargo floor
- [x] Test fixture: passes known-clean, fails known-leaked
- [ ] 3 clean weekly runs (requires live WF runs after merge)
