# 2026-08-05 — GOAL-6 Stage 2 stops scoring candidates on a regime mix

## STATUS

The Stage-2 candidate-scoring lane gains a per-regime split. Capability + wiring,
no behaviour change to any verdict.

## WHY

`summarize_lineage_scores` produced a **pooled** `mean_ic`. Measured 2026-08-05
(orch#805, census orch#807/#809), the pooled figure on this book is a
**regime-mix artifact** `[VERIFIED — stamped per-regime placebo profiles,
8 readings 07-05 → 08-04]`:

| regime | genuine_ic | n_dates | buys placed there |
|---|---|---|---|
| BEAR | +0.3346 … +0.3417 | 50 | **0** |
| BULL_CALM | −0.0339 … −0.0294 | 363–377 | **136 of 154** |

The pooled number comes out **positive** because BEAR's 50 dates drag it up. A
Stage-2 lane that ranks candidates on a pooled mean ranks them on exactly that
artifact — which is the evaluation-path problem GOAL-6 has been stuck on, one
level up from "the gate admits on recipe hash only".

## WHAT

- `summarize_lineage_scores(scores, labels_by_date, regime_by_date=None)` now
  stamps `by_regime` (per regime: `n_dates`, `mean_ic`, `min_ic`, `max_ic`),
  tags each `per_date` row with its regime, and sets
  **`pooled_is_a_regime_mix`** — true when the pooled sign disagrees with
  *every* regime's, which is the live shape in miniature.
- The regime map is supplied **by the caller**, exactly like `labels_by_date`.
  This module never derives a regime; the production chain
  (`build_regime_series`) is the only source.
- Threaded through `_score_segment` and the public
  `attempt_lineage_scoring_stamp` so it is reachable, not scaffolding.

## Discipline held

- **Absence reads as absence.** No regime map → `by_regime` is `None` with a
  stated `by_regime_reason`, never `{}` (which would read as "measured, and
  there were no regimes"). `pooled_is_a_regime_mix` is `None`, not `False`.
- **A date the caller could not label is BUCKETED as `__unassigned__`, not
  dropped** — discarding it would change the pooled mean the split exists to
  explain. A test asserts the buckets reconcile with `n_dates_with_labels`.
- **Unassigned dates never decide the mix flag.**
- **The seam is unchanged when no map is given**: the regime map is passed as a
  keyword and only when supplied, so the two-argument call contract other
  callers and test doubles rely on still holds.
- **Pooled output is bit-identical** with and without the split — the split
  explains the pooled figure, it does not move it. Pinned by a test.

## NOT done

The gate's own call site does not yet pass a regime map — that is a separate
change to the runner, and wiring it there means deciding which regime series the
Stage-2 window uses (the production chain per scored date). Filed as the next
step rather than guessed at here.

Suites: 11 new tests. The 3 failures on this branch (`test_b1_lift`,
`test_import_lift`, `test_lineage_stage2::test_REAL_run001_...`) all reproduce on
`origin/main` and are untouched by this change `[VERIFIED — re-run on main]`.
