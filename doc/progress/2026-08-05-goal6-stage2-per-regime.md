# 2026-08-05 — GOAL-6 Stage 2: per-regime scoring INSTRUMENTATION (not yet a selection rule)

## STATUS

The Stage-2 candidate-scoring lane gains per-regime **instrumentation**. Nothing
selects, ranks or decides on it yet, and this document does not claim otherwise
`[codex on bt#107]`: the runner does not supply a regime map, and no verdict
consumes `by_regime` or `pooled_sign_carriers`. **Production scoring still uses
the pooled `mean_ic`.** Making the pooled figure visibly decomposable is the
prerequisite; a regime-aware selection rule is a separately designed change that
needs the exposure/trading-regime input decided first.

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
  stamps `by_regime` (per regime: `n_dates`, `mean_ic`, `min_ic`, `max_ic`,
  `weight`, `contribution_to_pooled_ic`), tags each `per_date` row with its
  regime, and reports **`pooled_sign_carriers`** — the regimes whose REMOVAL
  flips the pooled sign.
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
- **Unassigned dates are never reported as pooled-sign carriers.** A bucket of
  dates the caller could not label is not a regime; naming it would hand an
  unknown-label bucket the interpretation. A regression constructs the case
  where removing that bucket WOULD flip the pooled sign and asserts it is still
  not named.
- **The seam is unchanged when no map is given**: the regime map is passed as a
  keyword and only when supplied, so the two-argument call contract other
  callers and test doubles rely on still holds.
- **The pooled FIELDS are unchanged** with and without the split (`mean_ic`,
  `n_dates_scored`, `n_dates_with_labels`, and every per-date IC and date).
  Supplying a map does add a `regime` key to each `per_date` row and new
  top-level keys — an earlier draft of this doc said "bit-identical output",
  which was too strong `[codex on bt#107]`. A test pins that the ONLY added
  per-date key is `regime`.

## NOT done

The gate's own call site does not yet pass a regime map — that is a separate
change to the runner, and wiring it there means deciding which regime series the
Stage-2 window uses (the production chain per scored date). Filed as the next
step rather than guessed at here.

### The flag I had to throw away

The first version reported `pooled_is_a_regime_mix` — true when the pooled sign
disagreed with EVERY regime. **That is arithmetically impossible** once dates are
assigned: the pooled mean is a date-weighted average of the regime means, so it
must lie between them. The flag was dead code for the exact shape it was written
for, and my test never asserted the true case, so it passed `[codex on bt#107]`.

The live shape is not sign-disagreement, it is **dominance**: a small,
high-|IC| regime supplying the pooled sign while the regime that carries the
trading has the opposite one. So the summary now decomposes — `weight` and
`contribution_to_pooled_ic` per regime, which sum to the pooled mean (pinned by
a test), plus `pooled_sign_carriers`, the regimes whose removal flips the pooled
sign. On the live proportions that is BEAR at ~12% of dates.

Suites: 16 tests. The 3 failures on this branch (`test_b1_lift`,
`test_import_lift`, `test_lineage_stage2::test_REAL_run001_...`) all reproduce on
`origin/main` and are untouched by this change `[VERIFIED — re-run on main]`.
