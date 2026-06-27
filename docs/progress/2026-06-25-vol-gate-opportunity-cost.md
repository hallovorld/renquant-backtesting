# Vol-gate opportunity-cost — EXPLORATORY diagnostic (no config change)

2026-06-25. Trigger: 2026-06-25 daily-full no-trade (`RealizedVolGateTask` dropped 21/97
candidates over the 60% vol cap). Operator: high-vol is opportunity too — but with theory +
rigorous data. Research/discussion only — **NO behavior change, and NO config proposal.**

## What this is
An exploratory diagnostic of whether the hard 60% realized-vol admission cap is conservative
given the downstream `1/σ²` Kelly sizing. The honest verdict is **inconclusive** and explicitly
**not** a basis for a config change.

## Re-homing (why this PR is in renquant-backtesting)
Originally opened as orchestrator PR #194. Model-fitting research violates a hard
`renquant-orchestrator` CLAUDE.md boundary (*"Do not implement model training internals here"*),
so it is re-homed to `renquant-backtesting` (walk-forward / forensics owner), same as a prior
re-home (#193). Orchestrator PR #194 is closed pointing here.

## Deliverables
- `research/research_vol_gate_opportunity_cost.py` — the experiment (XGB proxy ranker, purged WF,
  monthly 1/σ² book, cap sweep, regime slice, bootstrap CIs, robustness).
- `tests/test_research_vol_gate.py` — pure helpers tested: TRADING-SESSION fold embargo (the
  leakage fix), regime-uniformity fail-closed, bootstrap CI, metrics. 7 tests, all pass.
- `docs/research/2026-06-25-vol-gate-opportunity-cost.md` — theory, survivorship caveat,
  leakage-fix note, regime-uniformity note, FDR note, results, conclusion.

## Fixes applied during the move (Codex review)
1. **Boundary** — experiment + tests now in renquant-backtesting (done by the move).
2. **Leakage (the key bug)** — `fwd_60d_excess` = `close.shift(-60)` = 60 *trading sessions*, but
   the old purge subtracted `Timedelta(days=60)` ≈ 42 trading sessions → labels near the cutoff
   overlapped the test interval. Now embargoed by **trading sessions** on the sorted unique-date
   index, with **STRICT non-overlap**: `EMB_SESSIONS = horizon + 1 = 61` so the last training
   label ends at `lo_i − 1`, strictly before `test_lo` (a 60-session embargo would have touched
   the boundary). Matches the production WF-gate `data_end = cutoff − BDay(lookahead)` strict cut.
   `purged_test_windows` raises if `embargo ≤ horizon` (fail closed). Regression tests assert the
   last-label-end index is strictly `<` first-test index and pin the fail-closed rejection.
3. **Regime uniformity** — `assert_regime_uniform_per_date` now **raises** if any date has
   conflicting per-name regime labels (was a silent `.groupby().first()`). Tested both ways.
4. **Multiple comparisons** — added a doc note: the six-cap CIs are per-comparison; the no-change
   conclusion is conservative, but any FUTURE positive inference must pre-register the primary cap
   or apply a family-wise / FDR correction.

## Honest findings (re-run under the fixed strict 61-session purge, 2026-06-27)
- Point estimate: relaxing 0.6→1.0 raises Sharpe (+0.179→+0.642) without raising drawdown
  (maxDD −15.9%→−9.6%).
- **But the paired block-bootstrap CI for the 0.6-vs-1.0 monthly delta INCLUDES ZERO**
  (+0.00283/mo, 95% CI [−0.00019, +0.00711]) → **not significant** (and before any FDR correction).
- By **actual regime**: helps in BULL_CALM (n=42) and BULL_VOLATILE (n=47); **BEAR is n=3 →
  unmeasurable** (the earlier "cap helps in bear" was a calendar-period artifact — withdrawn).
- Panel is **survivorship-biased** (291/291 survive to 2026); proxy XGB ranker, not live PatchTST
  in the real sizing/QP/gate stack. These are the re-run figures, replacing the old leaked-purge
  table; they moved in both directions (Sharpe fell, drawdown improved), confirming the purge fix
  is NOT monotone — but the inconclusive verdict (CIs bracket 0) holds.

## Conclusion
**No config change supported.** A real decision needs: a re-run under the fixed 60-session purge
+ a PIT universe with delistings + live PatchTST + the real Kelly/QP/gate order + paired
net/DD/turnover deltas with uncertainty (FDR-corrected) → shadow-test before any production change.

## Note
The §4 numbers are a full re-run under the fixed strict 61-session purge on the umbrella panel
(OOS rows = 549,306). The earlier draft's "the leakage fix can only *lower* OOS scores" claim was
**wrong and is withdrawn** — changing the purge changes the training set/fit/rankings/selections,
and several metrics (e.g. drawdown) moved the other way. The verdict is unchanged only because the
bootstrap CIs still bracket zero, not because the fix is monotone.
