# A4-T1 TEMPORARY OVERRIDE — fallback floor lowered + infra set expanded

STATUS: operator-directed temporary amendment to RFC#210 A4 (2026-08-31).

WHAT: Two changes to `freshness_fallback.py`:
1. `FALLBACK_GENUINE_IC_FLOOR` lowered from `PLACEBO_GENUINE_IC_MARGIN` (0.02)
   to 0.001.
2. `INFRA_ONLY_FAILURE_CLASSES` expanded from `{placebo_ceiling}` to
   `{placebo_ceiling, wf_benchmark_economics, trade_contract,
   trade_monotonicity, alpha_economics}`.

WHY: The served production model lapsed on day 29 (trained 2026-08-02, SLA
28 days). The current recipe produces zero trades across all 3 WF cuts, so:
- Check 4 (quality floor): candidate genuine_ic=+0.0016 < 0.02 floor → REFUSE
- Check 5 (failure classes): wf_benchmark_economics + trade_contract +
  trade_monotonicity + alpha_economics are all SUBSTANCE → REFUSE

Both gates must be relaxed for the promote chain to unblock the model.

EVIDENCE (all read-only, 2026-08-31):
- Fallback verdict: `20260831T141820Z.fallback_verdict.json` — refused_on
  quality_floor, genuine_ic=0.001554 < 0.02 [VERIFIED].
- Staging WF stamp: passed=False, wf_reason="FAIL: zero trades across all
  WF cuts; decision tree admitted no buys" [VERIFIED].
- Served model: trained_date 2026-08-02, staleness 29 days > 28d SLA
  [VERIFIED].

RESTORE CONDITION: Revert to A4 values (floor = PLACEBO_GENUINE_IC_MARGIN,
infra = {placebo_ceiling} only) when a recipe+model combination produces
non-zero WF trades AND the candidate meets genuine_ic >= 0.02.

ROLLBACK: Single-commit revert of this PR + pin re-advance.

TESTS: 52 passed in test_freshness_fallback.py (all assertions updated to
match the A4-T1 constants).
