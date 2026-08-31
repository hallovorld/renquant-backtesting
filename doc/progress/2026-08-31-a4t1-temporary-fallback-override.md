# A4-T1 TEMPORARY BYPASS — narrow, time-limited fallback override

STATUS: operator-directed temporary bypass of RFC#210 A4 (2026-08-31).
Supersedes bt#116 (v1 was globally reclassifying — codex CHANGES_REQUESTED).

WHAT: A separate, narrow bypass in `freshness_fallback.py` that:
1. Is active only within a hard date window: 2026-08-31 to 2026-09-07
2. Lowers the quality floor from 0.02 to 0.001, ONLY when A4-T1 is active
3. Accepts zero-trade WF failures, ONLY when a parsed predicate confirms:
   - `wf_reason` explicitly names "zero trades" (anti-vacuity)
   - Every substance failure class is in the zero-trade set
   - `trade_contract`/`trade_monotonicity`/`alpha_economics` detail mentions
     "no round-trip" (anti-vacuity: absence-of-data, not quality failure)
4. Expires automatically (fail-closed) after 2026-09-07

The standing A4 constants are UNCHANGED:
- `FALLBACK_GENUINE_IC_FLOOR` = `PLACEBO_GENUINE_IC_MARGIN` (0.02)
- `INFRA_ONLY_FAILURE_CLASSES` = `{placebo_ceiling}`

WHY: The served production model lapsed on day 29 (trained 2026-08-02,
SLA 28 days). The current recipe produces zero trades across all 3 WF
cuts, so both the quality floor (genuine_ic=0.0016 < 0.02) and the
substance-class gate block the fallback.

OPERATOR AUTHORIZATION: Orchestrator session 428feb92-8ee7-4b4f-afed-
1e4fa82ef367, 2026-08-31. Operator chose option 3 from three options
presented after manual promote showed the double block. Option 3:
"临时降低 A4 quality floor — 比如 genuine_ic >= 0.001". Operator
confirmation: "3".

RESTORE: Remove the A4-T1 block (constants + `_a4t1_active` +
`_is_zero_trade_structural` + the two bypass paths in `decide()` +
the A4-T1 tests) when a recipe+model produces non-zero WF trades AND
meets genuine_ic >= 0.02, or after the expiry date (2026-09-07).

TESTS: 60 passed (52 existing A4 + 8 new A4-T1 tests covering:
promote, expiry fail-closed, below-floor refuse, non-zero-trade refuse,
wf-reason mismatch refuse, detail mismatch refuse, constants unchanged,
window bounds).
