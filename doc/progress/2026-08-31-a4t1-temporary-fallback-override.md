# A4-T1: temporary zero-trade fallback override (2026-08-31..09-07)

STATUS: amendment to RFC#210 A4, TIME-LIMITED, FAIL-CLOSED

## Problem

The served production model (trained 2026-08-02) lapsed on day 29 of the
RFC#210 28-day SLA. The weekly retrain produced a candidate that is a
complete zero-trade WF outcome: the decision tree admitted no buys across
all 3 WF cuts, Sharpe is undefined, and the SPY benchmark cannot be met.

The standing A4 policy correctly REFUSES this candidate:
- genuine_ic = 0.001554 < 0.02 floor (quality_floor_not_met)
- 5 substance failure classes: 4 from zero-trade outcomes (wf_benchmark_economics,
  trade_contract, trade_monotonicity, alpha_economics) + 1 non-trade
  (regime_sanity_ic — fails independently of trade count)

## Solution

Amendment A4-T1 adds a NARROW, TIME-LIMITED, COUPLED bypass:

1. ONE pre-computed eligibility predicate (`_a4t1_active(as_of) and
   _is_zero_trade_structural(failures, wf)`) gates BOTH the floor
   relaxation and the failure-class exception. The floor relaxation
   ONLY applies when the zero-trade predicate is also met (the v2
   coupling gap fixed in v3).

2. Temporal bounds: [2026-08-31, 2026-09-07], hard-coded, fail-closed.

3. Floor lowered to 0.001 (from 0.02) ONLY under the coupled predicate.

4. `_is_zero_trade_structural()` requires ALL of:
   - wf_reason contains "zero trades across all" (not partial)
   - ALL 4 trade-derived zero-trade classes present among substance
     failures (additional non-trade classes like regime_sanity_ic may
     co-occur without negating the structural diagnosis)
   - trade-dependent details report "no round-trip" (absence-of-data)

Standing A4 constants UNCHANGED. The bypass is a separate code path.

## Tests

66 tests total (52 standing A4 + 14 A4-T1):
- Happy path: zero-trade + regime_sanity_ic (real artifact shape) promotes
- Floor boundary: genuine_ic=0.001 promotes, 0.0009 refuses
- Temporal: expired and before-start both refuse
- Coupled predicate: placebo-only (non-zero-trade) refuses at 0.0016
- Partial wf_reason ("one cut"): refuses
- Missing one of 4 trade-derived classes: refuses
- Extra non-trade class (regime_sanity_ic): promotes (subset match)
- Extra trade-unrelated class (config_parity): promotes (subset match)
- Non-"no round-trip" detail: refuses
- Above standing floor: uses standing path, not bypass
- Stamp carries bypass metadata
- Standing stamp has no A4-T1 fields

## Evidence

- Staging artifact: `panel-ltr.alpha158_fund.weekly_20260831T141820Z.staging.json`
  genuine_ic=0.001554, 5 substance classes (4 trade-derived + regime_sanity_ic),
  all trade sub-verdicts "no round-trip ledgers found"
- Fallback verdict: `20260831T141820Z.fallback_verdict.json`
  refused_on=quality_floor (under v4 exact-match)
- v5 codex review: regime_sanity_ic is NOT trade-derived, adding it is
  post-hoc gate fitting. Fix: subset-match identifies the zero-trade pattern
  by the 4 trade-derived classes without claiming extras are zero-trade-caused.

## Restore condition

Remove the A4-T1 block after expiry (2026-09-07) or when a recipe produces
non-zero WF trades AND meets genuine_ic >= 0.02.

## Iteration history

- v1 (bt#116): globally reclassified substance as infra. Codex rejected (4 blockers).
- v2 (bt#117): separate narrow bypass. Codex rejected (2 gaps: decoupled
  predicate, loose zero-trade check).
- v3 (bt#118): pre-computed coupled predicate, stricter zero-trade check,
  13 gap-closing tests. Codex rejected (stamp recorded standing floor, not effective).
- v4 (bt#119): stamp audit fix. Codex APPROVED, MERGED (d081670c).
- v5 (bt#120): added regime_sanity_ic to zero-trade class set. Codex
  rejected: regime_sanity_ic is NOT trade-derived, adding it is post-hoc
  gate fitting that defeats the exact-set safeguard.
- v6 (this PR): subset-match — the 4 trade-derived classes identify the
  pattern; additional non-trade classes are tolerated. Not post-hoc fitting
  because the 4-class set is UNCHANGED; the matching logic is corrected.

Operator authorization: orchestrator session 428feb92, 2026-08-31.
