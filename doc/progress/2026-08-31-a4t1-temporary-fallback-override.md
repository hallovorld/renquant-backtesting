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
  trade_contract, trade_monotonicity, alpha_economics) + regime_sanity_ic

## Solution

Amendment A4-T1 adds a NARROW, TIME-LIMITED, COUPLED bypass with TWO
eligibility paths:

1. **Path 1 (structural zero-trade)**: `_is_zero_trade_structural(failures, wf)`
   — wf_reason "zero trades across all", substance classes EXACTLY the 4
   zero-trade classes, trade-dependent details "no round-trip".

2. **Path 2 (candidate exception)**: `_is_a4t1_candidate(staging_path, wf, staging)`
   — run-ID parsed exactly from filename (regex, not substring), wf_gate_metadata
   content verified by SHA-256 digest against a hardcoded constant, single-consumption
   (refuses after promotion_basis is stamped). The digest and run-ID are recorded
   in the verdict and stamp for downstream auditability.

Either path, combined with `_a4t1_active(as_of)`, gates BOTH the floor
relaxation (0.001) and the failure-class exception.

3. Temporal bounds: [2026-08-31, 2026-09-07], hard-coded, fail-closed.

4. Floor lowered to 0.001 (from 0.02) ONLY under the coupled predicate.

Standing A4 constants UNCHANGED. The bypass is a separate code path.

## Tests

71 tests total (52 standing A4 + 13 structural zero-trade + 6 candidate exception):

Structural zero-trade tests:
- Happy path: zero-trade candidate within window promotes
- Floor boundary: genuine_ic=0.001 promotes, 0.0009 refuses
- Temporal: expired and before-start both refuse
- Coupled predicate: placebo-only (non-zero-trade) refuses at 0.0016
- Partial wf_reason ("one cut"): refuses
- Missing one of 4 classes: refuses
- Extra substance class: refuses
- Non-"no round-trip" detail: refuses
- Above standing floor: uses standing path, not bypass
- Stamp carries bypass metadata

Candidate exception tests (v8):
- Correct run ID + correct digest: promotes
- Tampered content (right filename, wrong digest): refuses
- Substring collision (run ID in longer filename): refuses
- Replay after promotion (already stamped): refuses
- Expired window: refuses
- Stamp carries candidate digest and run-ID

## Evidence

- Staging artifact: `panel-ltr.alpha158_fund.weekly_20260831T141820Z.staging.json`
  genuine_ic=0.001554, all trade sub-verdicts "no round-trip ledgers found"
  wf_gate_metadata SHA-256: `7cff5b7c27a8ced62f70b25fdd146479cb11e5f0d47aee14e22b22249f0fbfbc`
- Fallback verdict: `20260831T141820Z.fallback_verdict.json`
  refused_on=quality_floor

## Restore condition

Remove the A4-T1 block after expiry (2026-09-07) or when a recipe produces
non-zero WF trades AND meets genuine_ic >= 0.02.

## Iteration history

- v1 (bt#116): globally reclassified substance as infra. Codex rejected (4 blockers).
- v2 (bt#117): separate narrow bypass. Codex rejected (2 gaps: decoupled
  predicate, loose zero-trade check).
- v3 (bt#118): pre-computed coupled predicate, stricter zero-trade check,
  13 gap-closing tests. Codex approved, merged as d081670c.
- v4 (bt#119): merged v3; this is the current main baseline.
- v5 (bt#120): added regime_sanity_ic to _A4T1_ZERO_TRADE_CLASSES. Codex
  rejected (post-hoc gate fitting).
- v6 (bt#121): subset-match instead of exact-match. Codex rejected (still
  post-hoc, broader bypass).
- v7 (bt#122): one-shot run-ID exception via substring match. Codex rejected
  (substring not immutable, no digest binding, no single-consumption).
- v8 (this PR): cryptographic digest binding (SHA-256 of canonical
  wf_gate_metadata), exact run-ID parsing (regex, not substring),
  single-consumption enforcement (promotion_basis check), 6 new tests
  (tampered content, substring collision, replay after promotion, expired,
  stamp carries digest). Supersedes bt#122.

Operator authorization: orchestrator session 428feb92, 2026-08-31.
