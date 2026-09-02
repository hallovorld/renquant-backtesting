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

Two layers, both time-limited and fail-closed:

### Layer 1: zero-trade structural bypass (v4, merged as bt#119)

ONE pre-computed eligibility predicate (`_a4t1_active(as_of) and
_is_zero_trade_structural(failures, wf)`) gates BOTH the floor relaxation
(0.001) and the failure-class exception. Requires EXACT match of the 4
trade-derived zero-trade classes — no extras tolerated. This covers
pure zero-trade candidates with only those 4 classes.

### Layer 2: one-shot candidate exception (v7, this PR)

The actual staging artifact (run `20260831T141820Z`) has 5 substance
classes: the 4 trade-derived plus `regime_sanity_ic`. The exact-match
predicate correctly excludes it. A ONE-SHOT exception, bound to the
immutable run ID `20260831T141820Z` and the A4-T1 temporal window,
bypasses the predicate for this single artifact. `regime_sanity_ic`
remains substance for EVERY other candidate.

Standing A4 constants UNCHANGED. The bypass is a separate code path.

## Tests

70 tests total (52 standing A4 + 13 zero-trade predicate + 5 candidate exception):
- Zero-trade happy path: 4-class candidate within window promotes
- Candidate exception: 5-class artifact with correct run ID promotes
- Wrong run ID: refuses (exception is bound to ONE artifact)
- Expired: refuses
- Below floor: refuses even with correct run ID
- Stamp metadata: records candidate-specific authorization
- All v4 tests preserved: exact-match, coupling, temporal bounds, etc.

## Evidence

- Staging artifact: `panel-ltr.alpha158_fund.weekly_20260831T141820Z.staging.json`
  genuine_ic=0.001554, 5 substance classes (4 trade-derived + regime_sanity_ic),
  all trade sub-verdicts "no round-trip ledgers found"
- v5 codex review: regime_sanity_ic is NOT trade-derived; adding to set is post-hoc
- v6 codex review: subset-match is same post-hoc problem, just broader
- v7 solution: one-shot run-ID exception, preserves exact-match safeguard

## Restore condition

Remove the A4-T1 block after expiry (2026-09-07) or when a recipe produces
non-zero WF trades AND meets genuine_ic >= 0.02.

## Iteration history

- v1 (bt#116): globally reclassified substance as infra. Codex rejected (4 blockers).
- v2 (bt#117): separate narrow bypass. Codex rejected (2 gaps: decoupled
  predicate, loose zero-trade check).
- v3 (bt#118): pre-computed coupled predicate, stricter zero-trade check.
  Codex rejected (stamp recorded standing floor, not effective).
- v4 (bt#119): stamp audit fix. Codex APPROVED, MERGED (d081670c).
- v5 (bt#120): added regime_sanity_ic to zero-trade class set. Codex
  rejected: not trade-derived, post-hoc gate fitting.
- v6 (bt#121): subset-match. Codex rejected: same post-hoc problem, broader.
- v7 (this PR): one-shot candidate exception bound to immutable run ID
  `20260831T141820Z`. Exact-match preserved. regime_sanity_ic stays substance.

Operator authorization:
- Layer 1 (zero-trade bypass): orch-session-428feb92-2026-08-31
- Layer 2 (candidate exception): orch-session-428feb92-2026-09-01
