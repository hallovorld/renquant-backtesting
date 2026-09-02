# A4-T1: temporary zero-trade fallback override (2026-08-31..09-07)

STATUS: amendment to RFC#210 A4, TIME-LIMITED, FAIL-CLOSED

## Problem

The served production model (trained 2026-08-02) lapsed on day 29 of the
RFC#210 28-day SLA. The weekly retrain produced a candidate that is a
complete zero-trade WF outcome: the decision tree admitted no buys across
all 3 WF cuts, Sharpe is undefined, and the SPY benchmark cannot be met.

The standing A4 policy correctly REFUSES this candidate:
- genuine_ic = 0.001554 < 0.02 floor (quality_floor_not_met)
- 4 substance failure classes from zero-trade outcomes (wf_benchmark_economics,
  trade_contract, trade_monotonicity, alpha_economics)

The staging artifact also carries a 5th substance failure (regime_sanity_ic),
which prevents the structural zero-trade predicate from matching (it requires
EXACTLY the 4 zero-trade classes, not 5).

## Solution

Amendment A4-T1 provides TWO eligibility paths, both gated by temporal
bounds [2026-08-31, 2026-09-07]:

### Path 1 — Structural

`_is_zero_trade_structural(failures, wf)` requires ALL of:
- wf_reason contains "zero trades across all" (not partial)
- substance classes are EXACTLY the 4 zero-trade classes (not a subset)
- trade-dependent details report "no round-trip" (absence-of-data)

Gates BOTH the floor relaxation (0.001) and the failure-class exception.

### Path 2 — Candidate exception (v10)

`_is_a4t1_candidate(staging_path, staging, ledger_dir)` binds to:
- EXACT run-ID: `20260831T141820Z`
- FULL-ARTIFACT SHA-256 digest: `760912ec...4af1e`
- Global consumption ledger (fail-closed without one)

Covers the actual staging artifact (5 substance classes: 4 zero-trade +
regime_sanity_ic) that the structural path rejects.

Consumption model: atomic O_CREAT|O_EXCL file marker in a CANONICAL ledger
at `~/.renquant/governance/` (module constant `_A4T1_LEDGER_DIR`), keyed by
`a4t1_{RUN_ID}.consumed`. The location is NOT a caller-supplied parameter —
it is a hardcoded constant, preventing ledger substitution. Written in
`stamp()` BEFORE the staging artifact is modified. Cross-directory replay
is impossible because all callers use the same canonical location.
Corrupt/unreadable marker → fail-closed (file exists = consumed).
Nonexistent ledger dir → fail-closed (candidate exception refuses).

Standing A4 constants UNCHANGED. The bypass is a separate code path.

## Tests

75 tests total (52 standing A4 + 13 structural + 10 candidate exception):
- Candidate happy path: correct run-ID + digest within window promotes
- Tampered wf_gate_metadata: digest mismatch refuses
- Tampered trained_date: digest mismatch refuses
- Wrong run-ID: refuses
- Cross-directory replay: canonical ledger blocks (codex blocker fix)
- Expired: refuses
- Stamp carries digest + authority + governance file reference
- No ledger dir: fail-closed refuses (codex blocker fix)
- Corrupt ledger marker: fail-closed refuses (codex blocker fix)
- Ledger not substitutable: two artifact dirs share one canonical ledger

## Evidence

- Staging artifact: `panel-ltr.alpha158_fund.weekly_20260831T141820Z.staging.json`
  genuine_ic=0.001554, all trade sub-verdicts "no round-trip ledgers found",
  regime_sanity_ic failed
- Full-artifact SHA-256: `760912ec122fa6e02628077df8b35e58145209ea3b6b395bd670d8ead9e4af1e`

## Restore condition

Remove the A4-T1 block after expiry (2026-09-07) or when a recipe produces
non-zero WF trades AND meets genuine_ic >= 0.02.

## Iteration history

- v1 (bt#116): globally reclassified substance as infra. Codex rejected (4 blockers).
- v2 (bt#117): separate narrow bypass. Codex rejected (2 gaps: decoupled
  predicate, loose zero-trade check).
- v3 (bt#118): pre-computed coupled predicate, stricter zero-trade check,
  13 gap-closing tests. Codex approved, merged.
- v4-v7: incremental cleanups merged to main.
- v8 (bt#123): candidate exception with wf_gate_metadata digest. Codex
  rejected (wf_gate_metadata-only digest doesn't cover trained_date;
  promotion_basis check not atomic).
- v9 (bt#124): full-artifact digest + directory-level marker. Codex rejected
  (directory-local marker not a global ledger; corrupt state fails open;
  regime_sanity_ic authority scope gap).
- v10 (bt#125): global consumption ledger with O_CREAT|O_EXCL atomic marker,
  fail-closed on corrupt state, separate operator authorization for
  regime_sanity_ic. Codex rejected: ledger path caller-substitutable,
  authority not independently verifiable.
- v11 (this PR): canonical ledger at `~/.renquant/governance/` (module
  constant, not caller parameter), committed governance authority file
  `doc/governance/a4t1-candidate-exception-authority.json`. 75 tests
  (52+13+10).

## Operator authorization

- Structural path: orchestrator session 428feb92, 2026-08-31.
- Candidate exception + regime_sanity_ic bypass: orchestrator session
  428feb92, 2026-09-02 (explicit "go" in response to authorization request).
