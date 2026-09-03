# A4-T1: temporary zero-trade fallback override (2026-08-31..09-07)   (PR #128)

STATUS:    delivered — v13 of the RFC#210 A4-T1 amendment, TIME-LIMITED
           [2026-08-31, 2026-09-07], FAIL-CLOSED; supersedes #127.
WHAT:      backtesting IDENTIFIES the one authorized candidate (exact run id +
           full-artifact digest) and VALIDATES a versioned consumption proof
           (`a4t1_consumption_proof.v1`: 8 bound fields + `receipt_id`) before
           `stamp()` touches the artifact; the direct CLI refuses to stamp a
           candidate (exit 1); the authority constant names the orchestrator's
           committed governance record and the local JSON is a pointer.
WHY/DIR:   closes the codex #127 blocker (`stamp()` accepted any non-empty dict
           as proof, so single consumption was not enforced). Direction:
           governance state (authorization record, ledger, atomic single
           consumption) is orchestrator-owned; backtesting stays the fail-closed
           verdict + proof validator. Paired with renquant-orchestrator#1110.
EVIDENCE:  `tests/test_freshness_fallback.py` 100 passed (52 standing A4 + 13
           structural + 35 candidate/proof) [VERIFIED — pytest 2026-09-03 at
           885d3ce, the last code-touching commit]. No model-quality claim is
           made; the artifact facts below are identification inputs only.
           artifact:      `backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.weekly_20260831T141820Z.staging.json` (umbrella live tree)
           prod or exp:   prod (staging candidate for the production pair promotion)
           existing data: canonical-JSON SHA-256 recomputed from the live artifact = `760912ec…4af1e`, equal to `_A4T1_CANDIDATE_DIGEST` and to the orchestrator authorization record [VERIFIED — python hashlib + record read, 2026-09-03]; genuine_ic 0.001554 (`sanity_placebo_genuine_ic`) and `wf_reason` = zero trades across all 3 WF cuts [VERIFIED — `metadata.wf_gate_metadata` of the live artifact read 2026-09-03]; regime_sanity_ic failed [VERIFIED — prior work, §Evidence below / bt#123–#127]
           best-known?:   no — a zero-trade candidate the standing A4 policy REFUSES; the served prod model (trained 2026-08-02) is the best-known artifact but lapsed the 28-day SLA
           scope:         "this is the 20260831T141820Z staging artifact, prod, vs the served 2026-08-02 model — a time-limited, operator-authorized fallback, not a signal claim"
NEXT:      merge this first; then renquant-orchestrator#1110 (record + data-root
           ledger + `promote_candidate` wrapper); then the umbrella PR (pins +
           `weekly_wf_promote.sh --promote-staged` rewired to the wrapper);
           remove the A4-T1 block at expiry (2026-09-07) per §Restore condition.

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

### Path 2 — Candidate exception (v13)

`_is_a4t1_candidate(staging_path, staging)` binds to:
- EXACT run-ID: `20260831T141820Z`
- FULL-ARTIFACT SHA-256 digest: `760912ec...4af1e`

Covers the actual staging artifact (5 substance classes: 4 zero-trade +
regime_sanity_ic) that the structural path rejects.

Division of labour (codex v11/v12 findings applied):

- **Backtesting IDENTIFIES** the candidate and **VALIDATES the consumption
  proof** it is handed. It exports the proof contract the orchestrator
  imports (names frozen): `A4T1_PROOF_SCHEMA = "a4t1_consumption_proof.v1"`,
  `A4T1_PROOF_FIELDS` (schema, exception_id, run_id, artifact_digest,
  authority, consumed_at, consumed_by, ledger_path), `A4T1_PROOF_KEYS`
  (+ receipt_id), `A4T1_CONSUMER = "renquant-orchestrator"`,
  `a4t1_receipt_id(proof)` = sha256 of the canonical JSON of the 8 bound
  fields, `validate_a4t1_proof(proof, verdict)`.
- **`stamp()`** on a candidate verdict validates the proof BEFORE touching
  the artifact: exact key set, non-empty strings, schema, run_id /
  exception_id / artifact_digest / authority equal to the verdict,
  consumed_by == the orchestrator, timezone-aware consumed_at, and
  receipt_id == recomputed receipt (an edited proof no longer matches its
  own receipt). A proof on a NON-candidate verdict is refused. Each defect
  has its own reason string.
- **The direct CLI refuses to stamp a candidate** (`--stamp` on a candidate
  verdict prints `stamp_refused =
  a4t1_candidate_requires_orchestrator_consumption`, `stamped: false`, exit
  1). The umbrella `weekly_wf_promote.sh --promote-staged` keys pair
  promotion off exit 0, so the pre-existing shell path is fail-closed on
  the candidate until it is rewired to the orchestrator wrapper.
- **The ORCHESTRATOR owns** the authorization record
  (`renquant-orchestrator:ops/governance/a4t1/20260831T141820Z.authorization.json`
  — the value of `_A4T1_CANDIDATE_AUTHORITY`), the ledger, atomic
  single-consumption and the ONLY producer of proofs
  (`renquant_orchestrator.a4t1_governance.promote_candidate`: identify →
  atomic consume → stamp). `doc/governance/a4t1-candidate-exception-authority.json`
  in this repo is reduced to a cross-repo pointer.

**What this enforces / what it cannot.** This module can reject malformed,
unbound or fabricated proofs and refuses to stamp candidates from the CLI,
so no shell caller and no lazy Python caller can promote the candidate
without a proof shaped exactly like the orchestrator's. It cannot verify
that a receipt exists in the ledger — that knowledge is the orchestrator's,
and the audit trail is the ledger receipt the orchestrator writes before
calling `stamp()`. A Python caller that deliberately reconstructs a valid
proof is not stopped here; it is visible, because the stamped artifact
carries a receipt_id that the ledger does not.

Standing A4 constants UNCHANGED. The bypass is a separate code path.

## Tests

100 tests total (52 standing A4 + 13 structural + 35 candidate exception)
[VERIFIED — pytest 2026-09-03]:
- Identification (5): happy path promotes within window; tampered
  wf_gate_metadata / tampered trained_date (digest mismatch) refuse;
  wrong run-ID refuses; expired refuses.
- Proof rejected, artifact byte-identical after each (26): None; `{"x": 1}`
  (the v12 hole); `{}`; every one of the 9 keys missing (parametrized);
  an extra key; each binding field defective with the receipt recomputed so
  the field check fires on its own (schema, run_id, exception_id,
  artifact_digest, authority, consumed_by, naive consumed_at, non-ISO
  consumed_at, empty ledger_path); a non-string value; receipt edited;
  a bound field edited without recomputing the receipt.
- Proof accepted (2): valid proof stamps run_id / digest / authority / the
  full proof; a self-consistent proof with a different consumed_at is
  accepted (binding is to the verdict, not to time).
- Boundary (2): a proof on a non-candidate promotion is refused; receipt id
  is canonical (key order, presence of receipt_id irrelevant; different
  ledger_path → different receipt).
- CLI (1): `--stamp` on the candidate exits 1 with `stamp_refused`, artifact
  untouched.

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
- v11 (bt#126): canonical ledger at `~/.renquant/governance/` (module
  constant, not caller parameter), committed governance authority file
  `doc/governance/a4t1-candidate-exception-authority.json`. 75 tests
  (52+13+10). Codex rejected: hardcoded user-home dir is host-local,
  decide() refuses before stamp() can create dir (first-use deadlock),
  orchestrator should own governance state, authority JSON is
  self-attestation.
- v12 (bt#127): split concerns — backtesting identifies only,
  orchestrator owns consumption; stamp() requires a4t1_consumption_proof
  from the caller. 73 tests (52+13+8). Codex rejected: any non-empty dict
  passed as proof (`{"x": 1}` stamped the candidate), so the boundary did
  not enforce single consumption; authority JSON still self-attestation.
- v13 (this PR): versioned proof contract exported (schema v1, 8 bound
  fields + receipt_id = sha256 of their canonical JSON); stamp() validates
  structure and exact binding to the verdict before touching the artifact,
  with a distinct reason per defect; a proof on a non-candidate verdict is
  refused; the direct CLI refuses to stamp candidates (exit 1); the
  authority constant now names the orchestrator governance record and the
  local JSON is a pointer. 100 tests (52+13+35). Paired orchestrator PR
  carries the authorization record, the data-root ledger, the narrow
  identify→consume→stamp wrapper and the e2e / concurrency tests.

## Operator authorization

- Structural path: orchestrator session 428feb92, 2026-08-31.
- Candidate exception + regime_sanity_ic bypass: orchestrator session
  428feb92, 2026-09-02 (explicit "go" in response to authorization request).
