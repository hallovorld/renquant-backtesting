# WF Gate Incumbent Regression Check

**Date:** 2026-07-05
**PR:** #70
**Status:** Design RFC — round 3, addressing Codex review

## What

Replace the WF gate's hard SPY benchmark requirement with an incumbent regression
check: candidate must not be significantly worse than the current production model.
SPY comparison becomes advisory-only (logged, not gated).

## Why

In bull markets (SPY Sharpe >1.0), the stock-selection strategy structurally
cannot beat SPY — every promote requires operator override. The right question is
"is the new model at least as good as the old one?" not "does it beat the index?"

## Key decisions

- `absolute_ok` (mean Sharpe >= 0.40, 2/3 cuts positive): **unchanged, hard gate**
- `benchmark_ok` (beat SPY): **demoted to advisory** (still logged)
- `regime_ok` (beat SPY per regime): **demoted to advisory** (still logged)
- `incumbent_ok` (new): **hard gate** — candidate_sharpe >= incumbent - 0.15,
  candidate_apy >= incumbent - 5pp. Auto-pass if no incumbent.

## Validation

The 2026-07-05 retrain (Sharpe 0.776) would PASS under this gate: incumbent
Sharpe = 0.697, delta = +0.079 (improvement, not regression).

## Round 2 (codex review)

STATUS: fixed (RFC revised)
WHAT: codex raised four issues: (1) demoting regime protection to advisory with
no equally-strong replacement, changing the failure mode from too-strict to
too-permissive; (2) the incumbent itself may be a policy exception (promoted
with override), so comparing against it risks institutionalizing a waived
standard; (3) `sharpe_tolerance=0.15`/`apy_tolerance=0.05` were asserted, not
justified; (4) mean-only comparison can hide a one-cut blowup.
WHY-DIR: all four are real methodology gaps in a gate that decides future model
promotions — a too-permissive redesign here has real downstream cost.
EVIDENCE:
- (2) Defined a concrete, checkable admissible-incumbent policy using the
  already-persisted `diagnostic_only`/`skipped_required_gates` fields. When
  the incumbent isn't admissible, the gate falls back to the pre-existing
  `absolute_ok` floor alone — not a silent auto-pass. Verified directly
  against the live production artifact
  (`artifacts/prod/panel-ltr.alpha158_fund.json`): its stamped
  `wf_gate_metadata` currently shows the *fresh retrain's* numbers
  (`wf_3cut_sharpe_mean=0.776`), not the original 2026-06-21 promotion's —
  which surfaced a genuine, previously unaddressed data-integrity gap:
  `run_wf_gate.py --artifact <active_path>` silently overwrites the active
  artifact's acceptance-record metadata on any later diagnostic run.
  Documented as a "Known limitation" with a concrete follow-up
  recommendation (persist acceptance-time metadata separately, or guard
  `--artifact` against pointing at known active paths).
- (1)+(4) Replaced mean-only comparison with a paired per-cut check using
  the fixed `CUTS` calendar constant (3 exact windows, already stamped
  per-cut in `wf_gate_metadata["cuts"]`): candidate must be no-worse-than-
  incumbent within tolerance in at least 2 of 3 cuts. This directly catches
  the one-cut-blowup-hidden-by-mean failure mode codex named.
- (3) Grounded the tolerances in the one real empirical anchor available
  (`wf_3cut_sharpe_std=0.205`, read directly off the live artifact) while
  being explicit this is n=1 and provisional — `apy_tolerance` has no
  empirical anchor at all (no `wf_3cut_apy_std` is even computed yet).
  Added a concrete validation trigger: compute `wf_3cut_apy_std`, accumulate
  5+ admissible gate runs, then set tolerances from that empirical sample
  rather than an asserted constant. Metadata now includes a
  `tolerance_basis: "provisional-n1"` field for forensic honesty.
NEXT: implement `_check_incumbent_regression()` per the revised design,
add `wf_3cut_apy_std` computation, add the unit/integration/gate-parity
tests listed in the revised Test Plan.

## Round 3 (codex review)

STATUS: fixed (RFC revised)
WHAT: codex confirmed round 2's paired per-cut framing and admissible-
incumbent policy address most of round-1's gaps, but elevated the round-1
"known limitation" (mutable `wf_gate_metadata` on the active artifact) to an
actual blocker: comparing against a record that a later diagnostic run can
silently overwrite is a governance bug, not an observability gap, and
cannot stay a deferred follow-up if the gate is to be implemented.
WHY-DIR: the hard gate's entire trustworthiness depends on its comparator
source being stable — a mutable source means the gate could compare against
evidence that was never the incumbent's actual promotion-time evidence.
EVIDENCE: added "Phase 0" to the RFC — a concrete append-only acceptance-log
design (`_acceptance_log/promotions.jsonl`, written only by `promote()`
after `_check_wf_gate()` validates the staging artifact, read by
`_load_incumbent_wf_meta()` instead of the mutable active file). Verified
the root cause directly in code: `runner.py::main()` (line ~3512) writes
`wf_gate_metadata` to whatever `--artifact` path is given with no
distinction between staging and active paths, while `model_acceptance.py
::promote()` (line 798) is the one function called exactly once per genuine
promotion — the correct, existing hook point. Also found and documented (via
`assert_artifact_gated()`'s own docstring, referencing a real 2026-06-05
PatchTST incident) a real scope boundary: a `strategy_config.json` direct-
edit promotion bypasses `promote()` entirely and would not seed a log entry
either — noted as an explicit known gap (falls back safely to
not-admissible, not silently trusted), not silently ignored. Added explicit
sequencing language: the hard incumbent-regression gate must not activate
until Phase 0 has landed AND at least one genuine `promote()` has occurred
after it lands. Updated the Test Plan with Phase-0-specific tests, including
one that directly proves a diagnostic `runner.py main()` run against
`--artifact <active_path>` leaves `promotions.jsonl` untouched.
NEXT: implement Phase 0 (`promote()` write path,
`_load_incumbent_wf_meta()` read path) before implementing
`_check_incumbent_regression()` itself — Phase 0 is now a hard prerequisite,
not a parallel or follow-up task.
