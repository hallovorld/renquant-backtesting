# WF Gate Incumbent Regression Check

**Date:** 2026-07-05
**PR:** #70
**Status:** Design RFC — round 2, addressing Codex review

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
