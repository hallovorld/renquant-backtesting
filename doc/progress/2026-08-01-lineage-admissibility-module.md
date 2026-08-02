# Lineage admissibility: the causal contract, checked against a caller-owned grid

2026-08-01 · `feat/lineage-admissibility-module` · PR #95

## What this adds

`renquant_backtesting.wf_gate.lineage_admissibility` decides whether a walk-forward
lineage may be used at all. Per window it recomputes
`effective_train_cutoff + embargo BDays < first OOS date` and records the margin in
business days; per lineage it recomputes the root over the fold digests and refuses the
whole thing on a root mismatch, a per-entry digest lie, a missing artifact, missing
self-carried provenance, or too few surviving windows.

Refusals carry the arithmetic in the reason string. A gate that says "refused" without
saying which date beat which is a gate nobody can act on.

## The two review rounds, because both were the same defect

**Round 1 — self-attestation.** The integration test derived `first_oos_dates` from each
artifact's own `oos_window`. The artifact was therefore judged against bounds it had
itself declared, which makes "43/43 admissible" a restatement of the artifact's opinion.
Fixed by taking the grid from the committed score corpus (per cutoff, `min(date)`), and
by a regression proving the caller's grid **governs**: an artifact whose self-declared
window would pass cannot rescue itself from a grid date that violates the contract.

**Round 2 — the same defect, one repo downstream.** That corpus was read from
`/Users/renhao/git/github/renquant-model-wt-clfrebuild/…`, with `pytest.skip` when
absent. The path is a transient **worktree**: the test ran on one machine, in a directory
that disappears when the worktree is removed, and skipped silently everywhere else —
while its result was being quoted as something this suite establishes. A skipped test
underwriting a published number is the recurring
`tests-that-measure-the-operators-disk` shape.

Cross-repo integration genuinely cannot be repo-contained here — `evaluate_lineage`
hashes 43 fold artifacts that live in `renquant-model`. So the responsibility splits, and
each half runs where its inputs actually are:

| half | owner | what it proves |
|---|---|---|
| digests + `lineage_root_sha` | `renquant-model#181` in-repo verifier | the 43 artifacts are the ones the manifest names |
| the admissibility contract | this repo, `tests/data/clf_lineage_window_grid.json` | all 43 windows clear the embargo against a corpus-derived grid |

The committed grid is 43 records — cutoff, effective train cutoff, embargo days, the
corpus-derived first OOS date, and (recorded but never read by the check) the artifact's
own declared window. It carries `lineage_root_sha`, the corpus `sha256`, and the
derivation rule, so a later reader can tell a changed bundle from a changed fixture.

## Evidence

| claim | value | provenance |
|---|---|---|
| all 43 windows admissible on the corpus grid | 43/43, min embargo margin ≥ 1 BDay | [VERIFIED — `pytest -q tests/test_lineage_admissibility.py`] |
| suite | 552 passed, 9 skipped, 12.67s | [VERIFIED — `pytest -q`] |
| committed grid matches the live bundle | 43/43 dates equal; `lineage_root_sha` equal | [VERIFIED — drift-guard body run against the bundle at `e9eefe81…`] |
| the drift guard detects a perturbation | perturbing one date is caught | [VERIFIED — same run] |

The one remaining `skip` is the drift guard, and only that one is allowed to: it detects
staleness and adds nothing to the result above, so skipping it costs a warning, not the
claim. That is the distinction round 2 was about.

## Not done here

No gate consumes this module yet. It is a decision procedure with tests, not a wired
admission check — `never-deploy-inert-scaffolding` applies, so the wiring should land
with the caller that needs it rather than ahead of it.
