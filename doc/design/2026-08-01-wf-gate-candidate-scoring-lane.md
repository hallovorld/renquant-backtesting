# WF gate candidate-scoring lane — design (docs only; no behavior change in this PR)

## The measured problem `[早前实测, two independent censuses 2026-08-01]`

Every stamped artifact in the 104 deployment carries
`candidate_artifact_used = false`: 13/13 distinct content digests (33 files, gate runs
2026-06-22 → 2026-08-01) in one census; 15/15 (55 files) in a wider one. The deployed
configs run `walkforward.enabled = true`, so `inspect_artifact_usage` takes the
`walkforward_manifest` scope — recipe-fingerprint validation against pre-existing
manifest artifacts — and the runner's own honest stamp says so. The machinery for
candidate-level evaluation EXISTS (`static_artifact` scope) but no deployed path uses
it.

Downstream costs, each independently measured:
- two artifacts of the SAME recipe are indistinguishable to the gate (GOAL-4's
  member-selection blocker; orch#744 audit UPHELD);
- a gate PASS attaches recipe-level evidence to a specific served booster (the
  incumbent's 06-21 stamp: `gate_verdict_before_override=false`,
  `operator_authorized_override=true`, computed with the candidate never scored);
- weekly challengers fail `benchmark_ok`/`regime_ok` while the incumbent sits behind
  an override they are not offered — freshness failure is downstream of gate design
  (orch#745's deferral condition "a validated remediation path exists" is
  unsatisfiable while no candidate can pass on its own merits);
- the gate's own placebo-adjusted read of the served recipe is `genuine_ic = +0.0008`
  against a `+0.020` bar — recipe-level evidence is not carrying its weight.

## Design: a third eval scope, `candidate_scored`

When `walkforward.enabled`, AFTER the existing recipe validation:

1. Load the candidate artifact's booster and its self-contained feature contract
   (`feature_cols` / `feature_means` / `feature_stds` / `feature_preprocess_version` —
   all present in stamped artifacts today).
2. For each manifest OOS window, score the window's panel rows with THE CANDIDATE's
   booster. Scoring is not training: one booster over ~600 OOS days × 292 names is
   seconds-to-minutes, far inside the 600 s budget that killed per-ticker retraining.
3. Compute the SAME gate statistics the recipe path already computes — aligned real
   IC, shift-placebo family, `genuine_ic`, benchmark/regime splits — on the
   CANDIDATE's own scores. No new thresholds; the existing bars apply unchanged.
4. Stamp `candidate_artifact_used = true`, `eval_scope = "candidate_scored"`, the
   candidate's content sha256, and the per-window statistics alongside the existing
   recipe block. The stamp binds evidence to the booster it describes.

Known limits, stated: the shift-placebo family and its bars are inherited unchanged —
this design fixes WHOSE scores are evaluated, not the placebo geometry (the 07-30
L = h erratum's scope is the corrected-eval bundle's block inference, not the gate's
difference bar; any bar recalibration is its own reviewed change).

## Rollout: dual-read, no OR-accept (the M6 §2c pattern)

- **Stage 1 — shadow:** stamp BOTH verdicts (recipe-level and candidate-scored);
  admission still on the old rule; each weekly run emits a divergence report
  (admitted-by-old vs would-admit-by-new).
- **Stage 2 — conjunction:** admission requires BOTH; the incumbent's standing
  override gains an EXPIRY in the same change, so incumbent and challengers face the
  same criteria (the orch#744 asymmetry closes here).
- **Stage 3 — cutover:** candidate-scored becomes THE admission; the recipe check
  remains as an identity precondition (an artifact from a foreign recipe never gets
  scored at all). Never OR-acceptable, per the M6 lesson.

Each stage transition is operator-authorized; nothing in this design self-promotes.

## Acceptance criteria

1. A real weekly candidate stamped `candidate_artifact_used = true` with its own
   `genuine_ic` — the first artifact-bound gate evidence in the deployment's history.
2. Divergence reports exist for ≥ 2 consecutive weekly runs before any Stage-2 ask.
3. orch#745's deferral condition becomes SATISFIABLE (a fresh candidate CAN pass on
   its own merits), unblocking the 28d-vs-60d governance decision.
4. No admission-rule change lands without the per-stage operator sign-off.

## Non-goals

Threshold/bar changes; promotion automation; the PatchTST lane (excluded until
orch#741's retire-or-fix); any production mutation in this PR (docs only).
