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

### Causal admissibility contract (review round 1 — fail-closed, before any verdict)

Scoring a booster over a HISTORICAL window is only valid evidence if the booster
could not have seen that window: a candidate trained today, scored over last year's
OOS windows, leaks its training labels into the historical gate and can "pass" on
information no deployable model had. Therefore, per window, the lane requires
**causally valid model provenance** before a score contributes to any verdict:

* the scoring model for window `w` must be the recipe's **per-window snapshot** (or
  equivalent immutable lineage) whose
  `effective_train_cutoff + label_horizon < first OOS score date of w`, with the
  realized embargo margin RECORDED in the stamp per window;
* absent that evidence for a window, the lane REFUSES that window (and with fewer
  admissible windows than the gate's minimum, refuses the verdict entirely) —
  `admissibility: refused` is a stamped outcome, never a silent skip;
* the SINGLE final candidate booster may additionally be shadow-scored over all
  windows as a DESCRIPTIVE diagnostic (clearly labelled, no admission weight, and
  excluded from Stage-1 divergence reports) — useful for drift inspection only.

The existing walkforward manifest already trains per-window artifacts for the recipe;
the contract above makes the lane consume THOSE (each window scored by its own
cutoff-valid snapshot), which is what "scoring the candidate" must mean for a
manifest recipe: the candidate IS the per-window lineage, identity-bound by the
recipe fingerprint plus each snapshot's content sha.

Known limits, stated: the shift-placebo family and its bars are inherited unchanged —
this design fixes WHOSE scores are evaluated, not the placebo geometry (the 07-30
L = h erratum's scope is the corrected-eval bundle's block inference, not the gate's
difference bar; any bar recalibration is its own reviewed change).

## Feasibility, measured `[本次实测 2026-08-01]`

The per-window lineage the admissibility contract requires ALREADY EXISTS for the
prod gbdt recipe: `walkforward_manifest_gbdt_prod_recipe_v2.calibrated.json` carries
**43 retrains, 0 failed cutoffs** (cutoffs 2023-10-02 → 2026-03-02, lookahead 60d),
and **43/43 per-window artifacts plus 43/43 per-window calibrators exist on disk**
(relative `artifact_uri`/`calibrator_uri` under the strategy dir). Every window
artifact self-carries the exact contract fields: `feature_cols/means/stds`,
`cutoff_date`, `cutoff_embargo_days: 60`, `effective_train_cutoff_date`. Stage 1 for
the gbdt recipe therefore needs ZERO new training runs.

The genuinely uncovered case is the **clf recipe**: its 43-fold corpus persisted
SCORES but not fold artifacts (gate-visible recipe match 0/85), so clf admissibility
requires one corpus rebuild WITH artifact persistence — bounded cost (the corpus
recipe is committed and reproducible), scheduled below as the clf on-ramp.

## Rollout: dual-read, no OR-accept (the M6 §2c pattern)

- **Stage 1 — shadow:** stamp BOTH verdicts (recipe-level and candidate-scored);
  admission still on the old rule; each weekly run emits a divergence report
  (admitted-by-old vs would-admit-by-new). Only causally admissible windows feed the
  divergence report; the descriptive final-booster sweep is excluded from it.
- **Stage 2 — conjunction:** admission requires BOTH; the incumbent's standing
  override gains an EXPIRY in the same change, so incumbent and challengers face the
  same criteria (the orch#744 asymmetry closes here).
- **Stage 1b — clf on-ramp:** rebuild the clf WF corpus once WITH per-fold artifact
  persistence and register those snapshots in a gate-visible manifest; the clf recipe
  then enters the same Stage-1 shadow on equal terms. Until then the lane simply has
  no clf verdict to offer — a stamped absence, not a silent pass.
- **Stage 3 — cutover:** candidate-scored becomes THE admission; the recipe check
  remains as an identity precondition (an artifact from a foreign recipe never gets
  scored at all). Never OR-acceptable, per the M6 lesson.

Each stage transition is operator-authorized; nothing in this design self-promotes.

## Acceptance criteria

1. A real weekly candidate stamped `candidate_artifact_used = true` with its own
   `genuine_ic`, every contributing window carrying recorded causal provenance
   (`effective_train_cutoff + label_horizon < first OOS date`, embargo margin
   stamped) — the first artifact-bound gate evidence in the deployment's history.
1b. A deliberately cutoff-violating window is REFUSED with `admissibility: refused`
   in the stamp (the fail-closed path exercised, not asserted).
2. Divergence reports exist for ≥ 2 consecutive weekly runs before any Stage-2 ask.
3. orch#745's deferral condition becomes SATISFIABLE (a fresh candidate CAN pass on
   its own merits), unblocking the 28d-vs-60d governance decision.
4. No admission-rule change lands without the per-stage operator sign-off.

## Non-goals

Threshold/bar changes; promotion automation; the PatchTST lane (excluded until
orch#741's retire-or-fix); any production mutation in this PR (docs only).
