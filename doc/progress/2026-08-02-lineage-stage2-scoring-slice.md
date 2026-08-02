# Stage-2 scoring lane: seam-separated pooling over the 125-window lineage (#94 slice 4, UNWIRED)

2026-08-02 · `feat/lineage-stage2-scoring` · DRAFT PR — **not to be merged until the
operator posts stage-2 sign-off on #94** (the sign-off request is the latest #94
comment; nothing here self-promotes).

## What this slice is

`src/renquant_backtesting/wf_gate/lineage_stage2.py` — one entry point,
`attempt_lineage_scoring_stamp(...)`, mirroring Stage-1's (#99) contract exactly:

* NEVER raises — every failure is `{"lineage_stage2": "unavailable", "reason": …}`;
* admission untouched, recipe stamps byte-unchanged; the block is a SIBLING key a
  future caller attaches. `runner.py` is NOT touched: a source-guard test pins it
  free of any `lineage_stage2` reference, so the wiring must land as its own
  reviewed change after the operator's "approved stage 2" — severability is
  mechanical, not a promise;
* scores the extension lineage (43 production + 82 run-001 windows) per the #96
  engine, POOLED PER INPUT-VINTAGE SEGMENT ONLY. The stamp emits NO cross-seam
  pooled statistic; a combined number must be computed downstream, in the open.

## Shape decision: in-run, budget-guarded (the design text governs)

The merged #94 design specifies IN-RUN scoring ("for each manifest OOS window,
score the window's panel rows … seconds-to-minutes, far inside the 600 s
budget") and has no offline-evidence provision; the run-001 bundle carries
ARTIFACTS (boosters), not precomputed scores, so an offline-evidence-by-digest
shape was not available even if preferred. Divergence noted rather than
improvised: the design asserts budget feasibility but mandates no guard; this
slice ADDS a bounded time budget (default 300 s `[推导 — half the 600 s budget
the design cites; overridable per call]`) whose breach is a stamped
`unavailable`, never an unbounded gate slowdown.

## Identity: content-bound to the exact evidence scored

The stamp refuses unless ALL hold:

* extension manifest bytes hash to the caller-pinned `expected_manifest_sha256`
  (the RUN_CLAIM binding, `b70119eb…` for run-001);
* OLD root over the 43 existing declared shas == claimed `d1161f8d…`, and the
  FULL-ladder root over all 125 == claimed `83496eac…` (the #94 root rule,
  recomputed);
* the Stage-1 block's admitted root == the extension's OLD root and the recipe
  ids match — the bundle must extend exactly the lineage Stage-1 admitted;
* per window, on-disk artifact bytes re-digest to the declared sha (via #95's
  `evaluate_lineage`); a tampered artifact refuses its WHOLE segment.

All four verified against the real bundle by the integration test
`[本次实测 2026-08-02 — 125 digests recomputed from committed/umbrella bytes]`.

## The seam, mechanical

Segments come from the manifest's own first-class seam: `new_windows` (pre-seam,
cutoffs 2019-01-14 → 2023-09-11, every row stamped
`input_vintage: 2026-08-01-rebuild`) and `existing_windows` (post-seam, cutoffs
2023-10-02 → 2026-03-02, the June-vintage production ladder). Structural
refusals: a new row missing the seam vintage, an existing row carrying it, a
non-chronological or overlapping ladder, count mismatches, an unknown schema.
The stamp's `vintage_seam` block names the boundary (`2023-09-11 → 2023-10-02`)
and carries the manifest's golden-evidence digest.

## Two findings made while building `[本次实测 2026-08-02]`

1. **The #96 public contract refuses the gbdt window artifacts.** Production
   and run-001 window artifacts self-carry `feature_means`/`feature_stds` as
   ORDERED LISTS (172 entries aligned to `feature_cols`; writer:
   `renquant_model_gbdt.panel_trainer`, alignment verified in source), while
   `load_fold_scorer` demands dicts (the clf fold shape it was golden-verified
   on). This slice adds `gbdt_window_scorer_factory`: a fail-closed RE-KEYING
   adapter — zero transform math, refuses any shape it cannot positively
   recognize. The alternative (widening the model-repo contract) belongs to
   renquant-model and is flagged for review, not smuggled in here.
2. **The final ladder window has no closing edge.** The `(cut, next_cut]` grid
   rule leaves 2026-03-02 without a closing cutoff; the design does not pin it.
   The window is REFUSED with a stamped reason (never invented), so the
   post-seam segment scores 42/43 in-run; a caller may score it only by
   supplying an explicit `oos_dates_by_cutoff` grid.

## Evidence

* `tests/test_lineage_stage2.py`: 18 tests — never-raises on every failure
  path; seam pooling proven by construction (pre-seam IC +1.0 vs post-seam
  −1.0 with a combined pool ABSENT); content binding (manifest byte-flip,
  declared-sha tamper, artifact-byte tamper each refuse); stage-1 cross-lane
  binding; final-window refusal + caller-grid override; time-budget refusal;
  input non-mutation; source guards (runner free of stage-2; module free of
  `artifact_usage` and runner imports); REAL run-001 integration (82/43
  windows, both roots, seam boundary, 125 digests); real-artifact default
  factory; adapter misalignment refusal.
* Full suite `[本次实测 2026-08-02]`: **594 passed, 1 skipped, 2 failed** — both
  failures are the pre-existing `test_byte_equivalent_to_umbrella` pair
  (repo-vs-live-umbrella byte drift in `model_acceptance.py`, present on
  origin/main, files this slice never touches).
