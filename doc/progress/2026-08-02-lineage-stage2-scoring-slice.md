# Stage-2 scoring lane: seam-separated pooling over the 125-window lineage (#94 slice 4, UNWIRED)   (PR #100)

STATUS:    in-progress — DRAFT held for the operator's stage-2 sign-off on #94; UNWIRED, must not merge until sign-off lands
WHAT:      adds `src/renquant_backtesting/wf_gate/lineage_stage2.py` (`attempt_lineage_scoring_stamp`) — a never-raising entry point that scores the 125-window extension lineage (43 production + 82 run-001) via the #96 engine, pooled per input-vintage segment only (`pre_seam` / `post_seam`, no cross-seam statistic). `runner.py` is untouched — a source-guard test pins it free of any `lineage_stage2` reference.
WHY/DIR:   prepares the module that the #94 stage-2 sign-off request's three points (read-alongside digest-verified / seam-separated pooling / admission byte-identical) govern; slice 4 of #94, after #95 (admissibility), #96 (engine), #99 (Stage-1 stamp). Advances the WF-gate lineage-extension thread under `doc/memory/mid-term/model-edge.md`.
EVIDENCE:
  artifact:      `tests/test_lineage_stage2.py` + full repo suite
  prod or exp:   experiment — module is UNWIRED, no `runner.py` path reaches it, nothing touches production
  existing data: targeted file `[VERIFIED — pytest -q tests/test_lineage_stage2.py, 2026-08-02]`: 18 passed, 0 skipped, 1.49s. Full suite `[VERIFIED — pytest -q, 2026-08-02]`: 594 passed, 1 skipped, 2 failed — both failures are the pre-existing `test_byte_equivalent_to_umbrella` pair (`tests/forensics/test_b1_lift.py`, `tests/reconciliation/test_import_lift.py`), confirmed unrelated: this commit (`60b8cdc..37718c8`) touches only the new module, its tests, and this progress doc.
  best-known?:   n/a — new module, no prior variant to compare against
  scope:         this is `tests/test_lineage_stage2.py`, experiment/UNWIRED, vs no existing baseline (first slice of its kind)
NEXT:      operator posts "approved stage 2" (or amendments) on #94; only then does the wiring PR (the `runner.py` touch) land as its own reviewed change

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

## Review round 2 (2026-08-02): budget enforced at every boundary

Codex finding: the 300 s whole-pass budget was polled only BEFORE each window,
so a slow `LS.score_lineage` call could cross the budget and — when it was the
last eligible call, or the next segment was refused — the lane returned a
NORMAL stage-2 stamp with `elapsed_seconds` over budget.

Fix (`lineage_stage2.py`): one `_budget_guard` helper enforcing the deadline at
EVERY boundary — before each scoring call (the existing poll), immediately
AFTER each scoring call, and once more at the successful-return boundary (label
summaries / frame concat run after the last call). Every breach is the same
stamped-unavailable shape; the reason now names the detection point
(`before scoring window …` / `after scoring window … — the scoring call itself
crossed the budget` / `at the successful-return boundary`). A hard wall-clock
containment boundary (thread/subprocess timeout) was considered and DEFERRED:
the scorer calls are the only long operations, the module has no such machinery
today, and the reviewed fix (deadline-after-each-call) covers every escape path
the finding names.

Regressions (`tests/test_lineage_stage2.py`, both fail on the pre-fix module —
verified by stashing the src change: normal `stage2` stamp escapes; both pass
with the fix):

* `test_slow_FINAL_scoring_call_is_stamped_unavailable` — fake-clock
  monkeypatch; ONLY the final eligible window's scorer (final ladder window,
  eligible via an explicit caller grid, so no subsequent pre-check exists)
  advances the clock past budget → must be the stamped unavailable with
  post-call detection in the reason, never a normal stamp.
* `test_budget_crossed_after_the_last_call_caught_at_return_boundary` — the
  budget crosses inside the post-seam label summary (after the LAST scoring
  call) → caught by the successful-return boundary check.

Counts `[本次实测 2026-08-02]`: targeted file **20 passed** (was 18, +2
regressions); full suite **596 passed, 1 skipped, 2 failed** — the same
pre-existing `test_byte_equivalent_to_umbrella` pair, re-verified failing on
the base commit (`839c0b4`) with this fix stashed (594 passed there).
