# Static sanity scope was scoring the wrong panel   (PR #86)

STATUS:    delivered
WHAT:      Static WF-gate sanity resolved `training_contract` only at the artifact
           root, so it silently fell back to the 292-ticker rawlabel corpus instead
           of the candidate's own training panel. Reads both locations now; refuses
           a fallback corpus carrying an INVALID receipt; records which panel a
           static run actually scored.
WHY/DIR:   Static scope is the ONLY gate path that scores a candidate's own booster
           (issue #83: manifest scope sets `candidate_artifact_used=false` and
           admits on a recipe hash four different artifacts share). Every question
           about universe or data width — the watchlist expansion, GOAL-6 Stage 2
           breadth retrain — has to be answered through this path. It could not
           answer them while it was scoring a corpus with neither change.
EVIDENCE:  §1 below (mechanism proof + discriminating-power measurement + suite A/B).
NEXT:      Unblocks evaluating a universe-extended candidate against its own panel.
           Does NOT unblock production full-panel artifacts — see §5, that is #84's
           remaining structural half and needs a separate design.

CORRECTIONS: this file's first committed revision tagged its measurements
`[VERIFIED-now]`, which is not an allowed form under LONG rule #10. No figure
changed; every tag below is restated in the required form with its source named.

## §1 EVIDENCE

### Mechanism — the lookup missed, and the fallback was actually taken

`renquant-orchestrator#620` stamps `training_contract` under `metadata`, and not by
preference: a ROOT-level `training_contract` key is UNCLASSIFIED in
`renquant_common.model_fingerprint`, so `model_content_sha256` raises
(`renquant-common#38`). The gate read only the root.

A static run reported `sanity_eval_end` **2026-04-28** — the fallback corpus's max
date — while the artifact's own training panel ends **2026-05-01**
`[VERIFIED — static-eval run, recorded in renquant-backtesting#84]`. Those two
dates cannot both describe the panel that was scored, which is what makes this a
measured silent fallback rather than an inferred one.

### Why this path and not the other one

| scope | metadata keys differing between a 296-ticker candidate and a 292-ticker control |
|---|---|
| `walkforward_manifest` | 4 of 68, all echoed paths |
| `static_artifact` | **16 of 68**, including every IC quantity |

`[VERIFIED — 3-cut gate run on candidate vs control, recorded in renquant-backtesting#83 and #84]`

In manifest scope the candidate, the 292-ticker control, a reproduction of the live
weekly and the live incumbent **all hash to `sha256:cfdd6cb8e950da0f`**, and the
two arms produced bit-identical decision fields (metadata diff = NONE)
`[VERIFIED — same run, recorded in renquant-backtesting#83]`. Static scope
separates the same two artifacts on `real_ic` **+0.0458 vs +0.0746**
`[VERIFIED — static-eval run, recorded in renquant-backtesting#84]`.

### Suite A/B

| tree | result |
|---|---|
| `origin/main` @ 2fcec87, separate worktree | 3 failed, 431 passed, 8 skipped |
| this branch | 3 failed, **440** passed, 8 skipped |

`[VERIFIED — python3 -m pytest -q, run in both worktrees this session]`. Same three
pre-existing failures; the delta is exactly the 9 tests added here.

## §2 The other two defects, same root

**INVALID receipt now refuses.** The fallback corpus can carry an `.INVALID.json`
receipt written by another component; nothing consulted it, so a run would happily
evaluate on a corpus this programme has disowned. Both receipt spellings are
checked and the refusal names the remedy.

**Static runs now record which panel they scored.** The manifest branch does
`sanity_meta.update(panel_meta)`; the static branch did not, so a static result
carried no evidence of its own evaluation corpus — which is precisely why the
defect in §1 survived. `panel_meta` is spread FIRST so the branch's own explicit
keys still win.

## §3 Tests

9 new, in `tests/test_wf_gate_static_scope_panel_defects.py`. Three exist to
prevent vacuity: `test_the_old_root_only_read_would_have_failed_this` asserts the
pre-fix expression returns `None` on the shape #620 emits;
`test_no_receipt_means_the_fallback_still_works` proves the refusal is caused by the
receipt and not by the fixture; the defect-3 test states in its own docstring that
it is structural and why (the merge sits inside a function no unit test reaches
without a full sim).

My first version of that structural test used `str.index()` and failed on an
unrelated occurrence of the same literal — the same read-the-wrong-object mistake
this issue is about. It now checks every occurrence.

## §4 Scope discipline

`training_contract_dataset()` is extracted as a pure function for one reason: the
regression is otherwise untestable. No threshold moves and no decision rule
changes.

## §5 Deliberately NOT in scope

Two further items in #84 change which fold WF selection picks and need their own
A/B: the eval-window derivation and the config-reference choice. Untouched here.

Separately, static scope still cannot evaluate a production full-panel artifact at
all: `safe_last_label` **2026-07-24** exceeds every date on the panel, so the
latest admissible cutoff is 2024-01-10, discarding **603 business days ≈ 2.31
years** and widening `[DERIVED — np.busday_count(2024-01-10, 2026-05-01) on the
artifact's own stamped safe_last_label and the training panel's max date]`. That is
#84's remaining structural half, not a bug this PR can fix.
