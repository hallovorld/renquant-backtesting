# Static sanity scope was scoring the wrong panel (#84)

**Bottom line.** Static scope is the ONLY gate path that scores a candidate's own
booster, and it was reading the training contract at a location the current
producer does not write, so it silently fell back to the 292-ticker rawlabel
corpus. Three bounded fixes; no threshold moves; no decision rule changes.

## Why the lookup missed

`renquant-orchestrator#620` stamps `training_contract` under `metadata`, and not
by preference: a ROOT-level `training_contract` key is UNCLASSIFIED in
`renquant_common.model_fingerprint`, so `model_content_sha256` raises
(`renquant-common#38`). The gate read only the root. `training_contract_dataset()`
now reads both, root winning, with the bare `dataset` key as a last resort.

**Evidence the fallback was actually taken** `[VERIFIED-now]`: a static run
reported `sanity_eval_end 2026-04-28` — the fallback corpus's max date — while
the artifact's own training panel ends `2026-05-01`. Those two dates cannot both
describe the panel that was scored.

## Why this mattered more than it looks

Manifest scope sets `candidate_artifact_used = false` and admits on recipe hash
alone; measured, a 296-ticker candidate, a 292-ticker control, a repro of the live
weekly and the incumbent **all hash to `sha256:cfdd6cb8e950da0f`** (issue #83).
Static scope is the discriminating path: 16 of 68 metadata keys differ between
those two artifacts, including every IC quantity (real_ic +0.0458 vs +0.0746),
against 4 of 68 in manifest scope where all four are echoed paths. So the one
path that can see a data or universe change was scoring a corpus that has
neither.

## The other two

**INVALID receipt now refuses.** The fallback corpus can carry an
`.INVALID.json` receipt written by another component; nothing consulted it, so a
run would happily evaluate on a corpus this programme has disowned. Both receipt
spellings are checked and the refusal is explicit about the remedy.

**Static runs now record which panel they scored.** The manifest branch does
`sanity_meta.update(panel_meta)`; the static branch did not. A static result
therefore carried no evidence of its own evaluation corpus — which is exactly
why the defect above survived. `panel_meta` is spread FIRST so the branch's own
explicit keys still win.

## Tests

9 new, in `tests/test_wf_gate_static_scope_panel_defects.py`. Three are
anti-vacuity: `test_the_old_root_only_read_would_have_failed_this` asserts the
pre-fix expression returns `None` on the shape #620 emits, and
`test_no_receipt_means_the_fallback_still_works` proves the refusal is caused by
the receipt and not by the fixture. The defect-3 test is structural and says so
in its own docstring — the merge sits inside a function a unit test cannot reach
without a full sim, so it asserts the exact shape instead of the behaviour. My
first version of that test used `str.index()` and failed on an unrelated
occurrence of the same literal, which is the same read-the-wrong-object mistake
this issue is about; it now checks every occurrence.

## Suite

| | result |
|---|---|
| `origin/main` (2fcec87), separate worktree | 3 failed, 431 passed, 8 skipped |
| this branch | 3 failed, **440** passed, 8 skipped |

Same 3 pre-existing failures, +9. `[VERIFIED-now]`

## Not in scope

Two further items in #84 need their own A/B because they change which fold
selection picks, and are deliberately untouched here: the eval-window derivation
and the config-reference choice. Separately, static scope still cannot evaluate a
production full-panel artifact at all — `safe_last_label 2026-07-24` exceeds
every date on the panel, so the latest admissible cutoff discards 603 business
days. That is issue #84's remaining structural half, not a bug this PR can fix.
