# 2026-08-05 — GOAL-6: Stage 2 scores and is never evaluated; and it broke today

## Two measurements, both from the live artifact corpus `[VERIFIED — this session]`

### 1. The lane SCORES. It has never been EVALUATED.

The 2026-08-04 staging stamp:

```
lineage_stage2      = stage2
n_scored_windows    = 124   (of 125)
recipe_id           = sha256:cfdd6cb8e950da0f
elapsed_seconds     = 4.462
```

It scored 124 windows — and carried **no `label_summary` at all**, because the
gate's call site (`runner.py:3438`) calls `attempt_lineage_scoring_stamp(...)`
with `stage1`, `extension_manifest_path`, `expected_manifest_sha256`, `panel`
and `label_horizon_bdays` — and **no `labels_by_date`**.

So the lane produces scores that are never compared with an outcome. **No IC of
any kind is computed.** And the per-regime split added in bt#107 sits downstream
of that summary, so it has never run either — my note there said "the runner
does not supply a regime map", which understated it: the runner supplies no
labels.

This is the "deployed-but-dark" shape: a scoring lane that runs, costs 4.5 s,
stamps `n_scored_windows=124`, and answers nothing.

### 2. It went UNAVAILABLE today

The 2026-08-05 stamp:

```
lineage_stage2 = unavailable
reason = stage-1 admitted root 2969e1d199e2… != extension's old root
         d1161f8d46b5… — this bundle does not extend the admitted lineage
```

Stage-1's admitted lineage root moved (`d1161f8d…` on 08-04 → `2969e1d1…` on
08-05) while the frozen extension bundle still declares the old root.

Stated precisely `[codex on bt#108]`: this is a **refusal BEFORE scoring**, not
"scoring nothing" — the stamp carries no `n_scored_windows` at all. And I have
**not** measured whether anything alarmed on it; the earlier "nothing alarmed"
was an assumption, withdrawn. What is measured is that between two consecutive
sessions the lane went from a scored stamp to an `unavailable` one.

## What lands here (small, on purpose)

Only the visibility fix: when a segment is scored but not evaluated,
`statistics["label_summary"]` is now an explicit `None` **with a stated reason**,
and the two reasons are distinguishable:

- *"the caller supplied no `labels_by_date` — this lane SCORED but was never
  evaluated against labels, so it produced no IC of any kind"*
- *"no rows were scored in this segment, so there is nothing to evaluate"*

Previously the key was simply absent, and a reader could not tell those apart —
nor tell either from a successful evaluation.

**This also changes the payload for callers that DO pass labels but score zero
rows** `[codex on bt#108]`: `main` omitted the key; now they get
`label_summary = None` plus the reason. That is deliberate and is pinned by a
test. The labels-plus-scores path is unchanged, and no verdict moves.

## What is NOT done, and why not here

- **Wiring real labels into the Stage-2 call site.** That is a substantive
  change to what the gate computes, it needs the label contract the sanity
  battery uses, and it should not be smuggled in behind a visibility fix.
- **The root mismatch.** Re-pinning the extension bundle to the current lineage
  root is a decision about which lineage Stage-2 extends, not a bug fix. Filed
  as the blocker it is.

Both are named in NEXT rather than guessed at.

## NEXT

1. Decide what Stage-2's labels are (the battery's `fwd_60d_excess` contract is
   the obvious candidate) and pass them — until then every `n_scored_windows`
   number in an artifact is a throughput count, not evidence.
2. Re-pin or regenerate the extension bundle against the current admitted root,
   and add an alarm for the mismatch: today the lane simply went quiet.

## The tests, after review

The first version of the test file **reimplemented the new branch and then
asserted the copy matched** — proving the helper equals itself — and its
live-bound test only checked that *some* scored stamp existed
`[codex on bt#108]`. Rewritten: the unit tests drive the real `_score_segment`
(stubbing only the scoring engine), and the live-bound tests assert the measured
facts — `n_scored_windows == 124` of `125`, both segments (`pre_seam`,
`post_seam`) carrying `n_rows_scored > 0` and **no** `label_summary`, and the
08-05 reason containing both root prefixes and no scored count.

Suites: 6 tests, four bound to the live stamps · **673 passed, 1
skipped, 3 failed** — all three failures (`test_b1_lift`, `test_import_lift`,
`test_lineage_stage2::test_REAL_run001_…`) reproduce on `origin/main` and are
untouched by this change `[VERIFIED — re-run on main earlier this session]`.
