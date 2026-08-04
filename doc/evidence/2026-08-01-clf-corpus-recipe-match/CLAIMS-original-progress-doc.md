# GOAL-6 — the clf anchor is CORRECT, proven by the gate's own criterion instead of a folder name

**Date:** 2026-08-01 · `renquant-orchestrator` · GOAL-6 (clf WF 语料补齐)

## Bottom line

The GOAL-6 anchor says the certified clf recipe has **no out-of-sample corpus**. On
2026-07-31 I "corrected" that anchor using the GBDT corpus's 43 folds and had to withdraw
it — I had attached one lane's corpus to another **by directory name**. This settles it
with `run_wf_gate._recipe_projection`, the gate's own definition of whether a fold
validates a candidate `[本次实测 2026-08-01]`:

| | |
|---|--:|
| clf recipe fingerprint | `a4141c076b6b9591` |
| prod recipe fingerprint | `7d6845222c15626d` |
| walk-forward corpus folds available | **85** |
| **folds matching clf's recipe** | **0** |

**The anchor is correct.** All 83 usable folds carry `objective: rank:pairwise`; clf carries
`binary:logistic`.

## The near-miss, and why it justifies a tool

My first pass omitted `params` from the projection — it is the one nested-dict field of the
six — and returned **82 of 85 matching**, the opposite conclusion. clf and prod agree on:

- `kind` (`panel_ltr_xgboost` — the clf lane is **not** a distinct kind)
- all **172** `feature_cols`
- `feature_norm_kind`
- `label_col` (`fwd_60d_excess`)
- `lookahead_days` (60)

**The entire difference is one key: `objective`.** The tool therefore reports *which fields
broke each match*, and on the real corpus that reads:

```
  82 fold(s)   params
   2 fold(s)   kind,feature_norm_kind,params
   1 fold(s)   feature_cols,feature_norm_kind,label_col,params
```

82 of 85 differ on `params` **alone** — which is exactly how dropping that one field flips
the verdict.

## How this sits beside orch#713

orch#713 measured that the same projection is **invariant across 12 same-recipe boosters**
— blind on every axis that separates them. Here it is the *only* thing separating clf from
prod. Both are true, and reporting the differing fields is what keeps them from being
confused: "the projection is weak" and "the projection is what saved this" are the same
projection viewed on different axes.

## Not claimed

That clf *should* have a corpus, or that building one is cheap — the orchestrator's only
scaling builder, `build_wf_manifest.py`, re-runs `renquant_orchestrator.train_gbdt` per
cutoff and has no model-kind argument, so a clf corpus is not a re-run of it. That the 2
`hf_patchtst` folds are adequate for that lane. That `feature_source_contract_keys`
matches — it is derived inside the gate, not an artifact field, and is declared unchecked
rather than silently dropped.

## Tests

10, including the near-miss as an executable check (`params` alone must change the
fingerprint), a companion artifact without `feature_cols` not counted as a non-match, an
unreadable fold counted separately, and **no folds SKIPping with 3** — "no fold matched"
and "no fold was read" are different facts and only the first is evidence.

Suite: **5194 passed, 2 skipped**, run before the push.
