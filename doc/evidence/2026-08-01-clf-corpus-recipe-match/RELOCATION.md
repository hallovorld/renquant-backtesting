# Relocated evidence — clf corpus/recipe match (UNREPRODUCED HISTORICAL MEASUREMENT)

STATUS: **an unreproduced historical measurement with partial surviving
provenance** — a record of what was measured on 2026-08-01 against the
then-live corpus and artifacts, NOT a statement about the current corpus
state. Non-performance, capability-gap record only.

RELOCATED 2026-08-04 from `renquant-orchestrator` branch
`goal6/wf-corpus-capability-census` (PR orch#718, CLOSED out-of-scope).
The closure ruling names the destination:

> `renquant-orchestrator` owns pinned-subrepo orchestration, and this PR
> hosts model-artifact evaluation. […] What must move, and where:
> **clf 语料/recipe 比对器 → `renquant-model + renquant-backtesting`**,
> with the source/provenance and the stated non-performance claims
> carried across. If orchestration ever needs the result, it consumes a
> versioned summary artifact, not the evaluator.

Placement note: the ruling names both repos; the archive lands HERE
because the measured object is the WALK-FORWARD CORPUS (this repo's
surface — the folds, and `run_wf_gate._recipe_projection` as the match
criterion). The clf recipe itself is `renquant-model`'s object; if a
model-side pointer is wanted, it should reference this directory rather
than duplicate the corpus-side evidence.

## Provenance manifest

Source commit OID: `21ece5db7323bbf2da55512ca53ec7204a083e2a`.

sha256 of every relocated file:

```
baa7ae06eb2af07c9df3e9d643e7e476d9c1f94e5b72cbef5d1a916301eae657  CLAIMS-original-progress-doc.md
0a8f336d17b60000b1f58c81c17dcb2baed91a691f1516820d8e84d61d4d0d19  clf_match.json
d46e63bf482b552ab9bbb565ba7f539242dec06514ce8d635b80453a07bd64d7  run.log
9e18a0eac75eaf64927999dbff21be07942fd6182a645fa5950829547c2fc5c9  test_wf_corpus_recipe_match.py
129c99af42becf273181bd927103aa41b1d7be44ad8c5468f2a184d7b431e371  wf_corpus_recipe_match.py
```

## Why UNREPRODUCED (the honest inventory)

- The corpus and artifacts are identified by recipe-fingerprint prefixes
  (`a4141c07…`, `7d684522…`) and machine paths, not immutable content
  identities; the corpus is a LIVE, growing surface, so 85-folds-at-
  measurement is a dated snapshot, not a current count.
- The evaluator is a MACHINE-LOCAL runner (operator machine's corpus +
  artifact tree); preserved as provenance, deliberately not in this
  repo's CI, and not sufficient for replay elsewhere.

## What the historical record claims (summarized; the byte-preserved
`CLAIMS-original-progress-doc.md` is the authored source)

Using the gate's own fold-match criterion (`_recipe_projection`) instead
of directory naming: of the 85 walk-forward corpus folds available on
2026-08-01, **0 matched the certified clf recipe** (`a4141c07…`) — the
GOAL-6 anchor "the certified clf recipe has no out-of-sample corpus" was
CORRECT, and the earlier folder-name-based "correction" (withdrawn
2026-07-31) stays withdrawn. Within this archive's label, that is a
property of the 2026-08-01 corpus snapshot; any current claim requires
re-measurement against today's corpus.
