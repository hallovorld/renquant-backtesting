# Stage-1 lineage lane: a sibling that cannot lie about its ladder

2026-08-01 · `feat/lineage-lane-slice3` · PR #99 · review round 1

Two integrity blockers, both real. The fix is in `e5150f5`; this note records what the
defects were and what the regressions do and do not prove.

## 1. The lane was mutating the block it claims not to touch

`runner.py` copied `artifact_usage`, nested `lineage_stage1` inside it, and then
serialised **that** object as the run's `artifact_usage` stamp. The recipe path's bytes
changed because a *different* lane had been added — the one thing the Stage-1 contract
promises will not happen. "Dual-read only" has to mean the first read is untouched.

The lane is now a top-level sibling in the output and `artifact_usage` is passed through
unmodified.

## 2. The ladder was taken on trust

`build_gbdt_lineage_view` walked `retrains` in manifest order and never checked that the
rungs were unique, ordered, or that each artifact was the one its rung names. Three ways
an admissible root could be minted from a lineage that does not exist:

| corruption | what happened before | now |
|---|---|---|
| reordered rungs | the grid is derived from the ladder by the `(cut, next_cut]` rule, so a shuffle yields a grid describing different retrains than it names — every digest still verifying | refused: *not chronologically ordered* |
| duplicated rung | both copies counted toward the admissible minimum while the grid, a dict keyed by cutoff, held one entry — one window counted twice | refused: *duplicates* |
| `artifact_uri` pointing at another window | the manifest supplied the cutoff, the artifact supplied the training bound, and **nothing compared them** — a mis-pointed URI was scored against another window's grid date and could come back admissible | refused: *wrong artifact behind this window* |

The third is the one with teeth, and it is the `renquant-model#182` lesson one level up:
**digests attest identity, never validity.** A wrong-but-consistent artifact hashes
perfectly.

## What the regressions prove, and what they do not

An end-to-end byte-diff of a real gate run needs the artifact corpus, which the suite
does not have. So the invariant is pinned at two levels, and it is worth being precise
about the seam between them:

* **behavioural** — `attempt_lineage_stamp` does not mutate the dict it is handed. Now
  covered on the AVAILABLE path (the only one with something to mutate: it reads the
  manifest, hashes nine artifacts, builds a view) and on all four unavailable branches,
  rather than on a single early return.
* **structural** — the runner writes into `artifact_usage` by no subscript at all,
  asserted over the AST rather than by pinning the one literal that was there.
  `artifact_usage["lineage_v2"] = …`, or an alias (`au = artifact_usage; au["x"] = …`),
  passes a string check and reintroduces the defect verbatim.

Neither is a byte-diff of a real run. Saying so is the point: the enforceable halves are
these two, and a reader should not read them as more.

## Evidence

| claim | value | provenance |
|---|---|---|
| suite | see PR body | [VERIFIED — `pytest -q`] |
| the AST guard catches what the literal misses | injecting `au = artifact_usage; au["lineage_v2"] = None` leaves the literal test GREEN and turns the AST test RED | [VERIFIED — direct injection, both tests re-run] |
| restoring the unvalidated ladder | the three structural refusals fail | [VERIFIED — `git show` of the pre-fix module, re-run] |
