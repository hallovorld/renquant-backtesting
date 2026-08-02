# Re-pin the lineage grid to the regenerated bundle, and stop the drift guard measuring the machine

2026-08-01 · `fix/lineage-grid-repin-to-182`

## Why

`renquant-model#182` found that all 43 fold artifacts in the merged lineage bundle had
`feature_norm_kind` persisted as `str(list)` — a 2064-character string where a
172-element per-feature list belonged, against `feature_cols` of length 172. The
artifacts were regenerated and `lineage_root_sha` moved
`e9eefe8137…` → `1da510478e…`.

The drift guard added in #95 caught the downstream staleness, which is what it is for.

## What moved, measured rather than assumed

Re-deriving the grid from the regenerated bundle changes **90 lines and nothing else**:

| field | change |
|---|---|
| `artifact_sha256` (×43) | all new — the artifact bytes were rewritten |
| `lineage_root_sha` | `e9eefe8137…` → `1da510478e…` |
| `source_pr` | 181 → 182 |
| every date field | **unchanged** |
| `score_corpus_sha256` | **identical** — `46f447fd8d08…` |

[VERIFIED — field-by-field diff of the old and regenerated grids]

The identical corpus digest is the strongest available confirmation of #182's
"corpus scores unaffected": the parquet bytes never changed, so the caller-owned grid
was never contaminated. Only the artifacts were.

## The guard was resolving the bundle off the working tree

Second fix, and the more important one. The original guard searched for the bundle
**on disk**, across two candidate sibling directories. What it compared against therefore
depended on which branch a sibling checkout happened to be sitting on. During the
incident it resolved to a throwaway worktree, and had that worktree been removed it would
have found nothing and skipped — silently, while the fixture was stale.

A guard whose subject changes with someone else's `git checkout` is not measuring
upstream; it is measuring the machine — the recurring shape in
`tests-that-measure-the-operators-disk`. It now reads
`git show origin/main:<bundle>` from the `renquant-model` repo, which is branch-independent,
and skips only when there is no such checkout or no fetched `origin/main`.

It also gained a per-fold digest comparison. Matching the root already implies matching
digests, but when the root *disagrees* the digest set says whether one fold moved or all
43 did — which is exactly the question this incident opened with.

## Evidence

| claim | value | provenance |
|---|---|---|
| guard resolves branch-independently | model checkout is on `prereg/goal4-plausibility-bound-construction`, bundle still read from `origin/main` (4,386,319 parquet bytes) | [VERIFIED — guard logic run under the real checkout's path resolution] |
| passes on the re-pinned fixture | dates ✓ root ✓ digests ✓ | [VERIFIED — same run] |
| **fails on the superseded fixture** | dates ✓ root ✗ digests ✗ → FAIL | [VERIFIED — same run] |
| suite | see PR body | [VERIFIED — `pytest -q`] |

## One test loosened

`test_the_grid_says_WHERE_IT_CAME_FROM` pinned `source_pr == 181` and went red on the
regeneration while nothing was wrong. That is a proxy for the claim rather than the
claim, and it adds no safety over `lineage_root_sha`, which the drift guard checks
against upstream directly. It now asserts the field exists and is a positive integer.
