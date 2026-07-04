# S3: Placebo difference test gate — reverted to shadow-only (already on main)

Date: 2026-07-04
PR: #67 (renquant-backtesting, `feat/s3-placebo-difference-gate`)
Reviewer: Codex (haorensjtu-dev) — CHANGES_REQUESTED addressed.

## What changed (single durable record)

**STATUS**: This PR's original implementation re-introduced an already-rejected
anti-pattern. It is now a no-op on `main` (CI-pin fix only) pending an operator
decision on whether to close it.

**WHAT**: The original PR made the placebo **difference test**
(`genuine_ic = real_ic − placebo_ic ≥ 0.02`) the PRIMARY, ENFORCING §5.2 sanity
criterion (falling back to the legacy absolute-ratio rule only when the Layer-1a
profile was unavailable). Codex blocked this for two reasons: (1) CI was red on
a stale `renquant-common<0.9` pin, and (2) no pre-registered justification was
shown for freezing `2x`/`0.02` independently of the candidates it currently
rescues.

**WHY-DIR (investigation, not just compliance)**: Tracing the history
(`docs/research/2026-06-28-wf-gate-genuine-ic-calibration-plan.md`, already on
`main`) shows this is **not a new question** — it is the exact same question
Codex already answered on 2026-06-28, on this exact repo, on this exact gate:

- PR #57 (2026-06-23/28) originally tried to enforce this same
  difference-test/margin, "hand-tuned so the 2026-06-23 candidate passed."
  Codex flagged it as gate-overfitting. The revision that landed instead kept
  the absolute-ceiling rule (gate v2) as the ONLY enforced criterion and
  shipped `genuine_ic` as diagnostic-only.
- The 2026-06-28 doc pre-registers exactly what a future enforcement PR must
  show before any switch-over: (1) a historical-corpus replay classifying
  false-accept/false-reject under the absolute rule, (2) synthetic
  leak-injection validation that `genuine_ic` separates edge from leak,
  (3) a threshold frozen from (1)+(2) — **explicitly not tuned to any live
  candidate** (the doc calls out `0.02` itself as "a DISPLAY value only... NOT
  the enforced bar" as of 2026-06-28), and (4) a shadow period running both
  verdicts in parallel over several weekly gates.
- `main` (via a later PR, ahead of where this branch forked) already
  implements exactly the SAFE outcome: `_pooled_placebo_verdict()` computes
  BOTH the enforced absolute-ceiling verdict (gate v2, unchanged) AND the
  shadow-only v3 difference-test verdict (`sanity_placebo_v3_gating: False`,
  always), with a bootstrap CI, a positive-aligned-real guard, and 28 tests in
  `tests/wf_gate/test_genuine_ic_placebo_gate.py` covering both.

This PR's implementation — submitted without apparent awareness of that
history — flips `pass_placebo` back to the difference test as primary, which
is precisely the change Codex already rejected once. None of the four
pre-registered requirements are met here either: no historical replay, no
synthetic injection validation, no independently-frozen threshold (the
`0.02` margin is the same display-only number from 2026-06-28, not a newly
derived one), and no shadow period.

**EVIDENCE**: `main`'s existing implementation is a strict superset —
28 tests vs. this PR's 9, plus CI-bound diagnostics and the guard against a
spuriously-positive `genuine_ic` from two negative ICs. This PR added nothing
that `main` doesn't already have, correctly, in shadow-only form.

**Resolution this round**:
- Merged `main` in. Conflicts in `wf_gate/runner.py` resolved by taking
  `main`'s side wholesale (the enforcing `pass_placebo` change is dropped
  entirely; the existing shadow-only `_pooled_placebo_verdict` stays as-is).
- Removed `tests/wf_gate/test_placebo_difference_gate.py` — it imported
  symbols (`PLACEBO_DIFF_MARGIN`, `_assemble_diagnostic_profiles`) that no
  longer exist post-merge, and it asserted the now-reverted enforcing
  behavior as correct.
- `renquant-common` pin picked up `main`'s already-bumped `<1.0` ceiling via
  the merge (Codex's CI-red finding #1) — no separate change needed.

**NEXT**: This PR is now a no-op relative to `main` (CI-pin fix aside — which
`main` already has). Recommend closing as superseded rather than merging,
since there is nothing left for it to contribute. If genuine enforcement of
the v3 difference test is still wanted, it needs a NEW PR that actually
performs the four pre-registered steps above against fresh evidence, not a
re-post of the already-rejected margin.

## Acceptance criteria (master plan) — corrected

- [ ] Gate uses difference test as enforced criterion — **NOT DONE, and per the
  pre-registered plan should not be done without the 4-step validation below**
- [x] Margin `0.02` exists — but per the 2026-06-28 doc it is explicitly a
  **display-only** reference value, not a validated frozen threshold
- [ ] Historical-corpus replay (pre-registered requirement, not started)
- [ ] Synthetic leak/no-edge injection validation (pre-registered requirement,
  not started)
- [ ] Shadow period vs. enforced rule (pre-registered requirement — `main`
  already runs shadow evaluation continuously; needs a defined observation
  window before any switch-over decision)
