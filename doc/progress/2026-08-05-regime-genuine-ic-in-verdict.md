# 2026-08-05 — the gate already knew; it just never said so (orch#805, #799 item 2)

## The finding this surfaces

The gate computes a full per-regime placebo profile on every run and stamps it
four levels deep in `metadata.wf_gate_metadata.model_placebo_profile.per_regime`,
where nobody read it. Read out of the 2026-07-06 staging stamp
`[VERIFIED — 2026-08-05, read directly; independently re-derived]`, at the 2×
shift the enforced placebo leg itself uses:

| regime | n_dates | aligned_real_ic | placebo_ic | genuine_ic | label_autocorr_ic |
|---|---|---|---|---|---|
| BEAR | 50 | +0.3509 | +0.0162 | **+0.3346** | +0.0296 |
| BULL_CALM | 363 | +0.0300 | +0.0595 | **−0.0295** | +0.0509 |
| BULL_VOLATILE | 11 | +0.0741 | +0.1542 | **−0.0800** | +0.0213 |
| CHOPPY | 28 | +0.0196 | +0.0606 | **−0.0410** | +0.0563 |
| pooled | 452 | +0.0659 | +0.0570 | +0.0089 | +0.0482 |

From the same artifact: BULL_CALM is 489 of 751 regime days and **136 of the
strategy's 154 buys**; BEAR is 73 days and **zero** buys. The pooled +0.0089 that
every promote/reject decision has been read off is a regime-mix artifact.

## The change

Reporting only. Two pure functions and two call sites:

- `regime_genuine_ic_summary()` flattens the 2× cell per regime, keeping
  `aligned_real_ic`, `placebo_ic` and `label_autocorr_ic` alongside `genuine_ic` —
  the number is not interpretable alone (BULL_CALM's placebo is mostly label
  autocorrelation, +0.0509 of +0.0595, which is the whole argument that the ratio
  rule is the wrong instrument).
- `format_regime_genuine_ic()` prints one line, **worst regime first**, next to
  the VERDICT line, and the summary is stamped as `sanity_regime_genuine_ic` so
  downstream consumers read one key instead of walking four levels.

A rejection that says `placebo ratio 0.95` sends the reader to the gate. One that
says `BULL_CALM=-0.0295(n=363)` sends them to the model.

## What it does NOT do

It decides nothing. A test reads `_compute_overall_pass`'s body and fails if
either helper or the new key ever appears inside it. A regime with no 2× cell is
OMITTED, never zero-filled — an absent measurement must read as absent, not as
"measured, and it is zero".

Suite: 9 new tests.

## Review round 2 (codex on bt#105)

Codex confirmed the change is reporting-only and that every number in the PR
body matches the artifact, including the 2×/120-session shift — then blocked on
the anti-vacuity guard, correctly: it scanned only the slice from
`def _compute_overall_pass` to `def _sanity_result_passed`, missing the other
verdict-producing spans. Above all `run_sanity_battery`, where
`pass_all = pass_shuf and pass_placebo and pass_regime` is formed — a future
reference there would change behaviour without touching the scanned slice, so
the advertised guard was not guarding.

The guard is now parametrised over every pass/fail-producing function
(`_compute_overall_pass`, `_sanity_result_passed`, `_placebo_difference_pass`,
`_placebo_absolute_rule_pass`, `_pooled_placebo_verdict`, `run_sanity_battery`),
each asserted to EXIST so a rename empties the list loudly instead of passing
vacuously. Two more tests: one proves the body extractor returns
`run_sanity_battery`'s real body and that a planted reference in it is detected;
one asserts the summary IS reachable from the stamping path, because
reporting-only must not quietly become unreachable.

16 tests.

## Review round 3 (codex on bt#106)

The anti-vacuity regression asserted only that the planted string was PRESENT in
the modified body — tautological, and it would still pass if the live check
stopped inspecting bodies at all.

The assertion is now a shared helper, `assert_decides_nothing(body, func)`,
called by BOTH the live check and its regression. The regression requires it to
RAISE on the planted body, and to still pass on the unplanted one (a guard that
rejects everything proves nothing either).

Writing that test immediately found a real bug in my own extractor: an
off-by-one dropped the leading `d` of `def`, so five of six bodies were
truncated. Fixed, and pinned — every extracted body must now start with
`def <name>(`, exceed 120 chars, and contain a `return`/`assert`, because a
body that is only a signature would satisfy a substring check vacuously.

22 tests.

## Review round 4 (codex on bt#106)

Two more, both correct:

1. **The extractor swallowed the NEXT function's decorator lines.** `runner.py`
   has no decorators today, so no guarded span was mis-scanned — but the
   extractor was wrong, and "it happens not to matter here" is not a fix. It now
   walks back over trailing top-level decorators and blank lines, pinned by a
   synthetic `def first / @dec / def second` fixture rather than by the current
   file.
2. **`main` was omitted from the guarded spans**, and `main` is where the final
   verdict is assembled (`overall_pass = _compute_overall_pass(...)`, then
   `sys.exit(0 if overall_pass else 1)`). A future
   `overall_pass = _compute_overall_pass(...) and sanity_regime_genuine_ic`
   would have evaded the guard exactly as the `run_sanity_battery` omission did.

`main` needs a NARROWER rule than the others, because it legitimately mentions
the reporting symbols — it is where they are STAMPED. So the check is on the
verdict itself: **every statement touching `overall_pass` must be free of the
reporting symbols**, and the test asserts such statements exist (one calling
`_compute_overall_pass`, one calling `sys.exit`) so it cannot scan nothing. A
companion test plants a reference into those statements and requires the same
assertion path to RAISE.

25 tests.
