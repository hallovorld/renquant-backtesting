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
