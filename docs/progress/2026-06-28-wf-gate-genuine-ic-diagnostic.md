# Progress: WF-gate genuine_ic — reframed to diagnostic-only

Date: 2026-06-28
PR: #57 (renquant-backtesting, `fix/wf-gate-placebo-autocorr-calibration`)
Reviewer: Codex (haorensjtu-dev) — CHANGES_REQUESTED addressed.

## What changed (single durable record)

PR #57 originally flipped the ENFORCED §5.2 placebo sub-gate to decide on
`genuine_ic = aligned_real_ic − placebo_ic`, with the bar hand-tuned so the
2026-06-23 candidate passed. Codex correctly flagged this as gate-overfitting on a
real-money promotion gate. This revision reframes the PR to the SAFE path:

- **Enforced gate UNCHANGED vs main.** Pooled + per-regime placebo sub-gates keep
  the conservative absolute rule (`abs(placebo_ic) < max(0.005, 0.5×|aligned_real_ic|)`).
  `GATE_VERSION` stays 2. The structural-floor candidate still gets the SAME enforced
  verdict (FAIL). No real-money behavior change in this PR.
- **`genuine_ic` shipped diagnostic-only.** Logged + stamped as `sanity_placebo_*`
  (point estimate, positive-aligned-real guard, overlap-aware bootstrap CI lower
  bound, reference display bar), tagged "diagnostic-only, gate unaffected". New
  `gate_diagnostic_version` versions the payload without affecting pass/fail.
- **Pathology guard.** `genuine_ic` is reported only when `aligned_real_ic > 0`;
  negative/zero/NaN aligned-real → `None` (no spurious positive).
- **Uncertainty.** Moving-block bootstrap on per-date real−placebo differences, block
  length ≈ label horizon, respecting overlapping 60d labels; lower 10% quantile.
- **Shuffled-label hard guard** kept as-is.
- **Calibration→enforcement plan** documented in
  `docs/research/2026-06-28-wf-gate-genuine-ic-calibration-plan.md` (historical
  replay, synthetic leak/no-edge injection, pre-registered threshold, shadow period;
  A1 produces the evidence).

## Validation

- `tests/wf_gate/test_genuine_ic_placebo_gate.py` rewritten: asserts the ENFORCED
  verdict is unchanged vs main (structural-floor candidate still FAILS), diagnostic
  fields stamped correctly, and negative-aligned-real → guarded/None genuine_ic.
- Full `tests/wf_gate/` suite: 162 passed.
- ruff: no new findings (3 pre-existing errors at lines 102 / 2339 / 2433 unrelated
  to this change). Pre-existing `renquant_pipeline`-import collection error in
  `tests/walk_forward/test_loader_uri_resolution.py` also unrelated (exists on main).
