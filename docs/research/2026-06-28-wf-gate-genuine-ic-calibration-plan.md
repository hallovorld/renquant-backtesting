# WF-gate genuine_ic — calibration → enforcement plan

Date: 2026-06-28
Status: DIAGNOSTIC SHIPPED / ENFORCEMENT DEFERRED
Owner: research workstream A1 (placebo-gate calibration)
Related: PR #57 (renquant-backtesting), `repro_m6_placebo_confound`,
`docs/research/2026-06-25-vol-gate-opportunity-cost.md` (precedent: ship-as-diagnostic-first)

## Why this doc exists

The §5.2 time-shift placebo sub-gate of the WF promotion gate decides whether a
freshly-trained model is leak-free enough to trade **real money**. It has
chronically false-rejected weekly candidates since ~2026-06-08. The suspected cause
is a **structural label-autocorrelation floor**: at the gate shift (2×label_horizon =
120 trading rows for the daily `fwd_60d` label), the label is itself
cross-sectionally autocorrelated (`label_autocorr_ic ≈ +0.04`), so for a leak-free
model `placebo_ic ≈ genuine_edge + autocorr_floor`. The conservative absolute rule
`abs(placebo_ic) < 0.5×|aligned_real_ic|` then sits *below* that floor and can never
pass — independent of the model's genuine edge.

A candidate fix is to gate on the difference `genuine_ic = aligned_real_ic −
placebo_ic` (the shared autocorr floor cancels). **PR #57 does NOT enforce that.**
Codex's CHANGES_REQUESTED correctly flagged that flipping a real-money gate's
enforced threshold, with the bar hand-tuned so the single candidate under review
passes, is gate-overfitting. So PR #57 ships the decomposition as **diagnostic-only**
(logged + stamped, gate verdict UNAFFECTED) and defers enforcement to a separately-
calibrated later PR. This doc is the pre-registered plan for that later PR.

## What PR #57 ships (diagnostic-only, gate UNCHANGED)

- Enforced placebo sub-gate is **identical to `origin/main`**: the conservative
  absolute rule `abs(placebo_ic) < max(0.005, 0.5×|aligned_real_ic|)` (pooled +
  per-regime). `GATE_VERSION` stays **2**. No real-money behavior change; the
  structural-floor candidate still gets the SAME verdict (FAIL) it got before.
- `genuine_ic`, a **positive-aligned-real guard**, a multi-shift placebo profile,
  `label_autocorr_ic`, and an **overlap-aware conservative lower confidence bound**
  on `genuine_ic` are LOGGED and STAMPED as `sanity_placebo_*` diagnostics, tagged
  "diagnostic-only, gate unaffected". A new `gate_diagnostic_version` versions that
  payload without touching pass/fail.
- The shuffled-label control (`abs(shuf_ic) < 0.005`) is unchanged — the HARD
  true-leak guard.

## Pathology guard (Codex point #4)

`genuine_ic` is reported **only when `aligned_real_ic > 0`**. A "positive"
`genuine_ic` produced by a NEGATIVE aligned-real IC minus a MORE-negative placebo IC
(e.g. −0.05 − (−0.09) = +0.04) is meaningless — there is no real edge to certify.
The guard returns `None` in that case (and for zero / NaN inputs), so the diagnostic
can never surface a spurious positive.

## Uncertainty (Codex point #6)

The diagnostic CI is a **moving-block bootstrap** on the per-date `(aligned_real_ic −
placebo_ic)` differences, with block length ≈ the label horizon in trading days, so
the serial dependence induced by the overlapping 60d labels is preserved in each
resample (an i.i.d. bootstrap would understate variance and overstate confidence).
We stamp the lower 10% quantile of the bootstrap mean as `ci_lower`. This is
diagnostic-only in PR #57; the enforcement PR will gate on a conservative lower
bound, not a point estimate.

## Enforcement requires (pre-registered, before any GATE_VERSION bump)

The future enforcement PR MUST present, and tie its threshold to, ALL of:

1. **Historical replay** of `genuine_ic` (point estimate + CI) over the full set of
   accepted/rejected candidates since the gate began, classifying false-accept vs
   false-reject under the absolute rule. The structural-floor hypothesis must hold
   on the *distribution*, not one candidate.
2. **Synthetic injection validation**: inject (a) known leakage (future-derived
   features / overlapping-label contamination) and (b) known no-edge noise, and show
   `genuine_ic` (with its CI / lower bound) **separates edge from leak** — leaky and
   no-edge models must fail, edge models must pass.
3. **A PRE-REGISTERED threshold** derived from (1)+(2) — explicitly NOT tuned to any
   live candidate. The reference bar surfaced in PR #57 (`max(0.02,
   0.25×|aligned_real_ic|)`) is a DISPLAY value only and is NOT the enforced bar.
4. **A shadow period** running the genuine_ic decision in parallel with the enforced
   absolute rule over several weekly gates, with both verdicts logged, to confirm the
   replay/injection conclusions hold out-of-sample before any switch-over.

Only after all four are green does a later PR bump the **enforced** gate version and
switch `pass_placebo` to the calibrated genuine_ic rule. The shuffled-label hard
guard stays in place regardless.

## Status / next

- [x] Diagnostic decomposition + guard + CI shipped (PR #57), gate UNCHANGED.
- [ ] A1: historical replay over accepted/rejected candidates.
- [ ] A1: synthetic leak / no-edge injection separation study.
- [ ] A1: pre-registered threshold from the calibration.
- [ ] Shadow over several weekly gates.
- [ ] Enforcement PR (separate) — bumps enforced GATE_VERSION.
