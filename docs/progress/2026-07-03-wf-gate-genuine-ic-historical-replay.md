# Progress: WF-gate genuine_ic (v3) — historical-corpus replay attempt

Date: 2026-07-03
PR: #61 (renquant-backtesting, `feat/wf-gate-placebo-difference`), round 3+
Reviewer: Codex (haorensjtu-dev) — CHANGES_REQUESTED on commit `d6b2b65`
(the shadow-only demotion): "I still do not think the gate threshold is
experimentally justified... I need to see the full historical
candidate/verdict corpus replayed under v3 plus a prospective held-out
shadow run... Until then this belongs as measurement, not authorization."

## What changed (single durable record)

This round attempts the two pieces of evidence Codex's review asks for, and
is explicit about which one is genuinely achievable right now vs. which one
requires real calendar time to pass.

### 1. Historical-corpus replay — ATTEMPTED, negative finding

The WF-gate does not maintain a queryable verdict ledger; each run stamps its
`wf_gate_metadata` directly onto the candidate model artifact it evaluated.
The only available historical corpus is the set of weekly staging/rollback
snapshots already stamped with `sanity_placebo_*`/`model_placebo_profile`
fields, found under the live artifact tree's
`backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.weekly_*.json`
(2026-06-15 through 2026-06-30; 21 files, 20 with placebo data).

**Critical finding: the raw file count overstates the real sample size.**
Grouping by `(run_at, candidate_recipe_fingerprint)` collapses 20 stamped
records to **12 unique (run_at, fingerprint) pairs**, and further inspection
shows those 12 break down to just **2 distinct candidate fingerprints**:

- `sha256:cfdd6cb8e950da0f` (the challenger candidate, re-staged and
  re-evaluated 9 times across 2026-06-17 → 2026-06-30 without a new
  training run — same model, same result every time)
- `sha256:aeb1cd20db700361` (the incumbent/rollback reference, evaluated
  across several rollback snapshots)

**Important correction made during verification:** the top-level `passed`
field in these artifacts is the FULL gate verdict (shuffle leg + placebo leg
+ regime-sanity leg + WF-config-parity, etc., combined) — it is **not** the
placebo sub-gate's own verdict. An early pass at this replay read `passed`
directly and produced a spurious "v2 passes the incumbent" data point; one
of the incumbent's own stamped records carries `sanity_reason: "FAIL:
placebo_ic=+0.0402 ... must be < threshold=+0.0265"` while `passed=True` —
proving the two fields answer different questions. Recomputing the actual
placebo-leg verdict via `_placebo_absolute_rule_pass` (ground truth,
verified against `runner.py`) corrects this.

**Replay result (v3's `genuine_ic > 0.02` rule vs. v2's placebo-leg verdict,
recomputed from raw `sanity_placebo_ic`/`sanity_placebo_aligned_real_ic`),
n=2 independent underlying models, 11 dated evaluations:**

| Candidate | v2 placebo leg | genuine_ic | v3 (shadow rule) | Agreement |
|---|---|---|---|---|
| `cfdd6cb8...` (challenger), 8 of 9 weekly evaluations (06-17,18,20,22,23,27,28,30) | FAIL | 0.0324 | PASS | **disagree** — v3 more lenient |
| `cfdd6cb8...` (challenger), 1 of 9 (06-21/22, rolling window shifted) | PASS | 0.0415 | PASS | agree |
| `aeb1cd20...` (incumbent), all observed | FAIL | 0.0128 | FAIL | agree |

There is **no case in this corpus** where v2's placebo leg passes and v3
fails (the "false reject" direction). The only disagreement is one-directional
and one-candidate: the challenger's placebo leg is borderline (its
`aligned_real_ic`/`placebo_ic` shift slightly week to week because the WF
eval window rolls forward), sitting just above v2's absolute threshold most
weeks and just below it one week — while v3's difference rule stays PASS
throughout since genuine_ic (0.032–0.042) clears 0.02 by a wider margin than
v2's threshold clears its own ceiling.

This is genuinely n=2 independent models, not the 20 raw records or even the
11 dated evaluations — the 8/9 repeats are the *same* frozen model re-staged
week after week, not independent evidence. **This is nowhere near enough
data to estimate a false-accept/false-reject rate for v3**, and does not
demonstrate v3 tracks v2's judgment reliably. It also does not demonstrate
the opposite — the one clean signal is that v3 is *more lenient* than v2 for
the one persistently-borderline candidate observed, which is exactly the
kind of asymmetry a promotion decision needs many more independent
candidates to characterize before trusting. This finding **strengthens
rather than resolves** Codex's original post-outcome-calibration concern.

**Regime-sensitivity red flag (challenger candidate, per-regime genuine_ic
at 2x):**

| Regime | genuine_ic |
|---|---|
| BEAR | **+0.355** |
| BULL_CALM | −0.011 |
| BULL_VOLATILE | −0.077 |
| CHOPPY | −0.011 |

The pooled genuine_ic (+0.032–0.042, which is what clears the 0.02 margin)
is being driven almost entirely by a single regime (BEAR). Three of four
regimes are actually negative. This is a concrete, measured instance of the
"single regime driving the result" failure mode the promotion criterion
below is designed to block.

**Leak/shuffle control:** `sanity_shuffled_ic` passes the hard `<0.005`
guard in every record checked (both candidates). No evidence of a raw
label/feature leakage problem — this is a separate axis from the
threshold-calibration concern and is not in question here.

**Verification of the replay code:** hand-checked one record
(`weekly_20260623T201007Z.staging.json`: `aligned_real_ic=0.0853`,
`placebo_ic=0.0529` → `genuine_ic=0.0324`) against the script's computed
output and against the independently-stamped `model_placebo_profile.pooled.2x.genuine_ic=0.0337`
field (small difference is a different eval-window/method between the two
stamped fields, not a bug — both agree in direction and magnitude).

### 2. Prospective held-out shadow run — genuinely blocked on calendar time

Confirmed by timestamp: no `wf_gate_metadata`-stamped artifact postdates
commit `d6b2b65` (the shadow-only demotion, 2026-07-02T23:02). No scheduled
WF-gate run has occurred since. This cannot be fabricated or approximated —
it requires waiting for real future runs. Tracked as an open next step, not
attempted here.

### 3. Frozen promotion criterion — SPECIFIED (not yet calibrated)

Per Codex's ask for "either a lower-confidence-bound or minimum effective-
sample guard," and informed directly by the two real findings above (n=2 is
not enough; single-regime dominance is a real, observed failure mode), v3's
eventual promotion criterion is pre-registered as:

v3 may be promoted from shadow to enforcing **only when all of**:

1. **Lower-confidence-bound gating, not point estimate.** The gate must use
   `_genuine_ic_block_bootstrap_lower` (already implemented, diagnostic-only
   today) as the decision quantity, not the raw `genuine_ic` point estimate.
   The lower bound must exceed the margin, not just the point estimate.
2. **Minimum effective-N of independent candidates.** At least **N ≥ 12
   independent candidate fingerprints** (not stamped records — see the
   dedup finding above) observed in shadow before promotion is considered.
   This number is chosen to be an order of magnitude past the n=2 this
   replay found available, not a validated statistical power calculation —
   it is a floor, not a target.
3. **No single regime may account for more than 60% of the pooled
   genuine_ic's magnitude.** Computed as
   `max(|per_regime genuine_ic| × regime_weight) / |pooled genuine_ic| < 0.6`
   across the four regimes. This directly targets the BEAR-dominance pattern
   found in the replay above. A candidate that fails this check is not
   promotion-eligible regardless of its pooled genuine_ic.
4. **Fail-closed on CI unavailability.** If the bootstrap CI cannot be
   computed (insufficient overlapping-window samples, degenerate variance),
   the gate must treat this as INCONCLUSIVE, not PASS — never fall back to
   the point estimate.
5. **Family-wise control for per-regime looks**, if the per-regime leg is
   ever proposed for its own enforcement (separate from the pooled leg) —
   not designed in detail here since the per-regime leg stays shadow-only
   per the round-2 fix regardless of pooled-leg promotion.

This is a rule *shape*, pre-registered before seeing whether it would pass —
it is deliberately not tuned to make the two known candidates above pass or
fail in any particular way. Calibrating its exact constants (the "60%"
regime-dominance threshold, the N≥12 floor) against a larger corpus is future
work once more shadow data exists.

## Validation

- Replay script and hand-verification run against real production artifact
  data (read-only; not persisted in-repo since the source data lives in the
  live umbrella artifact tree, not a portable fixture).
- `tests/wf_gate/test_genuine_ic_placebo_gate.py`: added three regression
  tests encoding the real historical (aligned_real_ic, placebo_ic) pairs
  found above — the challenger's typical week (v2 FAIL / v3 PASS), the
  challenger's one exception week (v2 PASS / v3 PASS), and the incumbent
  (v2 FAIL / v3 FAIL) — each asserting against the actual gate functions
  (`_placebo_absolute_rule_pass`, `_genuine_ic_value`,
  `_placebo_difference_pass`) so this finding doesn't silently rot if the
  gate logic changes.
- Full `tests/wf_gate/` suite passes (see PR for count).

## Status / next

- [x] Historical-corpus replay attempted — negative finding (n=2 independent
  models; one-directional disagreement — v3 more lenient than v2 — observed
  for the one borderline candidate; regime-dominance red flag).
- [x] Frozen promotion-criterion rule shape specified.
- [ ] Prospective shadow run — blocked on real calendar time, no action
  possible until scheduled WF-gate runs accumulate post-`d6b2b65`.
- [ ] Larger historical corpus (if one becomes available from a different
  source than the live artifact tree's weekly snapshots) would let the
  criterion's constants actually be calibrated rather than floor-set.
- v2 remains the sole enforcing gate. v3 remains shadow-only. This PR is
  **not** proposing enforcement in this round — that remains explicitly
  blocked pending the prospective run.
