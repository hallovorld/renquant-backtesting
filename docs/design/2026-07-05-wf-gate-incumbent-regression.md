# WF Gate v2.1: Incumbent Regression Check Replaces SPY Benchmark Gate

**Date:** 2026-07-05
**Status:** RFC (revised — round 2, addressing Codex review)
**Author:** Claude (operator-directed)
**Repo:** renquant-backtesting
**Module:** `src/renquant_backtesting/wf_gate/runner.py`

## Problem

The WF gate's benchmark check (`benchmark_ok`) requires the candidate model to
beat SPY in at least 2/3 cuts on both Sharpe and APY. In sustained bull markets
(SPY Sharpe >1.0, APY >15%), a multi-name stock-selection strategy with
Sharpe ~0.7-0.8 structurally cannot clear this bar. Result: every weekly promote
is rejected and requires `operator_authorized_override`, defeating the gate's
purpose as an automated quality check.

The current production model (trained 2026-06-21) was promoted with override.
The fresh retrain (2026-07-05, eval_ic=0.062, WF Sharpe=0.776, APY=11.2%) also
fails the same benchmark check (beat SPY Sharpe 1/3, APY 0/3).

## Root Cause

The benchmark comparison answers the wrong question. The relevant promote
question is not "does this model beat SPY?" (a structural property of the
strategy, not the model) but **"is this model at least as good as the one it
replaces?"** A model that's equal-or-better than the incumbent and is fresher
should auto-promote. A model that's significantly worse should be rejected.

## Round-2 revision summary

Codex's round-1 review raised four issues: (1) demoting the regime/regression
guard to advisory changes the failure mode from too-strict to too-permissive;
(2) the incumbent itself is not a clean comparator (promoted with override);
(3) the tolerance values (`sharpe_tolerance=0.15`, `apy_tolerance=0.05`) were
asserted, not justified; (4) mean-only comparison can hide a one-cut blowup.

This revision addresses all four: (a) defines a concrete, checkable
admissible-incumbent policy — the incumbent is a hard comparator **only** if
its own gate metadata shows a clean, non-diagnostic pass, otherwise the gate
falls back to the pre-existing absolute floor alone, not a silent auto-pass;
(b) grounds the tolerances in the one real empirical anchor this repo
currently has (within-run cut dispersion) while being explicit that this is
n=1 and provisional, with a concrete validation trigger; (c) replaces the
mean-only incumbent comparison with a **paired per-cut** check using the
fixed `CUTS` calendar windows, so a candidate that wins on average but loses
badly on one cut is caught. It also surfaces a genuine, previously
unaddressed data-integrity gap in the proposed mechanism (see "Known
limitations" below) discovered while investigating (b).

## Design

### Changes to `run_walk_forward()` (lines 1479-1489)

**Before:**
```python
absolute_ok = mean_sharpe >= 0.40 and n_pos >= 2
benchmark_ok = (has_spy_sharpe and has_spy_apy
                and mean_sharpe_vs_spy >= 0 and n_beat_spy_sharpe >= 2
                and mean_apy_vs_spy >= 0 and n_beat_spy_apy >= 2)
regime_ok = not regime_benchmark_failures
pass_sharpe = bool(absolute_ok and benchmark_ok and regime_ok)
```

**After:**
```python
absolute_ok = mean_sharpe >= 0.40 and n_pos >= 2

# SPY comparison: advisory (logged + stamped, not gated)
benchmark_advisory = (has_spy_sharpe and has_spy_apy
                      and mean_sharpe_vs_spy >= 0 and n_beat_spy_sharpe >= 2
                      and mean_apy_vs_spy >= 0 and n_beat_spy_apy >= 2)
regime_advisory = not regime_benchmark_failures

# Incumbent regression: hard gate (paired per-cut, admissibility-checked)
incumbent_result = _check_incumbent_regression(
    candidate_cuts=results,
    incumbent_wf_meta=incumbent_wf_meta,
    sharpe_tolerance=0.15, apy_tolerance=0.05,
)
incumbent_ok = incumbent_result["passed"]

pass_sharpe = bool(absolute_ok and incumbent_ok)
```

Note `benchmark_ok`/`regime_ok` are **not removed** — they become
`benchmark_advisory`/`regime_advisory`, still computed and stamped every run
(per Codex point 1, demoting them to advisory needs the actual regime-failure
signal to remain visible, not disappear from the metadata; see "Regime
protection" below for what replaces their gating role).

### Admissible incumbent comparator (Codex point 2)

**Policy:** the incumbent's own `wf_gate_metadata` is admissible as a hard
comparator if and only if:

1. `wf_gate_metadata.diagnostic_only is False` — i.e.
   `skipped_required_gates == []`. This is a real, already-persisted field
   (`_required_validation_skip_reasons()` in this file) covering exactly the
   emergency-flag paths that constitute a policy exception:
   `walk_forward_skipped`, `sanity_skipped`, `trade_gates_skipped`,
   `trade_monotonicity_pass_open_allowed`, `config_parity_skipped`,
   `trade_trace_disabled`.
2. `wf_gate_metadata.passed is True` (a diagnostic_only=False *failed* run is
   not an acceptance record either — this should not normally occur since
   `promote()` refuses non-passing staging artifacts, but the check verifies
   it directly rather than assuming).

If either condition fails, the incumbent is **not admissible**. Critically,
this does **not** mean the check auto-passes (the original draft's "no
wf_gate_metadata → passed=True" default would have silently no-opped exactly
when it matters most — right now, with a real override-tainted incumbent).
Instead: **`incumbent_ok` falls back to `absolute_ok` alone** — the
pre-SPY-gate-era floor (`mean_sharpe >= 0.40 and n_pos >= 2`), which predates
and is independent of both the SPY gate and any override history. This keeps
automation working (the RFC's stated goal) without quietly promoting the
override-tainted incumbent's specific numbers into the new implicit
standard. Once a genuinely clean (`diagnostic_only=False`, `passed=True`)
model is promoted, it becomes the first admissible incumbent and the
paired-cut regression check activates against it.

**Current state, verified directly against the live active artifact** (read
2026-07-05; see "Known limitations" for why this can't be taken as a
permanent historical record): `artifacts/prod/panel-ltr.alpha158_fund.json`
currently shows `wf_gate_metadata.diagnostic_only = False`,
`skipped_required_gates = []`, but `passed = False` (it fails on
`benchmark_ok`, not on a skipped gate) and its `wf_3cut_sharpe_mean = 0.776`
— i.e. these are the **fresh retrain's own numbers**, not the original
2026-06-21 promotion's. This is direct evidence of the data-integrity gap
below: the file's stamped metadata does not reliably reflect what justified
its own promotion, because at least one later diagnostic gate run has
already overwritten it in place. Under the policy above, whether today's
incumbent is admissible cannot be verified from this file alone right now —
treat it as **not admissible** until a `promote()`-time acceptance-history
record exists (see below), and rely on `absolute_ok` alone in the interim.

### Regime protection replacement (Codex points 1 and 4): paired per-cut check

Mean-only comparison can hide a one-cut blowup: a candidate could win big in
one cut and lose badly in another and still clear a mean-based bar. `CUTS`
(module-level, `runner.py` lines 129-133) is a **fixed** 3-window calendar
schedule:

```python
CUTS = [
    ("2024-01-02", "2024-12-31"),
    ("2024-07-01", "2025-06-30"),
    ("2025-04-01", "2026-03-28"),
]
```

Every WF gate run (candidate and, when it was itself gated, the incumbent)
evaluates the *same* three calendar windows, and each cut's `start`/`end`/
`sharpe`/`apy` is already stamped into `wf_gate_metadata["cuts"]`
(`runner.py` line 3444). This makes a genuine paired-per-cut comparison
possible without inventing new data collection:

```python
def _check_incumbent_regression(
    candidate_cuts: list[dict],
    incumbent_wf_meta: dict | None,
    *,
    sharpe_tolerance: float = 0.15,
    apy_tolerance: float = 0.05,
) -> dict:
    """Paired per-cut regression check against an admissible incumbent.

    Returns passed=True if:
    - No incumbent artifact exists (first promote), OR
    - Incumbent is not admissible per policy above (diagnostic_only=True,
      or passed=False, or unreadable) — falls back to absolute_ok alone,
      this function returns passed=True with admissible=False so the
      caller can log the fallback explicitly, OR
    - Incumbent's stamped cut windows no longer match the current CUTS
      constant — paired comparison is invalid (CUTS changed since the
      incumbent was evaluated); falls back the same way, OR
    - Admissible AND cut-window-aligned: candidate is no-worse-than-incumbent
      (within tolerance) in at least 2 of 3 cuts, on BOTH Sharpe and APY.

    A cut "passes" if:
      candidate_sharpe[i] >= incumbent_sharpe[i] - sharpe_tolerance
      AND candidate_apy[i] >= incumbent_apy[i] - apy_tolerance
    """
```

This directly closes the failure mode Codex describes: a candidate that
matches the incumbent's mean by winning huge in cut 1 and losing badly in
cuts 2-3 would only pass 1/3 cuts and fail the `>= 2/3` requirement, even
though a mean-only check might have passed it.

`benchmark_advisory`/`regime_advisory` remain stamped every run (Codex point
1's concern that regime signal shouldn't vanish) — they're just no longer
gating. The gating role that `regime_ok` used to play (per-regime
cross-checking) is now subsumed by the per-cut check above, since each of
the 3 fixed cuts spans a materially different regime mix (2024 calm bull,
2024-25 mixed, 2025-26 volatile) — a per-cut failure is, in practice, a
regime-correlated failure.

### Tolerance justification (Codex point 3)

**We do not have a multi-run historical dataset of paired incumbent-vs-
candidate deltas in this repo** (verified: no `promote_history` or
per-artifact acceptance log exists yet that would let us compute an
empirical distribution of "how much do two models of genuinely comparable
quality differ, run to run"). The values below are grounded in the one real
empirical anchor currently available, but are explicitly **provisional**.

- `sharpe_tolerance = 0.15`: the runner already computes `wf_3cut_sharpe_std`
  — the standard deviation *within a single model's own 3 cuts*. Read
  directly off the live artifact today: `wf_3cut_sharpe_std = 0.205` for the
  fresh retrain. This is a different quantity than "noise between two
  separate models' cut means," but it's a real, available upper-bound-ish
  reference: if one model's own cuts vary by ~0.20 Sharpe just from regime
  mix, requiring less than that (0.15) as the tolerance for comparing two
  *different* models is not obviously too generous. This is n=1 and must not
  be treated as validated.
- `apy_tolerance = 0.05` (5pp): **has no equivalent empirical anchor at all**
  — the runner does not currently compute an APY analogue of
  `wf_3cut_sharpe_std` (verified: no `apy_std`/`wf_3cut_apy_std` field
  exists). On an 11% strategy, a flat 5pp allowance is large in relative
  terms (~45% of the mean). This value is **placeholder**, not justified.

**Concrete validation trigger** (blocks nothing now, but must be revisited):
add `wf_3cut_apy_std` to the runner's stamped output (a small, low-risk
addition — `_s.stdev(apys)` alongside the existing `_s.stdev(sharpes)`).
Once 5 or more *admissible* (`diagnostic_only=False`, `passed=True`) gate
runs exist, treat their `wf_3cut_sharpe_std`/`wf_3cut_apy_std` values as an
empirical noise-floor sample and set `sharpe_tolerance`/`apy_tolerance` to a
stated function of that sample (e.g. the mean, or a percentile) rather than
an asserted constant. Until then, both tolerances are logged in the stamped
metadata as `sharpe_tolerance`/`apy_tolerance` (already planned) with an
additional `tolerance_basis: "provisional-n1"` field so any future audit of
a specific promote decision can see the values were not yet evidence-backed.

### Loading the incumbent

The incumbent is the **current active production artifact**:
`STRATEGY_DIR / "artifacts/prod/panel-ltr.alpha158_fund.json"`

This path is already known in `runner.py` (line 918 fallback). The function
reads `wf_gate_metadata` (`diagnostic_only`, `passed`, `cuts`) from the
active artifact. If the file doesn't exist, or fails the admissibility
policy above, the check falls back to `absolute_ok` alone (see policy
section — not a silent unconditional auto-pass).

### Changes to `main()` (line ~3290)

Add incumbent loading between artifact_usage inspection and the WF run:

```python
incumbent_wf_meta = _load_incumbent_wf_meta()
```

Pass it through to `run_walk_forward()` or compute the check after `wf_result`
returns (cleaner: post-hoc check on the result dict, not inside the WF runner,
since it needs `wf_result["cuts"]` for the paired comparison).

### Changes to `_compute_overall_pass()`

Add `incumbent_result` as a parameter. The incumbent check is a top-level gate
like parity or trade_contract:

```python
def _compute_overall_pass(..., incumbent_result: dict) -> bool:
    ...
    return (
        bool(wf_result["passed"])
        and _sanity_result_passed(sanity_result)
        and bool(trade_contract_result["passed"])
        and bool(trade_gate_result["passed"])
        and bool(alpha_economics_result["passed"])
        and validation_scope_ok
        and bool(parity_result.get("passed", True))
        and bool(incumbent_result.get("passed", True))
    )
```

### Changes to `wf_meta` stamping (line ~3418)

Add incumbent comparison fields:

```python
"incumbent_ok": incumbent_result.get("passed"),
"incumbent_admissible": incumbent_result.get("admissible"),
"incumbent_fallback_reason": incumbent_result.get("fallback_reason"),
"incumbent_sharpe_by_cut": incumbent_result.get("incumbent_sharpe_by_cut"),
"incumbent_apy_by_cut": incumbent_result.get("incumbent_apy_by_cut"),
"n_cuts_within_tolerance": incumbent_result.get("n_cuts_within_tolerance"),
"incumbent_sharpe_tolerance": incumbent_result.get("sharpe_tolerance"),
"incumbent_apy_tolerance": incumbent_result.get("apy_tolerance"),
"tolerance_basis": incumbent_result.get("tolerance_basis"),
"incumbent_source": incumbent_result.get("source"),
# Existing SPY fields remain (advisory)
"benchmark_advisory": benchmark_advisory,
"regime_advisory": regime_advisory,
```

### CLI flag

Add `--incumbent-artifact` optional arg to override the default active artifact
path (for testing with a specific incumbent).

### Gate version

Keep `GATE_VERSION = 2` (the structural gate didn't change; threshold semantics
did). Add a `gate_policy_version` field = `"v2.1-incumbent"` to the stamped
metadata for forensic traceability.

## Known limitations

1. **Acceptance-record mutability (discovered during this revision).** The
   active artifact's `wf_gate_metadata` is not a stable historical record of
   *its own promotion decision* — any later invocation of
   `run_wf_gate.py --artifact <active_path>` (e.g. a diagnostic or
   validation run, as appears to have happened here on 2026-07-05) silently
   overwrites it with that run's results. `promote()` itself copies the
   whole staging file (including its metadata) into `active_path`, so the
   metadata is correct *at promotion time*, but nothing prevents a
   subsequent read-only-intent gate run from clobbering it afterward. This
   is why "current state" above could not be verified as the *original*
   2026-06-21 promotion's evidence. **Recommended follow-up (out of scope
   for this RFC):** either (a) `run_wf_gate.py` should warn or refuse when
   `--artifact` points directly at a known active/prod path outside of the
   staging→promote flow, or (b) `promote()` should additionally append the
   accepted `wf_gate_metadata` to a separate, append-only acceptance-history
   log (e.g. a JSONL file or a `decision_ledger`-style table) that the
   incumbent-regression check reads from instead of the mutable active file.
   Until one of these lands, the admissibility check above is a best-effort
   read of current file state, not a guaranteed audit trail.
2. **Tolerances are provisional (n=1 anchor).** See "Tolerance justification"
   above — `apy_tolerance` in particular has no empirical anchor at all yet.
3. **Cut-window alignment is a precondition, not a guarantee.** If `CUTS` is
   ever changed, an incumbent evaluated under the old windows cannot be
   paired against a candidate evaluated under new ones; the check must
   detect and fall back (see `_check_incumbent_regression` docstring above),
   not silently compare mismatched windows.

## Backward Compatibility

- Existing artifacts with `benchmark_ok` metadata are unaffected (read-only).
- The `model_acceptance.promote()` function's `_check_wf_gate()` only reads
  `passed` — it doesn't inspect sub-verdicts, so no change needed there.
- `check_model_bundle_consistency.py` in the orchestrator reads `passed` +
  numeric fields — will need a minor update to log the new incumbent fields.

## Test Plan

1. Unit: `_check_incumbent_regression()` — no incumbent, incumbent not
   admissible (`diagnostic_only=True`) falls back to `absolute_ok` (not
   auto-pass), incumbent admissible + all 3 cuts better, incumbent
   admissible + exactly 2/3 cuts within tolerance (passes), incumbent
   admissible + only 1/3 cuts within tolerance (fails, catches the
   one-cut-blowup-hidden-by-mean case), cut-window mismatch (falls back).
2. Integration: mock the active artifact path to inject a known incumbent
   with per-cut data, both admissible and non-admissible cases.
3. Gate parity: replay the 2026-07-05 WF gate run and verify the same model
   now PASSES *only if* the incumbent it's compared against is itself
   admissible under the new policy — if not (current live state, per "Known
   limitations"), verify it PASSES via the `absolute_ok`-only fallback
   (mean Sharpe 0.776 >= 0.40, 3/3 cuts positive), not via a suspect
   incumbent comparison.
