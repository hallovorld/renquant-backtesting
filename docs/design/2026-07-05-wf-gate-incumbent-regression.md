# WF Gate v2.1: Incumbent Regression Check Replaces SPY Benchmark Gate

**Date:** 2026-07-05
**Status:** RFC
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

# Incumbent regression: hard gate
incumbent_result = _check_incumbent_regression(
    mean_sharpe, mean_apy, incumbent_wf_meta,
    sharpe_tolerance=0.15, apy_tolerance=0.05,
)
incumbent_ok = incumbent_result["passed"]

pass_sharpe = bool(absolute_ok and incumbent_ok)
```

### New function: `_check_incumbent_regression()`

```python
def _check_incumbent_regression(
    candidate_sharpe: float,
    candidate_apy: float,
    incumbent_wf_meta: dict | None,
    *,
    sharpe_tolerance: float = 0.15,
    apy_tolerance: float = 0.05,
) -> dict:
    """Check that the candidate does not regress vs the incumbent model.

    Returns passed=True if:
    - No incumbent exists (first promote), OR
    - Incumbent has no wf_gate_metadata (legacy model), OR
    - candidate_sharpe >= incumbent_sharpe - sharpe_tolerance
      AND candidate_apy >= incumbent_apy - apy_tolerance
    """
```

Tolerance values:
- `sharpe_tolerance = 0.15`: allow candidate to lag incumbent by up to 0.15
  Sharpe (noise floor of 3-cut WF). A 0.8 incumbent accepts candidates >= 0.65.
- `apy_tolerance = 0.05`: allow up to 5pp APY lag. An 11% incumbent accepts
  candidates >= 6%.

### Loading the incumbent

The incumbent is the **current active production artifact**:
`STRATEGY_DIR / "artifacts/prod/panel-ltr.alpha158_fund.json"`

This path is already known in `runner.py` (line 918 fallback). The function
reads `wf_gate_metadata.wf_3cut_sharpe_mean` and `wf_gate_metadata.wf_3cut_apy_mean`
from the active artifact. If the file doesn't exist, or the metadata is missing,
the check auto-passes (no incumbent to regress against).

### Changes to `main()` (line ~3290)

Add incumbent loading between artifact_usage inspection and the WF run:

```python
incumbent_wf_meta = _load_incumbent_wf_meta()
```

Pass it through to `run_walk_forward()` or compute the check after `wf_result`
returns (cleaner: post-hoc check on the result dict, not inside the WF runner).

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
"incumbent_sharpe": incumbent_result.get("incumbent_sharpe"),
"incumbent_apy": incumbent_result.get("incumbent_apy"),
"incumbent_delta_sharpe": incumbent_result.get("delta_sharpe"),
"incumbent_delta_apy": incumbent_result.get("delta_apy"),
"incumbent_sharpe_tolerance": incumbent_result.get("sharpe_tolerance"),
"incumbent_apy_tolerance": incumbent_result.get("apy_tolerance"),
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

## Backward Compatibility

- Existing artifacts with `benchmark_ok` metadata are unaffected (read-only).
- The `model_acceptance.promote()` function's `_check_wf_gate()` only reads
  `passed` — it doesn't inspect sub-verdicts, so no change needed there.
- `check_model_bundle_consistency.py` in the orchestrator reads `passed` +
  numeric fields — will need a minor update to log the new incumbent fields.

## Test Plan

1. Unit: `_check_incumbent_regression()` — no incumbent, better candidate,
   equal candidate, within-tolerance candidate, regressed candidate.
2. Integration: mock the active artifact path to inject a known incumbent.
3. Gate parity: replay the 2026-07-05 WF gate run and verify the same model
   now PASSES (candidate Sharpe 0.776 vs incumbent 0.697 → delta +0.079 > -0.15).
