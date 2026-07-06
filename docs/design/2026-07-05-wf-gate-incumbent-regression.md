# WF Gate v2.1: Incumbent Regression Check Replaces SPY Benchmark Gate

**Date:** 2026-07-05
**Status:** RFC (revised — round 3, addressing Codex review). **The hard
incumbent-regression gate MUST NOT activate until Phase 0 below has landed
and produced at least one genuine acceptance-log entry** — see "Phase 0" and
"Sequencing" below.
**Author:** Claude (operator-directed)
**Repo:** renquant-backtesting
**Module:** `src/renquant_backtesting/wf_gate/runner.py`,
`src/renquant_backtesting/forensics/model_acceptance.py`

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
badly on one cut is caught. It also surfaced a genuine, previously
unaddressed data-integrity gap in the proposed mechanism (see "Known
limitations" below) discovered while investigating (b).

## Round-3 revision summary

Codex's round-2 review confirmed the paired per-cut framing and admissible-
incumbent policy address most of round-1's gaps, but elevated the round-1
"known limitation" — the active artifact's `wf_gate_metadata` being mutable
and silently overwritable by a later diagnostic gate run — from a limitation
to an **actual blocker**: comparing against a mutable record is a governance
bug, not just an observability gap, because the hard gate could be comparing
against evidence that was never the incumbent's actual promotion-time
evidence.

This revision adds **Phase 0** below: a concrete, append-only acceptance-log
design that stabilizes the comparator source at its one genuine write point
(`promote()`), so no later diagnostic run can ever affect what the hard gate
reads. The incumbent-regression gate (Phases 1+, i.e. everything already
designed above) is unchanged in its own logic, but **explicitly cannot
activate as a hard gate until Phase 0 has landed** — see "Sequencing" below.

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
2026-07-05; see "Phase 0" below for why this can't be taken as a permanent
historical record): `artifacts/prod/panel-ltr.alpha158_fund.json` currently
shows `wf_gate_metadata.diagnostic_only = False`, `skipped_required_gates =
[]`, but `passed = False` (it fails on `benchmark_ok`, not on a skipped gate)
and its `wf_3cut_sharpe_mean = 0.776` — i.e. these are the **fresh retrain's
own numbers**, not the original 2026-06-21 promotion's. This is direct
evidence of the exact governance bug Phase 0 exists to close: the file's
stamped metadata does not reliably reflect what justified its own promotion,
because at least one later diagnostic gate run has already overwritten it in
place. This is not a hypothetical risk — it already happened to the artifact
currently in production. Per Phase 0's design, today's incumbent has no
`promotions.jsonl` entry and is therefore **not admissible**; the gate relies
on `absolute_ok` alone until this artifact (or its successor) is next
genuinely promoted through `promote()`.

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

**Superseded by Phase 0 below.** The incumbent's evidence is loaded from the
append-only acceptance log (`_load_incumbent_wf_meta()` reads the latest
`active_path`-matching entry from `_acceptance_log/promotions.jsonl`), **not**
from `wf_gate_metadata` on the mutable active artifact file directly. If no
acceptance-log entry exists for the current active artifact (e.g. it was
promoted before Phase 0 landed, or promoted via a `strategy_config.json`
bypass — see Phase 0's scope note), the incumbent is treated as **not
admissible**, exactly like a missing/unreadable file, and the check falls
back to `absolute_ok` alone.

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

## Phase 0: stabilize the incumbent comparator source (REQUIRED prerequisite)

Codex round 2: comparing against a mutable record is a **governance bug**,
not an observability gap, so this cannot stay a documented limitation — it
must be closed before the hard gate can read from the active artifact at all.

### Root cause, verified directly in code

`runner.py::main()` (line ~3512-3518) unconditionally does:

```python
md = artifact.get("metadata") or {}
md["wf_gate_metadata"] = wf_meta
artifact["metadata"] = md
written = _write_artifact_payload(artifact_path, artifact)
```

for **whatever path `--artifact` points to** — there is no distinction
between "this is a staging artifact about to be gated for promotion" and
"this is the live active artifact being diagnostically re-checked." Every
invocation of the umbrella's `scripts/run_wf_gate.py --artifact <active_path>`
(a valid, supported diagnostic use — re-checking the currently-active model,
e.g. after a data refresh) overwrites the active file's `wf_gate_metadata` in
place, including the fields (`wf_3cut_sharpe_mean`, per-cut arrays, `passed`)
that a hard incumbent-regression gate would need to trust as "what actually
justified this model's promotion." This is exactly what happened to the
current active artifact between its 2026-06-21 promotion and 2026-07-05: a
diagnostic run replaced the promotion-time evidence with the fresh retrain's
numbers, in the same file.

By contrast, `model_acceptance.py::promote()` (line 798) is called **exactly
once per genuine promotion** — the staging→active atomic swap — and already
validates `_check_wf_gate(data, staging_path)` (requiring `passed=True`) as
a precondition. This is the one clean, existing hook point where a durable
record can be captured.

### Design: append-only acceptance log

Add one file, written only by `promote()`:

- **Path:** `active_path.parent / "_acceptance_log" / "promotions.jsonl"` —
  this directory already exists as a convention in this file (`reject()`
  archives rejected artifacts to `active_path.parent / "_acceptance_log"`,
  line 890); `promotions.jsonl` is a new sibling file in the same directory,
  not a new convention.
- **Write path (`promote()`, after the atomic active-swap succeeds, before
  returning):** append one JSON line:
  ```python
  {
      "promoted_at": <UTC ISO-8601 timestamp of this promote() call>,
      "active_path": str(active_path),
      "staging_source": str(staging_path),
      "wf_gate_metadata": <the exact dict from `data["metadata"]["wf_gate_metadata"]`
                            that `_check_wf_gate` just validated — i.e. the
                            staging artifact's metadata AS VALIDATED, not a
                            re-read of the post-swap active file>,
  }
  ```
  Append-only (`open(path, "a")`), never rewritten or truncated. This makes
  the record structurally immune to a later diagnostic `runner.py main()`
  invocation, because that code path never calls `promote()` and never
  touches this file.
- **Read path (`_load_incumbent_wf_meta()` in `runner.py`):** read
  `promotions.jsonl` line-by-line (or tail it — the file is expected to stay
  small, one promotion at a time, at most a handful of KB per entry), filter
  to entries whose `active_path` matches the current active-artifact path,
  and take the **last matching entry** as the incumbent's `wf_gate_metadata`.
  If the file doesn't exist, or has no entry matching the current
  `active_path`, the incumbent is **not admissible** (same as today's
  missing-file case) — the gate does **not** fall back to reading the
  mutable active-file metadata directly, since that's precisely the
  unstable source this design replaces.
- **Migration for the current active artifact:** the artifact currently in
  production was promoted (2026-06-21) before this log exists, so it has no
  entry and will correctly be treated as not-admissible until it is next
  genuinely re-promoted (which writes the first entry) or until an
  operator-run one-time backfill script seeds a single entry from the
  `.previous.json` rotation history if that's still on disk with credible
  metadata (a fast-follow, not required to land Phase 0 itself — the
  `absolute_ok`-only fallback already covers this gap safely).

### Known scope boundary (not fully closed by Phase 0 alone)

`assert_artifact_gated()`'s own docstring documents a real prior incident:
the 2026-06-05 PatchTST promotion reached production via a **direct
`strategy_config.json` scorer-pointer edit**, bypassing `promote()` entirely.
An artifact activated that way would never get a `promotions.jsonl` entry
either — Phase 0 closes the specific bug Codex flagged (diagnostic-run
overwrite of an artifact that WAS promoted via `promote()`), but does not by
itself retroactively cover a `promote()`-bypass activation. Recommended
fast-follow (out of scope for Phase 0): wherever `assert_artifact_gated()` is
already invoked as a pre-write guard at the config-write boundary, also
append a `promotions.jsonl` entry there, so both promotion paths are
captured. Until that lands, an artifact activated via the bypass path is
correctly treated as not-admissible by the same "no matching log entry"
rule above — safe by default, not silently trusted.

### Sequencing

The hard incumbent-regression gate (everything under "Design" above)
**must not activate** — i.e. `_compute_overall_pass()` must not gate on
`incumbent_result["passed"]` in a way that can fail a promote — until:

1. Phase 0 (the append-only log + read/write wiring) has landed and its
   own tests (see Test Plan) pass, AND
2. at least one genuine `promote()` call has occurred after Phase 0 landed,
   seeding a real `promotions.jsonl` entry for the then-current active
   artifact.

Until both hold, `_check_incumbent_regression()` always finds no admissible
incumbent (no log, or no matching entry) and the effective behavior is
identical to `absolute_ok` alone — i.e. deploying the code in this RFC is
safe immediately (it can't be stricter or more permissive than today until
Phase 0 has actually produced history to compare against), but the
regression-catching value of Phases 1+ only exists after Phase 0 has run
at least once.

## Known limitations

1. **Acceptance-record mutability — RESOLVED by Phase 0 above** (was:
   round-1 limitation, elevated to blocker in round-2 review). See "Phase 0"
   for the append-only-log design that replaces the mutable active-file
   read.
2. **Tolerances are provisional (n=1 anchor).** See "Tolerance justification"
   above — `apy_tolerance` in particular has no empirical anchor at all yet.
3. **Cut-window alignment is a precondition, not a guarantee.** If `CUTS` is
   ever changed, an incumbent evaluated under the old windows cannot be
   paired against a candidate evaluated under new ones; the check must
   detect and fall back (see `_check_incumbent_regression` docstring above),
   not silently compare mismatched windows.
4. **Promote()-bypass activation paths are not covered by Phase 0 alone.**
   See "Known scope boundary" under Phase 0 — a `strategy_config.json`
   direct-edit promotion (as happened 2026-06-05) would not seed a log
   entry either; correctly falls back to not-admissible, not silently
   trusted, but the fast-follow (wiring `assert_artifact_gated()`'s call
   site to also append) is not yet in scope.

## Backward Compatibility

- Existing artifacts with `benchmark_ok` metadata are unaffected (read-only).
- The `model_acceptance.promote()` function's `_check_wf_gate()` only reads
  `passed` — it doesn't inspect sub-verdicts, so no change needed there.
- `check_model_bundle_consistency.py` in the orchestrator reads `passed` +
  numeric fields — will need a minor update to log the new incumbent fields.

## Test Plan

### Phase 0 (must land and pass first — see "Sequencing")

0. `promote()` appends exactly one well-formed JSON line to
   `_acceptance_log/promotions.jsonl` per call, containing the staging
   artifact's validated `wf_gate_metadata` (not a re-read of the post-swap
   active file). Two consecutive promotions append two lines; the file is
   never rewritten or truncated. A `runner.py main()` invocation with
   `--artifact <active_path>` (the diagnostic-overwrite scenario Codex
   flagged) must leave `promotions.jsonl` byte-for-byte unchanged — this is
   the regression test that directly proves the governance bug is closed.
   `_load_incumbent_wf_meta()` reads the last `active_path`-matching entry
   and returns "not admissible" when the file is absent or has no matching
   entry (covering both the pre-Phase-0 and the promote()-bypass cases from
   "Known scope boundary").

### Phases 1+ (the incumbent-regression gate itself — unchanged from round 2)

1. Unit: `_check_incumbent_regression()` — no incumbent, incumbent not
   admissible (`diagnostic_only=True`) falls back to `absolute_ok` (not
   auto-pass), incumbent admissible + all 3 cuts better, incumbent
   admissible + exactly 2/3 cuts within tolerance (passes), incumbent
   admissible + only 1/3 cuts within tolerance (fails, catches the
   one-cut-blowup-hidden-by-mean case), cut-window mismatch (falls back).
2. Integration: mock `promotions.jsonl` to inject a known incumbent entry
   with per-cut data, both admissible and non-admissible cases — NOT the
   active artifact's own metadata directly (that's the source Phase 0
   replaces).
3. Gate parity: replay the 2026-07-05 WF gate run and verify the same model
   now PASSES via the `absolute_ok`-only fallback (mean Sharpe 0.776 >= 0.40,
   3/3 cuts positive) — since, per "Sequencing," no genuine `promotions.jsonl`
   entry exists yet for the current incumbent, the paired-cut comparison
   cannot activate and must not be exercised as if it had.
