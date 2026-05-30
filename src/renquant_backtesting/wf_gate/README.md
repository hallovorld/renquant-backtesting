# `renquant_backtesting` Phase 1 — copy from umbrella `scripts/`

This package holds **byte-for-byte copies** from the umbrella's `scripts/`
directory, organized by subject per placement-by-owner (renquant-backtesting
owns sim / WF / forensics / LEAN export / analysis). The umbrella files remain
**authoritative and running**; nothing in production references the copies
here yet.

## What's in each subdir

| Subdir | What's here | From umbrella |
|---|---|---|
| `wf_gate/` | WF gate runner + sim driver + sim ledger + all wf_*/walkforward_*.py helpers | `scripts/run_wf_gate.py`, `run_sim_104.py`, `sim_trade_ledger.py`, 18 wf_/walkforward scripts |
| `analysis/` | analyze_*, compute_portfolio_metrics, backtest_and_analyze, trade_*, smoke tests | `scripts/analyze_*.py`, `compute_*.py`, `backtest_*.py`, `trade_*.py`, `smoke_test_model.py`, `qp_cvxpy_smoke.py` |
| `sim/` | reconcile_live_sim | `scripts/reconcile_live_sim.py` |
| `lean_export/` | export_lean_data, export_lean_watchlist | `scripts/export_lean_*.py` |

## Phase plan

1. ✅ **Phase 1 — copy** (2026-05-30): byte-for-byte; nothing live points here.
2. ⏳ **Phase 2 — refactor**: split each module's procedural flow into
   Task/Job/Pipeline per §1c (especially `wf_gate/runner.py`'s 5 stages).
3. ⏳ **Phase 3 — kernel deps**: lazy `kernel.*` imports currently still come
   from the umbrella. Either move those to `renquant-pipeline` or expose
   them as cross-repo deps.
4. ⏳ **Phase 4 — smoke**: run each entry point through umbrella path AND the
   package path on a known fixture, assert byte-equivalent output.
5. ⏳ **Phase 5 — flip callers**: cron / preflight / orchestrator point at
   this package; umbrella scripts become thin shims that forward here.

## How callers should think about this today

| If you need to run it today | use the umbrella script | `python scripts/<name>.py …` |
| If you are reviewing where it SHOULD live | look here | `renquant_backtesting.<subdir>.<name>` |
| If you want to refactor / test cleanly | edit here, smoke-test vs umbrella, then flip caller | see Phase 4-5 |
