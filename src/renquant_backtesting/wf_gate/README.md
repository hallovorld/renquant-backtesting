# `renquant_backtesting.wf_gate` — staging area, NOT yet the live path

This package holds the parallel copy of the umbrella's WF gate plumbing:

| Module here | Copied from (still live in umbrella) | Lines |
|---|---|---|
| `runner.py` | `RenQuant/scripts/run_wf_gate.py` | 2525 |
| `sim_driver.py` | `RenQuant/scripts/run_sim_104.py` | 304 |
| `sim_ledger.py` | `RenQuant/scripts/sim_trade_ledger.py` | 1205 |

## Status

**Copy-only, no refactor yet.** The umbrella files **remain authoritative and
running**; nothing in production references the copies in this package.

This is Phase 1 of the move per `RenQuant/doc/arch/multirepo-sop.md`. The
sequence is:

1. **Phase 1 — copy** (✅ done 2026-05-30): byte-for-byte duplicate so the
   files exist in the right repo per placement-by-owner, without touching the
   live umbrella scripts (§5.4 — don't edit files of running scripts).
2. **Phase 2 — refactor to Task/Job/Pipeline** (pending): split `runner.py`'s
   procedural 5-stage flow into 5 Jobs (`ConfigParityJob`, `RecipeMatchJob`,
   `RunWfSimJob`, `SanityBatteryJob`, `StampArtifactJob`), each with its own
   Tasks. Each module ≤ 50 lines per §1c.
3. **Phase 3 — kernel deps** (pending): the lazy `kernel.*` imports
   (`kernel.panel_pipeline.panel_scorer`, `kernel.walk_forward.loader`,
   `kernel.preflight`) currently still come from the umbrella. Either move
   those to `renquant-pipeline` or expose them as cross-repo deps.
4. **Phase 4 — smoke** (pending): run WF gate on a known artifact through
   the umbrella path AND the package path, assert byte-equivalent metadata
   stamps.
5. **Phase 5 — flip callers** (pending): update `scripts/weekly_wf_promote.sh`,
   `scripts/daily_104.sh`, orchestrator callers to import from this package.
   Umbrella scripts become thin shims that forward to the package.

## What this does NOT do (yet)

- `python -m renquant_backtesting.wf_gate.runner` will **fail at load time** in
  the .venv that does not have the umbrella's `kernel/` on sys.path. That is
  fine — Phase 3 fixes it. Today's invocation continues to be
  `python scripts/run_wf_gate.py` from the umbrella checkout root.
- Tests have not been moved; the existing tests in `RenQuant/tests/` still
  exercise the umbrella scripts.

## How callers should think about this

| If you need to run the WF gate today | use the umbrella script | `python scripts/run_wf_gate.py …` |
| If you are reviewing where this code SHOULD live long-term | look here | `renquant_backtesting.wf_gate.*` |
| If you want to refactor / test cleanly | edit here, mirror to umbrella, run smoke | see Phase 4 |
