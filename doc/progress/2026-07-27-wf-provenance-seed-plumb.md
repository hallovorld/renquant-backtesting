# Plumb the sim RNG seed through the WF sim drivers — provenance chain (pipeline#215 §3)

STATUS:    delivered
WHAT:      Backtesting piece of the merged WF sim-time provenance chain
           (pipeline#215 design §3, pipeline#216 emitters, umbrella#531
           adapter). Final API read from umbrella origin/main
           (`backtesting/renquant_104/sim/runner.py` + `kernel/walk_forward/
           provenance_adapter.py`): `run_backtest` exposes exactly ONE
           provenance-relevant kwarg — keyword-only `seed: Optional[int] =
           None`. It mints its own `sim_run_id` (`wfsim-<utc>-<uuid8>`) and
           constructs the JSONL sink internally behind `walkforward.enabled`,
           recording `seed` + revision pins at sim start. There are NO
           `sim_run_id`/sink kwargs for callers to forward — the seed is the
           whole caller-side surface.
           What was actually missing vs already-flowing:
           - `wf_gate/sim_driver.py` (run_sim_104 body): MISSING — the driver
             owned no seed at all (no `--seed` CLI arg; both `run_backtest`
             calls omitted `seed`, so sims always ran seed=None/legacy
             non-deterministic). Added `--seed` (int, default None = legacy
             behavior) and forwarded it to BOTH legs: the candidate run and
             the golden comparison run get the SAME seed, keeping the A/B
             paired while each leg still mints its own `sim_run_id`.
           - `wf_gate/dump_walkforward_sim_metrics.py`: ALREADY FLOWING —
             `--seed` existed and was passed straight to
             `run_backtest(seed=args.seed)` and echoed into the metrics JSON.
             VERIFIED the call path drops nothing: `args.seed` reaches the
             call unmodified. Its forced `persistence={"enabled": False}` is
             the documented design §2.1 leg — when `walkforward.enabled`, the
             sim adapter (umbrella side) still emits provenance with
             `persisted:false` on the score-committed record; nothing to do
             in this repo. Change here is docs-only (comment pinning the
             verified finding).
           Out of scope (unchanged, noted for the record):
           `wf_gate/runner.py::_sim_driver_cmd` passes no `--seed`, so WF
           gate sim cuts still run seed=None (valid provenance records, seed
           field null). Wiring runner-level seeds belongs to the prereg'd
           rerun batch, which decides the seed set.
WHY/DIR:   codex reviews on model#64/#65/#66: post-hoc reconstruction of
           which fold/artifact/seed produced which score is inadmissible;
           provenance must persist at generation time. The umbrella sink
           records the seed it is handed — a driver that never hands one
           makes every run's seed field permanently null and the sim
           unreplayable. Note the sink only activates once the pipeline pin
           advances past #216 (pre-#216 pin: loud warning + no emit,
           byte-identical sim); this plumb is inert until then and changes
           no behavior at seed=None (run_backtest's `_apply_seed(None)` is
           an explicit no-op).
EVIDENCE:  `PYTHONPATH=<common src>:<base-data src>:<artifacts src>:
           <pipeline src>:src /Users/renhao/git/github/RenQuant/.venv/bin/
           python -m pytest -q` (full repo suite) -> 389 passed, 8 skipped,
           0 failed (385 + 4 new). New
           `tests/wf_gate/test_sim_driver_seed_plumb.py` (4 tests) captures
           the `run_backtest` call via a fake `sim.runner` and asserts:
           sim_driver forwards `--seed 7`; sim_driver with no flag still
           forwards an explicit `seed=None`; the golden-compare leg receives
           the SAME seed as the candidate leg (2 calls, [7, 7]); the dump
           driver forwards `--seed 11` with `persistence={"enabled": False}`
           intact and echoes the seed in the metrics JSON.
NEXT:      Pipeline + backtesting pin advance (past pipeline#216 and this
           PR) ships WITH the prereg'd rerun batch (XGB multi-seed) — that
           is the moment WF sims start emitting seeded provenance JSONL;
           the batch also decides whether `wf_gate/runner.py` cut legs get
           explicit per-cut seeds.
