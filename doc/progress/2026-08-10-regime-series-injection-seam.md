# Regime-series injection seam for the BEAR-exit confirmatory arms (orch#962 B2)

STATUS:    delivered. Default-off: without `--regime-series` nothing is
           imported, nothing is patched, and the driver's behavioral surface
           is byte-identical to before (invariance test pins the config
           handed to `run_backtest`, the equity-JSON key set, and the
           untouched `RegimeFinalizeTask.run`). Nothing deployed, no live
           path touched.

WHAT:      `wf_gate/regime_injection.py` (new) + `wf_gate/sim_driver.py`
           `--regime-series <path>`:
           * `load_regime_series(path)` — CSV `date,regime` (optional
             header) or JSON `{date: label}` -> `RegimeSeries(by_date,
             sha256, source)`; sha256 is of the exact source bytes; any
             structural defect (dup/bad date, empty label, empty series,
             unknown extension) is a hard `RegimeInjectionError`.
           * `install_regime_injection(series)` — wraps the umbrella's
             `kernel.pipeline.task_regime.RegimeFinalizeTask.run` (the task
             that commits `ctx.regime`): computed finalize still runs, then
             the injected per-date label REPLACES `ctx.regime` +
             `ctx.final_regime` before any downstream decision task reads
             them; `regime_counts` follow the injected series and
             `_regime_evidence` records {computed, injected, sha256}. The
             class-level patch covers BOTH driver legs (candidate + golden
             comparison) — the prereg's paired-arm/same-series contract.
             Install is fail-closed (kernel module or anchor class missing,
             double install, label outside `kernel.config.REGIMES` when
             importable) and verified by reading the marker back off the
             class.
           * Per-bar guard: a sim date not in the series raises — never a
             silent fallback to the computed regime. Driver preflight
             (`required_sim_dates` = `spy_df.loc[start:end].index`, mirroring
             `run_backtest`'s own bar derivation) exits 5 before any compute
             when coverage is incomplete; verdicts echoed to stdout for the
             wrapper's tee (the input-bundle idiom).
           * Sim output metadata: equity JSON + trade-output extra_metrics
             gain `regime_injection = {active, sha256, source, n_dates,
             label_set_validated}` ONLY when injection is active.

WHY/DIR:   orch#962 blocker B2: the frozen BEAR-exit prereg (orch
           `doc/design/2026-08-08-bear-exit-prereg.md` §3/§3.1) requires book
           sims under INJECTED regime series — D2 placebo (episode-block
           label permutation, 200 seeds) and D4 shifts (+5/+10/+20 days) —
           but the simulator computed regimes internally with no injection
           point. This seam is the minimal runnability repair: the
           confirmatory runner can now key both arms to a permuted/shifted
           series with the series' identity bound into the run's outputs.

EVIDENCE:  artifact:      regime chain read from the umbrella (read-only):
                          `kernel/pipeline/job_regime.py` (Hurst -> CUSUM ->
                          GMM -> BEAROverride -> TrendOverlay -> Finalize)
                          and `task_regime.py::RegimeFinalizeTask` committing
                          `ctx.regime`/`ctx.final_regime`; decision readers
                          consume `ctx.regime` (`adapters/sim.py:1774,1793`).
           prod or exp:   experiment tooling only — the seam is opt-in via a
                          CLI flag on the WF sim driver; the daily/live path
                          never constructs it.
           existing data: none replaced — no prior injection point existed
                          anywhere in the sim call chain (B2's finding).
           best-known?:   yes — first regime-injection seam in the system.
           scope:         renquant-backtesting only; the umbrella and the
                          orchestrator are untouched. Tests
                          `[VERIFIED — pytest -q in this worktree]`: suite
                          716 passed / 1 skipped with the change vs 687
                          passed / 1 skipped on clean origin/main (delta =
                          the 29 new tests); the SAME 3 environment-bound
                          failures on both sides (2x umbrella byte-
                          equivalence drift, 1x REAL run001 bundle identity)
                          — pre-existing, reproduced on unmodified main in
                          the canonical checkout, unrelated to this diff.

NEXT:      orch#962 confirmatory runner: generate the 200-seed episode-block
           permutation series + the three shifted series from the production
           regime artifact, run both arms per series through
           `--regime-series`, and assemble the §3.1 D1-D5 verdict with each
           run's recorded series sha256.
