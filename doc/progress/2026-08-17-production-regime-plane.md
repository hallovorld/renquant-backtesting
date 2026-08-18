# ONE production regime-label plane — accessor + committed corpus (orch#985 item 1)

STATUS:    delivered. Research metrology ONLY — no serving path, no strategy
           config, no GMM artifact, no trading decision is touched. The WF
           gate's SANITY leg already replayed the production chain, so gate
           verdicts on that leg are unchanged BY CONSTRUCTION (the leg's
           `build_regime_series` now merely delegates to the extracted
           accessor — same code, one path).

WHAT:      * `analysis/regime_plane.py` (new) — `production_regime_labels()`,
             the ONE accessor for production-plane labels, refactor-extracted
             (not copied) from `analyze_manifest_sanity_placebo.
             build_regime_series` together with its config/GMM/SPY loaders.
             `build_regime_series` is now a thin delegate, so the sanity leg
             and every new consumer share one code path. Plane contract:
             `RENQUANT_REGIME_PLANE=production|legacy_stateless`, default
             production, typos rejected (fail-closed).
           * `tools/publish_production_regime_labels.py` (new) — deterministic
             publisher of the committed per-date corpus
             `doc/research/data/production_regime_labels.csv` (+ provenance
             manifest: input sha256s, the replayed chain copy =
             renquant-pipeline lifted `kernel/regime.py`+`task_regime.py`+
             `regime_hmm.py` with file sha256s, the config flag-state under
             which the two divergent chain copies are behaviorally equal,
             date bounds, row count, occupancy, series sha256). Publish-time
             FAIL-CLOSED cross-check: the corpus must reproduce the #985 memo
             replica facts over 2016-11-01..2026-08-14 — measured
             2,459 trading days / 413 BEAR days / 57 BEAR episodes,
             EXACT match, recorded in the manifest.
           * Site (i) repoint: `wf_gate/runner.py::cut_market_context` now
             emits `regime_plane` + `regime_counts` from the production
             accessor by default; `legacy_stateless` reproduces the old
             stateless + SPY-grid keys verbatim. Downstream aggregation keys
             (`dominant_regime`, `regime_counts_total`) added; the legacy
             `hmm_regime_counts*` keys survive only under the hatch (grep
             across orchestrator/pipeline/model/common found ZERO external
             consumers of the old keys).

WHY/DIR:   #985 measured FOUR regime label planes at 25-70% same-day
           agreement (P2) and version-unstable fold covariates (P8);
           ranked item 1 = consolidate every regime-keyed research surface
           onto the production task-chain plane. The committed corpus is the
           data-plane channel for import-restricted consumers: the
           renquant-model harness PR (cross-referenced) consumes it BY PATH
           because model-family boundary tests forbid importing
           renquant_pipeline/renquant_backtesting (codex round-2 on
           model#65). The corpus ALSO doubles as the input for the upcoming
           serving-plane power-map re-derivation (#985 "Implication for the
           MoE design", consequence (a) for orch#984) — second consumer,
           by design.

EVIDENCE:  artifact:      doc/research/data/production_regime_labels.manifest
                          .json — `memo_985_crosscheck.exact_match = true`
                          (expected == measured == 2459/413/57)
                          `[VERIFIED — publish-time computation against the
                          committed #985 memo facts M:stats_S.n_days,
                          M:stats_S.occupancy.BEAR, M:bear_episodes_serving]`
           prod or exp:   neither — research metrology; no trading behavior
                          change; sanity-leg gate verdicts unchanged BY
                          CONSTRUCTION (delegation, not reimplementation)
           existing data: corpus spans 2016-02-16..2026-08-17 (2,641 rows;
                          publisher was asked for 2016-01-04 but the replay's
                          own ≥30-bar warmup guard skips the first 29 SPY
                          dates — recorded, not hidden). Occupancy
                          BULL_CALM 1929 / BEAR 419 / BULL_VOLATILE 185 /
                          CHOPPY 108 `[VERIFIED — manifest occupancy]`
           best-known?:   yes — first committed production-plane label series
                          anywhere in the system (the #985 memo's replica CSV
                          lives in the orchestrator and is memo-frozen at
                          2026-08-14, not a consumable contract)
           scope:         renquant-backtesting only. The chain DRIFT found en
                          route (pipeline lifted task_regime.py carries
                          2026-06-11 bear-audit guards the umbrella copy
                          lacks; equal only while those flags stay unset) is
                          NOT fixed here — filed as an issue on
                          hallovorld/renquant-pipeline; the manifest records
                          the flag-state assumption.

TESTS:     baseline origin/main: 721 passed / 1 failed / 7 skipped — the
           failure (`test_lineage_stage2.py::test_REAL_run001_bundle_identity
           _seam_and_admissibility`) is PRE-EXISTING and untouched by this
           change. After: same pre-existing failure only; +10 new tests
           (plane resolution incl. typo rejection; accessor determinism +
           BEAR-on-crash on a synthetic fixture; wrapper==accessor identity;
           publisher two-run byte-identity; committed-corpus/manifest
           integrity incl. the memo cross-check; site (i) default +
           escape-hatch behavior). All hermetic — synthetic SPY/GMM/config,
           no production DB reads in tests.

ESCAPE HATCH: every repointed site honors
           `RENQUANT_REGIME_PLANE=legacy_stateless` for reproduction of
           historical results; documented at each site.
