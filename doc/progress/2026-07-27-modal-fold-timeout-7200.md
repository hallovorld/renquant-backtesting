# Modal WF fold timeout 3600 -> 7200 (staged-1 T4 probe findings)

STATUS:    delivered
WHAT:      Raise the per-fold Modal timeout default from 3600s to 7200s in
           BOTH places it lives:
           - `wf_gate/modal/executor.py` `--timeout-seconds` CLI default
             (the value the driver bakes into `@app.function` via
             `RENQUANT_WF_MODAL_TIMEOUT_SECONDS` before importing the app
             module);
           - `src/wf_patchtst_modal_app.py` `DEFAULT_TIMEOUT_SECONDS`
             fallback (kept in lockstep so a direct import without the env
             var cannot silently bake the too-small value back in).
           New test `test_timeout_default_covers_measured_fold_runtime`
           asserts both defaults are 7200 (CLI parse + fresh app-module
           import with no env override).
           SCOPE NOTE (honest record): the task brief also asked to land
           the working-tree `EXTRA_BUNDLE_SUBDIRS` +
           `_assert_strategy_config` bundle fix from the dev checkout's
           `feat/wf-patchtst-modal-rescore` working tree. Investigation
           found that fix is ALREADY on origin/main in identical-or-
           stronger form — it was committed to the branch during the codex
           #76 review round and merged via PR #76 (merge 3709774),
           together with its tests
           (`test_bundle_code_missing_strategy_config_fails_closed`,
           `test_assert_strategy_config_passes_when_present`,
           `test_assert_strategy_config_rejects_missing`, plus the
           positive `configs/strategy_config.json` bundle assertion).
           The dev checkout's uncommitted diff is a stale pre-commit draft
           of that same change (its base, local HEAD 1788104, is 3 commits
           behind the branch tip); cherry-picking it verbatim onto main
           would REGRESS ~799 lines of #76 review hardening (run
           quarantine, pinned-assembly bundle_code, provenance fail-close),
           so this PR deliberately does NOT touch bundle_code. The
           uncommitted working tree was left untouched.
WHY/DIR:   The 2026-07-27 staged-1 T4 probe (executor dispatch, 1 fold,
           cutoff 2026-03-02) proved every fold dies at the old 3600s
           default: training alone took 2388.1s, the calibrator leg
           (`fit_calibrator`, mandatory for a usable corpus) had run
           ~1200s and was still running (right-censored) when Modal killed
           the input at exactly 3600s. 3600 < 2388 + calibrator, so a full
           43-fold dispatch at the default would burn the entire training
           spend and return zero folds. 7200 gives the measured train time
           ~2x headroom for the uncensored calibrator leg.
EVIDENCE:  artifact:      probe run log
           `<scratchpad>/modal-probe/logs/pod-run2-full.log` + failed-fold
           provenance
           `<scratchpad>/modal-probe/repo-root/backtesting/renquant_104/
           artifacts/walkforward_patchtst_manifest.json.provenance.json`
           (n_folds_requested=1, n_folds_succeeded=0).
           prod or exp:   experiment — staged-1 probe on the isolated
           Modal Volume; no production surface touched.
           existing data: pod log `train cutoff=2026-03-02 done in
           2388.1s` (17:14:41Z), calibrate leg started 17:14:41Z, input
           cancelled 17:34:42Z; provenance error verbatim:
           `FunctionTimeoutError("Task's current input
           in-01KYJ6J8JGZW04TA59JAHNTQP7:1785170082212-0 hit its timeout
           of 3600s")`.
           best-known?:   yes — single T4 measurement, but a hard lower
           bound: the timeout is a kill switch, so one right-censored
           observation suffices to show 3600 is below the true fold time.
           scope:         ENGINEERING correctness (dispatch config), not a
           model/IC claim.
           Tests: `PYTHONPATH=<this repo src>:<sibling srcs>
           /Users/renhao/git/github/RenQuant/.venv/bin/python -m pytest
           tests/test_modal_wf_patchtst.py tests/wf_gate/ -q` ->
           266 passed, 2 skipped, 0 failed (265+2 baseline + 1 new test,
           no regressions).
NEXT:      (1) full 43-fold dispatch can now use the default; per-fold
           wall time ~3600-4500s expected on T4 — budget accordingly or
           probe a larger GPU; (2) the dev checkout's stale working-tree
           draft on `feat/wf-patchtst-modal-rescore` can be discarded by
           its owner once independently confirmed (NOT done here — brief
           forbade discarding).
