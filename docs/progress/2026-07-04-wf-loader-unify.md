# B1: unify the WF loader's verification onto the M6 dispatch

DATE: 2026-07-04
CAMPAIGN: compliance fix campaign B1 (orchestrator PR #297; findings RQ#444
F-2, orchestrator#295 P0-2, orchestrator#296 BT-1). Coordinated with the
umbrella B2 PR (`fix/wf-gate-loader-repoint`). Design authority: orchestrator
`doc/design/2026-07-03-m6-stage2-fingerprint-migration.md` §2a/§3 step 1.

## Architecture decision

Import the pipeline's `fingerprint_dispatch` DIRECTLY (no lift to
renquant-common now):

- the import edge already exists — this repo declares
  `renquant-pipeline>=0.4,<0.5` and this very module already imported
  `renquant_pipeline.kernel.walk_forward.leakage_guard` and
  `...panel_pipeline.panel_scorer`; no new dependency direction;
- the dispatch is M6 migration-WINDOW machinery (flag semantics, census
  telemetry format, deprecated-shim usage) owned by the pipeline per the
  #210 §6 ownership split (runtime admission enforcement → pipeline);
  lifting it mid-window would fork the telemetry contract and demand a
  common release + model-repo review while the window is open;
- the post-window end state (step 5) collapses the dispatch to `verify()`
  exact-match — IF a lift to common is ever worth it, that is the moment
  (a small stable surface), not now. Revisit at M6 stage-2 step 5.

## What changed

`renquant_backtesting.walk_forward.loader` was a 434-line FULL fork of the
pipeline loader (12-char-prefix matcher + venv-coupled bare
`model_content_sha256` recompute — the pipeline#160 hazard; one of three
divergent verifiers behind the 2026-05-27/06-22/07-01 incidents). It is now
a subclass of `renquant_pipeline.kernel.walk_forward.loader
.WalkForwardModelLoader` overriding ONLY the backtesting URI-resolution
layer (`_resolve_uri` strategy-dir inference, preserved verbatim).

Deltas beyond verification, stated explicitly:

- `calibrator_as_of` now loads the pipeline's `GlobalPanelCalibration`
  (runtime contract class) instead of `training_panel.global_calibrator` —
  an umbrella-layout import that CANNOT resolve inside this repo's own
  environment (not on the declared pythonpath; the in-repo path was
  dead-on-arrival). Production in-repo usage (`wf_gate/runner.py`) calls
  `entry_as_of` only. On the real WF corpus under the umbrella layout the
  verdict set is unchanged (see the equivalence table).
- 12-char prefix acceptance: MEASURED over the real corpus — ZERO
  currently-green artifacts rely on it. It survives only inside the
  dispatch's legacy route behind the `accept_legacy_stamps` window flag
  (default ON) and is retired fleet-wide at M6 stage-2 step 4.

## Behavior-invariance proof (protection contract, run 2026-07-04)

Read-only A/B against the REAL live-tree inventory (fingerprint census
GREEN: 47/47 legacy-stamped):

| Leg | old (main) | new (branch) | delta |
|---|---|---|---|
| this loader, 2 in-scope manifests x 43 folds, real `_assert_calibrator_matches_entry` under the umbrella layout + pinned runtime pipeline | 43 PASS / 43 NO_CALIBRATOR | 43 PASS / 43 NO_CALIBRATOR | NONE |
| 12-char-prefix reliance among green matches | — | — | ZERO |

Suite A/B vs pristine main: identical failure set (5 pre-existing
environmental failures both sides: sibling-layout byte-equivalence pins +
umbrella-path-dependent CLI tests); branch adds 8 passing pins in
`tests/walk_forward/test_loader_fingerprint_dispatch.py`.

## Round 2 (codex review — import-surface regression)

Codex: "The new loader imports renquant_pipeline.kernel.panel_pipeline.fingerprint_dispatch
through the panel_pipeline package, and in CI that import path pulls panel_scorer and
hard-requires xgboost during test collection."

Confirmed and fixed — but NOT in this repo. `walk_forward/loader.py`'s import statement
was already correct (`fingerprint_dispatch` is the right owner per M6 stage-2 design);
the bug was `renquant-pipeline`'s `panel_pipeline/__init__.py` eagerly importing
`panel_scorer` (xgboost at module scope) at PACKAGE-import time, which Python triggers
for any submodule import regardless of what that submodule itself needs.

Fixed upstream: `renquant-pipeline#172` converts `panel_pipeline/__init__.py`'s eager
imports to PEP 562 lazy attributes. Verified directly against that fix: this repo's
`walk_forward/loader.py` now imports cleanly with `xgboost` import blocked outright
(the exact CI failure class). 197/197 loader + wf_gate tests pass under the same
xgboost-blocked condition. No code change needed in this repo — this PR's own import
was correct all along.
