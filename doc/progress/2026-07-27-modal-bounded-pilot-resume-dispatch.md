# Modal bounded pilot/resume dispatch + cost projection (model#82 P0)

STATUS:    delivered
WHAT:      Make the $20 HARD budget cap of the renquant-model#82 prereg
           OPERATIONALLY enforceable in the WF Modal executor. The codex
           P0 ruling on model#82, verbatim:

           > P0: the stated $20 HARD cap is not enforceable by the frozen
           > invocation. `--staged 43` selects all 43 cutoffs, and the
           > executor's `dispatch_folds()` fans all requests out through
           > `train_fold_remote.map` at once. The proposed check occurs
           > only after three pods complete, by which time the remaining
           > pods have already been dispatched and can continue accruing
           > cost; `modal app stop` is not a pre-dispatch budget guard.
           > The $16.8 GPU-only projection plus $1.45 already spent also
           > leaves only $1.75 for the explicitly acknowledged overhead.
           >
           > Before merge/freeze, make the budget control operationally
           > enforceable. The clean solution belongs in
           > `renquant-backtesting`: add a tested bounded-dispatch/resume
           > mechanism (or an explicit-cutoff selection mechanism) that
           > runs exactly a three-fold pilot, records the observed-cost
           > decision, and dispatches exactly the remaining 40 only after
           > the projection is below the remaining cap, without
           > retraining/overwriting the pilot folds and while preserving
           > one auditable corpus manifest. Then update this prereg with
           > that merged dependency, frozen SHA, and the exact commands.
           > A prose tripwire cannot satisfy a hard spending cap.

           Delivered in `wf_gate/modal/executor.py` (app module untouched):

           1. `--select-cutoffs <ISO,ISO,...>` — explicit fold selection,
              mutually exclusive with `--staged` (argparse group). Every
              date must be ON the start/end/cadence corpus grid, no
              duplicates; input is normalised to grid (chronological)
              order. The dispatched set is EXACTLY what the prereg froze.
           2. `--run-id <existing>` resume: `partition_resume()` inventories
              the run namespace BEFORE any cloud call. A fold already on
              disk that passes the existing integrity gate
              (`validate_fold_promotable`, minus the diagnostic-run marker)
              is SKIPPED — never retrained/overwritten (the pod map fan-out
              carries only absent folds; `collect_and_write` additionally
              hard-errors on any returned pod result colliding with an
              existing fold). A SELECTED fold that exists but FAILS
              integrity is a pre-dispatch hard error (exit 2). A resumed
              run whose prior provenance names a different recipe_id is
              refused (one namespace = one recipe). The run's single
              manifest is REBUILT over the union of existing+new folds via
              the reviewed `write_manifest`; the provenance sidecar is
              rebuilt over the union and APPENDS one per-dispatch audit
              record (`dispatches[]`: dispatched/skipped cutoff lists,
              Modal app_id, per-pod facts, failures, volume_commit_id,
              optional `--dispatch-note` for the observed-cost GO
              decision), with prior `pod_facts` preserved.
           3. `--print-cost-projection` (+ required `--run-id`,
              `--project-folds N`, `--rate-usd-per-hour X`) — reads the
              run's provenance sidecar, averages completed pods'
              `elapsed_seconds`, prints projected cost for N more folds.
              Pure stdout; never dispatches (refuses `--execute`); no
              baked-in GPU price.

           Deviations from the task sketch (adjusted to code reality):
           - `dispatch_folds()` now returns `(results, dispatch_info)` so
             the Modal app_id lands in the audit record (4 existing test
             call sites updated).
           - Recipe-mismatch refusal on resume added (not in the sketch):
             without it, phase 2 could silently mix hyperparameters into
             the "one auditable corpus".
           - Existing-fold entries are reconstructed from the on-disk
             metadata sidecar (`training_contract`); `lookahead_days`
             falls back to the manifest default (60) when the contract
             omits it — same default the collector already used.
           - Panel-existence/AC7/readiness checks now run only when there
             ARE folds to dispatch; a resume no-op rebuild makes zero
             cloud calls and appends an empty-dispatch audit record.
WHY/DIR:   A prose tripwire ("watch 3 pods then `modal app stop`") cannot
           enforce a hard cap because `.map` fans out all 43 folds at
           dispatch time. Bounding the map fan-out itself (pilot = exactly
           the selected folds; remainder = a second bounded dispatch that
           cannot touch the pilot) turns the cap into a mechanical
           property of the invocation, which the prereg can freeze as
           exact commands.
EVIDENCE:  artifact:      `src/renquant_backtesting/wf_gate/modal/executor.py`
           + `tests/test_modal_wf_patchtst.py` (12 new tests: selection
           exactness / off-grid + duplicate + mutual-exclusion errors /
           resume skip with captured map fan-out carrying ONLY the absent
           fold / hard-fail on existing-but-invalid via `main()` with no
           modal import / no-op rebuild with zero cloud calls / recipe
           mismatch refusal / overwrite-collision refusal / union manifest
           at one path / dispatch-history append / projection math +
           CLI gating).
           prod or exp:   experiment tooling — no production path written;
           no Modal calls anywhere (fake SDK injected in tests; CLI
           verified with --dry-run only).
           existing data: dry-run on the frozen model#82 matrix reproduces
           recipe_id `sha256:b4e47e2cd77af660` with a 3-fold explicit
           selection normalised to grid order.
           best-known?:   yes — the skip/hard-error partition reuses the
           reviewed `validate_fold_promotable` gate rather than inventing
           a second integrity definition; the union manifest goes through
           the reviewed `write_manifest` leakage validation.
           scope:         ENGINEERING (dispatch control + audit records);
           no model/IC claim.
           Tests: `make test` with umbrella venv python + sibling-src
           PYTHONPATH -> 422 passed, 8 skipped, 0 failed (410+8 baseline
           at origin/main 9942bce + 12 new, no regressions).
NEXT:      (1) model#82 prereg update: cite this PR as the merged
           dependency + frozen SHA, and freeze the exact three commands
           (pilot `--select-cutoffs` 3 folds / `--print-cost-projection`
           / resume remainder with `--dispatch-note` recording the
           observed-cost GO); (2) the projection's $/h rate must come from
           the prereg (no default is baked in — deliberate).
