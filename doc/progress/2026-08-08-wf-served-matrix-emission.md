# WF replay served-matrix emission — the orch#905 wiring

STATUS:    delivered. Default-off: with no sink the replay is byte-identical to
           before (the emission branch is never entered). Nothing deployed,
           no live path touched.

WHAT:      `wf_gate/wf_sanity_paired.py`:
           * `run_wf(...)` gains `cuts=CUTS` (testability) and
             `served_sink=None`. When a sink is set, each fold's TEST-partition
             per-date cross-section `(ticker, score)` is persisted BEFORE
             `cs_ic` collapses it to a scalar — the exact emission point named
             in the orch#905 recon comment.
           * `ServedMatrixEmission` — persists via the pipeline#268 sink's
             `write_served_matrix(out_dir, rows, manifest)` (explicit out_dir;
             `PersistServedMatrixTask` is InferenceContext-shaped and is the
             live path's interface, not a replay loop's). One parquet+sidecar
             pair per test date, manifest carries fold_train_end/label/seed.
           * `emit_served_matrix_main` CLI (`--emit-served-matrix` dispatch):
             one clean unperturbed seed=42 pass over a panel with emission.

WHY/DIR:   Stage −1 (orch#911) funnelled every open MoE question into one
           blocker: the panel arm exists for 33 served dates. This wiring
           produces the walk-forward point-in-time panel cross-section the
           frozen gate (sd(paired ΔIC) < 0.0929), the §4.3 transfer, and
           Stage 0 all require. Point-in-time is structural: each fold's
           booster trains only on dates ≤ the fold's train end.

EVIDENCE:  artifact:      orch#905 recon comment (2026-08-08) — emission point
                          `run_wf`, before the `cs_ic` collapse; sink API read
                          from pipeline served_matrix_sink.py
           prod or exp:   experiment tooling — replay-only; the lane guard
                          makes replay rows non-mistakable for served-live rows
           existing data: logs/served_matrix/ holds ONE live day (2026-08-05,
                          pipeline#268's live call site); the replay side has
                          never emitted — this is that emitter
           best-known?:   yes — first served matrix from replay anywhere in
                          the system
           scope:         renquant-backtesting only; pipeline needs NO change
                          (`write_served_matrix` already takes an explicit
                          out_dir). The orchestrator is untouched, per the
                          ownership split on orch#905.

TWO REFUSALS BUILT IN (both tested):
  1. A perturbed arm (shuffle/shift) with a sink raises — shuffled scores
     persisted as a served matrix would let a placebo be read as evidence.
     The alternative (silently skipping emission) would hide a caller bug.
  2. A lane name without the `wf_replay` prefix raises — a replay row must
     never be mistakable for something the book actually served.

TESTS:     tests/test_wf_served_matrix_emission.py — 5 passed:
             default behaviour unchanged without a sink (determinism pinned);
             one parquet+sidecar pair per test date, full cross-section,
             manifest fields exact; emitted row count == the scored test rows;
             perturbed arms refuse a sink and write NOTHING; lane prefix
             enforced.
           Synthetic panel, one tiny fold via the new `cuts` param — no real
           data dirs touched.

NEXT:      Run the emitter over the production panel dataset to produce the
           541-date point-in-time panel arm (a compute run, needs no review),
           then execute the §10 confirmatory protocol exactly as frozen —
           label-interval purge row governing — and report PASS/KILL.
