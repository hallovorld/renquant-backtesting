# Modal execute-time hard cost cap (model#82 P0 round 2)

STATUS:    delivered
WHAT:      Enforce the renquant-model#82 prereg's $20 HARD cap AT EXECUTE
           TIME inside the WF Modal executor — round 2 of the codex P0
           ("a prose tripwire cannot satisfy a hard spending cap"): #81
           bounded WHAT can dispatch; this bounds WHETHER it may, in
           dollars, before a single cloud call.

           New execute-mode flags (`wf_gate/modal/executor.py`):
           - `--max-total-usd <float>` — the hard all-in cap. REQUIRED
             (argparse error) whenever `--execute` is combined with
             `--select-cutoffs` (both prereg phases), together with an
             explicit `--run-id` (both phases must target ONE namespace;
             phase 1 creates it, phase 2 resumes it).
           - `--rate-usd-per-hour <float>` — required with the cap; the
             same shared flag `--print-cost-projection` uses. No baked-in
             GPU price, deliberately.
           - `--pre-spend-usd <float>` (default 0) — dollars already
             billed to the grant outside this run's pod_facts.
           - `--overhead-frac <float>` (default 0.15, documented) —
             multiplier on every GPU-time dollar covering the observed
             non-GPU accrual (image build/pull, queue, Volume
             storage/egress), so the cap compares an ALL-IN projection.

           Gate placement: in `main()` after the resume partition + the
           one-run-one-recipe belts, BEFORE `modal_readiness()` (the
           first modal import) and before staging/dispatch. Formula as
           implemented (`compute_cost_gate`, pure math on the run's
           provenance `pod_facts`):

             usd_per_sec     = rate / 3600 x (1 + overhead_frac)
             measured_usd    = sum(completed pods' elapsed_seconds)
                               x usd_per_sec
             per_fold_bound  = mean(completed elapsed)   if any completed
                               timeout_seconds           if none (phase 1
                               WORST CASE — the timeout is the hardest
                               upper bound a fold can bill)
             remaining_usd   = per_fold_bound x n_to_dispatch x usd_per_sec
             projected_total = pre_spend_usd + measured_usd + remaining_usd
             GO  iff projected_total <= max_total_usd, else REFUSED ->
             exit 4, one clear line with the full calculation, nothing
             dispatched.

           Audit: the full calculation + verdict is persisted into the
           dispatches[] audit record AUTOMATICALLY — GO rides the dispatch
           record (`dispatch_meta["cost_gate"]`); REFUSED is appended to
           the sidecar by `persist_refused_cost_gate` (creating a minimal
           identity-stamped sidecar on a phase-1 refusal so the later
           retry inherits the record and the recipe belts see the claimed
           recipe_id). `--dispatch-note` remains optional free-text
           context only — the verdict does not rely on it.

           Scope notes: a no-op rebuild (nothing to dispatch) spends
           nothing and is not gated; `--staged --execute` (legacy path)
           does not REQUIRE the cap but honours it when given.
WHY/DIR:   The codex P0 on model#82: watching 3 pods then `modal app
           stop` is not a pre-dispatch budget guard — by the time the
           check runs, the whole fan-out has been dispatched and accrues.
           With #81 the fan-out is bounded to the selected absent folds;
           with this PR the executor itself refuses to launch a dispatch
           whose all-in projection breaches the cap, so the prereg can
           freeze exact capped commands instead of prose.
EVIDENCE:  artifact:      `src/renquant_backtesting/wf_gate/modal/executor.py`
           (`compute_cost_gate`, `persist_refused_cost_gate`, parse_args
           requiredness, main() gate) + `tests/test_modal_wf_patchtst.py`
           (6 new tests: measured-basis math incl overhead + pre-spend;
           phase-1 worst-case timeout basis GO + REFUSED; argparse
           requiredness incl run-id + rate; phase-1 CLI refusal exit 4
           with NO modal import + persisted REFUSED record; phase-2 CLI
           refusal from real pilot pod_facts with partition respected;
           GO path prints + gate record rides the dispatches[] entry).
           prod or exp:   experiment tooling; no production path written;
           zero Modal calls (fake SDK in tests; real CLI exercised only
           to the exit-4 refusal, which precedes any modal import).
           existing data: real-CLI demo against a scratch repo-root on
           the frozen matrix (recipe `b4e47e2cd77af660`): 3-fold pilot at
           $0.59/h, pre-spend $1.45, cap $5 -> "COST GATE REFUSED:
           projected_total $5.52 = pre_spend $1.45 + measured $0.00 (0
           completed pods, 0s) + remaining $4.07 (3 folds x 7200s
           [worst_case_timeout]) ... vs --max-total-usd $5.00", exit 4,
           verdict persisted to the run sidecar.
           best-known?:   yes — worst-case bound uses the fold timeout
           (the hardest upper bound Modal enforces); measured basis uses
           the run's own recorded pod_facts, no invented estimates.
           scope:         ENGINEERING (spend-control gate); no model/IC
           claim.
           Tests: `make test` with umbrella venv python + sibling-src
           PYTHONPATH -> 430 passed, 8 skipped, 0 failed (424+8 baseline
           at origin/main 0e140c5 + 6 new, no regressions; 4 existing
           execute-mode tests updated to carry the now-required cap
           flags).
NEXT:      (1) model#82 prereg round 2: freeze the exact capped commands
           (phase-1 pilot / --print-cost-projection / phase-2 remainder,
           all with --max-total-usd 20 --rate-usd-per-hour <frozen> and
           the documented --pre-spend-usd); (2) the overhead default 0.15
           is a documented estimate — the prereg may freeze a different
           measured value explicitly.
