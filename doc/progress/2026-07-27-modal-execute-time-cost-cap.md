# Modal execute-time hard cost cap (model#82 P0 round 2)

STATUS:    delivered (P0 round-2 review fixes applied same PR — see
           WHAT §"P0 fixes")
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
             remaining_usd   = timeout_seconds x n_to_dispatch x usd_per_sec
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

           P0 fixes (round-2 review, same PR, both addressed BEFORE
           merge — the review's two P0s were substantive, not resolved
           by a later checklist-only approval on the same commit):

           - **P0-1 (bound was an expectation, not a hard cap).** The
             original formula priced every ABSENT fold at the mean of
             already-completed pods once any pod had completed — a later
             fold can legally run up to `timeout_seconds`, so actual
             billed cost could exceed the asserted cap even on a GO
             verdict. Fixed: `remaining_usd` now ALWAYS prices absent
             folds at `timeout_seconds` (the provider-enforced ceiling —
             Modal kills any fold at the timeout, so none can bill more).
             The mean-of-completed figure survives only as an
             `info_mean_remaining_usd_estimate` field / "info-only mean
             estimate" summary suffix — informational, never part of the
             bound. `basis` is now always `"hard_timeout_bound"`.
             Worked example (operator arithmetic, pinned in
             `test_cost_gate_worked_example_25_dollar_grant`): cap $25
             (operator raised 2026-07-27, includes probe pre-spend
             $1.45), rate $0.59/h, overhead 0.15, phase-2
             `--timeout-seconds 2900` (measured fold 2384s + ~21.6%
             headroom): projected_total = 1.45 + 3 measured pods
             (~$1.35) + 40 x 2900s hard bound (~$21.86) ~= $24.66 <= $25
             -> GO; a fold that actually hits 2900s is provider-killed
             -> failed_folds non-empty -> the existing
             halt-on-failed-fold rule fires.
           - **P0-2 (budget inputs were mutable across resumes).**
             Nothing compared a resume's `--max-total-usd` /
             `--rate-usd-per-hour` / `--pre-spend-usd` / `--overhead-frac`
             against the run's first capped attempt, so a caller could
             reuse the same `--run-id` with a larger cap (or lower
             pre-spend/overhead) and bypass the original authorization.
             Fixed: `budget_contract_from_args()` freezes
             `{max_total_usd, rate_usd_per_hour, pre_spend_usd,
             overhead_frac, timeout_seconds}` into the run's provenance
             sidecar (`persist_budget_contract()`) on the FIRST capped
             dispatch or refusal. Every later capped `--execute` on that
             `--run-id` is compared field-by-field
             (`budget_contract_mismatches()`) BEFORE the cost gate runs
             and BEFORE any Modal import; a mismatch prints "BUDGET
             CONTRACT MISMATCH" naming the differing fields and exits 4
             with nothing dispatched. The frozen contract also survives
             every provenance rebuild (`collect_and_write`), including a
             no-op rebuild. Re-freezing the budget requires a new
             `--run-id`, matching the existing one-run-one-recipe pattern
             already enforced for `recipe_id`.
WHY/DIR:   The codex P0 on model#82: watching 3 pods then `modal app
           stop` is not a pre-dispatch budget guard — by the time the
           check runs, the whole fan-out has been dispatched and accrues.
           With #81 the fan-out is bounded to the selected absent folds;
           with this PR the executor itself refuses to launch a dispatch
           whose all-in projection breaches the cap, so the prereg can
           freeze exact capped commands instead of prose.
EVIDENCE:  artifact:      `src/renquant_backtesting/wf_gate/modal/executor.py`
           (`compute_cost_gate`, `budget_contract_from_args`,
           `budget_contract_mismatches`, `persist_budget_contract`,
           `persist_refused_cost_gate`, parse_args requiredness, main()
           gate) + `tests/test_modal_wf_patchtst.py` (10 new tests total:
           hard-bound math incl overhead + pre-spend + info-only mean
           estimate; hard-bound phase-1 zero-completed-pods GO + REFUSED;
           argparse requiredness incl run-id + rate; phase-1 CLI refusal
           exit 4 with NO modal import + persisted REFUSED record +
           frozen budget_contract; phase-2 CLI refusal from real pilot
           pod_facts with partition respected, proving the hard bound —
           not the 42s measured mean — drives the refusal; budget-contract
           bypass attempt (same run id, larger cap) refused before any
           modal import; budget-contract matching resume proceeds past
           the contract check to the ordinary cost-gate REFUSED; GO path
           prints + gate record rides the dispatches[] entry; the $25 /
           2900s worked example pinned to the dollar
           (`test_cost_gate_worked_example_25_dollar_grant`); the full
           five-field contract drift matrix — larger cap, LOWER
           pre-spend, overhead, timeout, rate — each refused exit 4
           before any modal import, plus matching resume passes with GO
           (`test_budget_contract_frozen_and_immutable`)).
           prod or exp:   experiment tooling; no production path written;
           zero Modal calls (fake SDK in tests; real CLI exercised only
           to the exit-4 refusal, which precedes any modal import).
           existing data: real-CLI demo re-run against a scratch
           repo-root on the frozen matrix (recipe `b4e47e2cd77af660`):
           2-fold selection at $0.59/h, pre-spend $1.45, cap $3 ->
           "COST GATE REFUSED: projected_total $4.16 = pre_spend $1.45 +
           measured $0.00 (0 completed pods, 0s) + remaining $2.71 (2
           folds x 7200s [hard per-fold timeout bound]) ... vs
           --max-total-usd $3.00"; same selection at cap $5 -> GO,
           freezes `budget_contract` into the sidecar (logged:
           `budget_contract frozen for run demo-run: {'max_total_usd':
           5.0, 'rate_usd_per_hour': 0.59, 'pre_spend_usd': 1.45,
           'overhead_frac': 0.15, 'timeout_seconds': 7200}`); a
           follow-up invocation on the SAME run id with
           `--max-total-usd 999` is refused BEFORE any modal import
           (verified: no panel/readiness/dispatch step runs) with
           "BUDGET CONTRACT MISMATCH: run 'demo-run' froze its budget
           contract on the first capped attempt; this invocation differs
           on: max_total_usd (contract=5.0, invocation=999.0)." exit 4.
           best-known?:   yes — the bound is now the provider-enforced
           per-fold ceiling (Modal kills any fold at timeout_seconds, so
           none can bill more) for EVERY absent fold, always; the
           mean-of-completed figure is reported but never load-bearing.
           Budget inputs are frozen at first capped attempt and compared
           field-by-field on every resume, closing the reuse-with-a-
           bigger-cap bypass the round-2 review identified.
           scope:         ENGINEERING (spend-control gate); no model/IC
           claim.
           Tests: sibling-src PYTHONPATH pytest ->
           `tests/test_modal_wf_patchtst.py` 60 passed, 0 failed (56
           baseline + 4 new this round; 4 existing tests updated for the
           hard-bound formula, none skipped/removed). Full suite (umbrella
           venv + sibling PYTHONPATH): 434 passed, 8 skipped, 0 failed
           (baseline 424 at origin/main 0e140c5; no regressions).
NEXT:      (1) model#82 prereg round 2: freeze the exact capped commands
           (phase-1 pilot / --print-cost-projection / phase-2 remainder,
           all with --max-total-usd 20 --rate-usd-per-hour <frozen> and
           the documented --pre-spend-usd); (2) the overhead default 0.15
           is a documented estimate — the prereg may freeze a different
           measured value explicitly.
