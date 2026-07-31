# The WF gate has issued zero unaided passes in 11 artifacts   (PR #91)

STATUS:    delivered
WHAT:      Relocated from `renquant-orchestrator#670` (Codex boundary finding):
           the evidence CSV and all 5 pinned tests move unchanged, only this
           doc's header records the relocation. Nothing was re-measured or
           re-worded in transit. Filed against `renquant-backtesting` because
           this repo owns the WF gate, the candidate-versus-recipe admission
           semantics, and the referenced artifacts.
WHY/DIR:   GOAL-6 — is the WF gate measuring what it was built to measure?
           Across every `panel-ltr.alpha158_fund` artifact carrying
           `wf_gate_metadata` (11 total), 2 carry `passed=True` and **both are
           operator overrides** — zero unaided passes. The artifact trading the
           live book today is one of the two overrides, admitted 2026-06-22
           over its own sanity battery's FAIL.
EVIDENCE:  artifact:      doc/research/evidence/2026-07-31-wf-gate-unaided-passes/gate_verdicts.csv
           prod or exp:   prod — audits `wf_gate_metadata` stamped on the
                          production and staging `panel-ltr.alpha158_fund`
                          artifacts already on disk; no new training run.
           existing data: census of all 11 artifacts carrying `wf_gate_metadata`
                          for this recipe: 2 rows `passed=True` (the deployed
                          artifact, override dated 2026-06-22; one staging
                          artifact, override dated 2026-07-06), 9 rows
                          `passed=False` with `override_reason` empty.
           best-known?:   n/a — this is a census of every scored artifact for
                          this recipe, not a comparison against a better-known
                          variant.
           scope:         "this is gate_verdicts.csv, prod, 11/11 alpha158_fund
                          artifacts audited: 2 passed (both operator overrides),
                          9 rejected unaided, 0 unaided passes"
           `[VERIFIED — measured 2026-07-31, gate_verdicts.csv;
           tests/test_wf_gate_unaided_passes.py, 5 tests pinned to the frozen CSV]`
NEXT:      Two findings compound and both need resolution before an unaided
           pass can be trusted: (1) all 11 artifacts admit on
           `recipe_validated=True` / `candidate_artifact_used=False`
           (`renquant-backtesting#83`'s shape — the gate scores the recipe, not
           the candidate's own booster); (2) the nine rejects are the gate
           working as specified, but every admission to date has been manual.
           Whether the gate's criteria need revision or the candidates really
           are this bad is not decidable from this evidence alone.

## The deployed artifact's own testimony

```
gate_verdict_reason : passed=false solely from skipped_required_gates=
                      [trade_monotonicity_pass_open_allowed] (diagnostic_only)
override_reason     : Operator directive 2026-06-22 ("全放宽 + 上 XGB"). Primary config
                      wf_gate already opts out benchmark/regime/sanity_regime_ic
                      (2026-05-30 operator decision accepting SPY-laggard GBDT).
                      trade-monotonicity has no pass-enabling opt-in and is OVERRIDDEN
                      by explicit operator authority.
sanity_reason       : FAIL: regime sanity IC failed: BULL_CALM,CHOPPY
wf_reason           : PASS ... Sharpe 0.70 vs SPY 1.08, beat SPY Sharpe 1/3,
                      beat SPY APY 0/3
```

Three gates opted out by config; the fourth overridden by directive; sanity says FAIL;
and the WF leg it *did* pass records **losing to SPY on 2 of 3 cuts and on APY 3 of 3**.

## Two corrections to what I published earlier tonight

1. I wrote that the deployed artifact is **"the only one that passes."** That held for
   the single enforced placebo sub-criterion I had computed
   (`placebo_ic < max(0.005, 0.5·|aligned_real_ic|)`). Its **overall** sanity verdict is
   **FAIL**, on regime sanity IC. A sub-criterion is not the battery.
2. **"Chronic reject" is the wrong frame.** The nine rejects are the gate working as
   specified. The two admissions are the exceptions. The live question is not why
   retrains get rejected — it is whether a gate with an **0-for-11 unaided pass rate** is
   measuring what it was built to measure. While every admission is manual there is no
   way to separate *"the gate is right and the candidates are bad"* from *"the gate is
   mis-specified."*

All 11 also carry `candidate_artifact_used=False` / `recipe_validated=True` — the
recipe-identity admission of `renquant-backtesting#83`. The two findings **compound**:
the gate scores a recipe rather than the candidate, and when it does say no, a human says
yes.

## Not claimed

Whether the overrides were wrong. Both carry an explicit operator directive with a stated
rationale — which is what the containment protocol asks for. This is a statement about the
**gate's** state, not the operator's decisions.
