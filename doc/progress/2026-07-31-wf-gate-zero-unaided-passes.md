# The WF gate has issued zero unaided passes in 11 artifacts

**RELOCATED from `renquant-orchestrator#670`.** Codex on that PR: this evidence and its
assertions belong here, because `renquant-backtesting` owns the WF gate, the
candidate-versus-recipe admission semantics, and the referenced artifacts. The
orchestrator should consume a gate verdict, not become a second frozen-data authority
for backtesting behaviour. Agreed — the finding is about the gate, so it is filed
against the gate's owner.

**Bottom line.** The model trading the live book today was **not admitted by the gate**.
An operator override dated **2026-06-22** admitted it, over its own sanity battery's
`FAIL`. Across every `panel-ltr.alpha158_fund` artifact carrying `wf_gate_metadata`:
**11 artifacts, 2 with `passed=True`, both overrides, zero unaided passes**
`[VERIFIED — 本次实测 2026-07-31, evidence/2026-07-31-wf-gate-unaided-passes/gate_verdicts.csv]`.

| artifact | `passed` | `diagnostic_only` | admitted by |
|---|:--:|:--:|---|
| **`panel-ltr.alpha158_fund.json` (DEPLOYED)** | True | True | **operator override 2026-06-22** |
| `weekly_20260706T230931Z.staging` | True | False | **operator override 2026-07-06** |
| `weekly_20260712` … `weekly_20260730` (9) | False | False | — |

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

Filed: `renquant-backtesting#90`. Tests: 5, pinned to a frozen CSV.
