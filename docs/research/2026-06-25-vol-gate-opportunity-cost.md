# RealizedVolGate: EXPLORATORY diagnostic on the hard 60% vol cap

2026-06-25. Trigger: the 2026-06-25 daily-full no-trade — `RealizedVolGateTask` dropped
21/97 buy candidates over the 60% annualized-vol cap. Operator: high-vol is opportunity too;
raise the bar, don't freeze — but with theory + rigorous data. **This is EXPLORATORY evidence
only — NOT a config proposal.** An earlier version over-claimed a "regime-aware rule"; that
claim is withdrawn (it was based on a calendar split, not the regime label, and does not survive
proper uncertainty).

**Re-homed.** This study was first opened in `renquant-orchestrator` (PR #194). The orchestrator
must not implement model-fitting research (a hard CLAUDE.md boundary: *"Do not implement model
training internals here"*), so the experiment, its tests, and these docs live here in
`renquant-backtesting`, which owns walk-forward validation and decision forensics. Two
correctness fixes were applied during the move (§3a, §3b).

## 1. Theory (kept honest)
- **Kelly / Merton:** optimal weight `f* = μ/σ²` — vol enters sizing **continuously**; there is
  no binary admission threshold in optimal theory. A hard cap forces `f*=0` above a line.
- **Low-volatility anomaly / BAB** (Ang 2006; Baker 2011; Frazzini–Pedersen 2014): high
  idiosyncratic-vol / high-beta names earn **lower risk-adjusted** returns. This is the real
  theoretical case FOR penalising vol — but as a *continuous* penalty (which Kelly `1/σ²`
  already is), not specifically a 60% admission line.
- **Moreira–Muir (2017)** studies **portfolio-level** volatility TIMING (scale the whole book by
  1/σ_portfolio). It is **NOT** direct evidence about a **cross-sectional single-security**
  admission gate. Cited only as background on vol-scaling — explicitly NOT as support for this
  gate change.

## 2. Survivorship caveat — up front, and it dominates
The panel is **291/291 names that all survive to 2026** (zero delistings). The high-vol names
that blew up are MISSING, biasing high-vol returns UP. A raw "high-vol wins" reading therefore
**contradicts the well-replicated low-vol anomaly**, which is itself a sign the data is
survivorship-contaminated. So everything below is an **upper-bound diagnostic**, not deployable
evidence. No robustness op here removes survivorship.

## 3. Method
Monthly rebalance; top-quintile by OOS model score (pooled purged-WF **XGB proxy — NOT the live
PatchTST**; omits the Kelly μ numerator, QP, concentration caps, daily rebalance, and the live
gate ordering); weight ∝ 1/σ² (clip [.05,1.5]); forward 1-mo **excess** vs SPY from OHLCV;
turnover cost Σ|Δw|·(5bps+20bps·vol). **5 non-overlapping purged test folds; fold boundaries do
not share a date.** Vary ONLY the cap. Sliced by the **actual market regime label** (now
*asserted* uniform across names per date — §3b), with sample counts; paired block-bootstrap CIs;
TRUE-exclusion vs winsorization robustness. Pure helpers are unit-tested in
`tests/test_research_vol_gate.py`.

### 3a. Leakage fix — embargo by TRADING SESSIONS, not calendar days
The label `fwd_60d_excess` is `close.shift(-60)` = **60 trading sessions**. The original purge
subtracted `Timedelta(days=60)` = **60 calendar days**, which on a business-day index is only
~42 trading sessions — so training labels near the cutoff still reached into the test interval
(leakage that inflates OOS scores). The embargo is now counted in **trading sessions** off the
sorted unique-date index: the training cutoff is the date `EMB_SESSIONS` positions before the
test-start position, guaranteeing `train_end + 60 sessions ≤ test_start`. Empirically the fixed
purge yields exactly 60 sessions (≈84 calendar days) of separation per fold; a 60-calendar-day
purge would have left only 42–44 sessions. A regression test
(`test_purge_is_60_trading_sessions_not_calendar_days`) asserts the last training-label-end date
precedes the first test timestamp by ≥60 trading sessions (no overlap).

### 3b. Regime-uniformity assert (fail closed)
The monthly regime slice assumes the regime is a *market-level* label identical for every name on
a given date. The original code claimed this was "verified uniform" but silently took
`.groupby("date").first()`. `assert_regime_uniform_per_date` now **raises** if any date carries
more than one distinct regime label across names, before regimes are assigned. Tested both ways.

### 3c. Multiple comparisons
The six-cap bootstrap CIs below are **per-comparison** (each cap vs the 0.6 baseline), not
family-wise. The no-change conclusion is *conservative* under that (every CI already includes
zero, so no correction can manufacture significance). **But any FUTURE positive inference from a
cap sweep must pre-register the primary cap, or apply a family-wise / FDR (Benjamini–Hochberg)
correction across the caps before claiming significance.** Sweeping six caps and reporting the
best uncorrected CI would be a multiple-comparisons error.

## 4. Results (from the orchestrator run; re-run pending under the fixed purge)
> The figures below are from the original orchestrator run, which used the looser
> 60-calendar-day purge. The fixed 60-trading-session embargo *removes* leakage, which can only
> *lower* OOS scores — so it strengthens (never weakens) the inconclusive verdict. A re-run on
> the umbrella data is required before any of these point estimates is cited as evidence; it is
> not run here (slow + needs the umbrella panel). Treat the numbers as illustrative.

**Overall cap sweep (≈92 months, net of cost, excess vs SPY):**

| cap | Sharpe | annRet | maxDD | CVaR5 | median mo |
|---|---|---|---|---|---|
| **0.60 (current)** | +0.20 | +1.5% | −15.2% | −4.8% | +0.0012 |
| 0.80 | +0.65 | +4.9% | −13.1% | −3.9% | +0.0028 |
| 1.00 | +0.70 | +5.3% | −13.8% | −3.9% | +0.0037 |
| 1.50 | +0.71 | +5.4% | −14.0% | −3.8% | +0.0037 |
| ∞ | +0.71 | +5.3% | −14.0% | −3.8% | +0.0037 |

Point estimate: relaxing the cap *raises* the Sharpe and does NOT raise vol/drawdown (the 1/σ²
sizing keeps high-vol names tiny). BUT — see the uncertainty below.

**By ACTUAL regime — Sharpe by cap (n months):**

| regime | 0.6 | 0.8 | 1.0 | 1.5 | ∞ |
|---|---|---|---|---|---|
| BULL_CALM (n=42) | +0.27 | +0.44 | +0.45 | +0.45 | +0.45 |
| BULL_VOLATILE (n=47) | +0.36 | +0.78 | +0.85 | +0.83 | +0.83 |
| **BEAR (n=3)** | — unmeasurable — | | | | |

Relaxing helps in both BULL regimes; **BEAR has only 3 months → no regime-level conclusion is
possible.** (The earlier "the cap helps in the 2022 bear" was a *calendar-period* artifact, not a
regime result — withdrawn.)

**Paired block-bootstrap CI (2000 reps, 3-mo blocks) — monthly return delta vs cap 0.6
(per-comparison; see §3c):**

| comparison | Δ mean / mo | 95% CI |
|---|---|---|
| 1.0 − 0.6 | +0.0032 | **[−0.0002, +0.0080]** |
| 0.8 − 0.6 | +0.0028 | [−0.0001, +0.0073] |
| ∞ − 0.6 | +0.0032 | [−0.0003, +0.0081] |

**Every CI includes zero.** The relaxation's benefit is a positive point estimate but is **NOT
statistically significant** at 95% on this sample — and that is *before* any multiple-comparisons
correction, which would only widen the bar.

**Robustness (top-1% winner months):** true-exclude Sharpe ≈ winsorize (0.6: +0.11 vs +0.10;
1.0: +0.61 vs +0.60) — but **neither removes survivorship** (both keep only 2026 survivors).

## 5. Honest conclusion (exploratory)
- The point estimates are *consistent with* the theory that, given a downstream `1/σ²` sizer, a
  hard 60% admission cap is conservative — relaxing raised Sharpe without raising drawdown.
- BUT this is **not significant** (bootstrap CIs include 0, before FDR), BEAR is **unmeasurable**
  by regime, the panel is **survivorship-biased** (upper bound), the ranker is a **proxy** (not
  the live PatchTST in the live sizing/QP/gate stack), and the cited numbers predate the leakage
  fix. **No config change is supported by this evidence.**
- The prior "60% is the worst point" and "1.0/0.6 regime rule" claims are withdrawn.

## 6. What a real decision needs (before ANY config PR)
Re-run under the **fixed 60-trading-session purge** → pre-register per-regime hypotheses +
acceptance/risk bars and the **primary cap** (with FDR across any sweep) → re-run with **live
PatchTST** scores and the **real Kelly μ/σ², QP, concentration caps, daily rebalance, and live
gate order** → use a **point-in-time universe including delisted outcomes** → report paired
net-return, drawdown/CVaR, and turnover deltas **with uncertainty** → **shadow-test** the chosen
rule before production. Repro: `research/research_vol_gate_opportunity_cost.py`.
