# Latest Simulation Run

Generated: `2026-05-31 01:21:21Z`
Source: `../RenQuant/backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json`
Detected format: `wf_gate`
Selection quality score: `610`

## Selection Notes

- Newer lower-information artifact ignored: `../RenQuant/backtesting/renquant_104/artifacts/diagnostics/wf_trade_traces/20260530T220958Z/2025-04-01_to_2026-03-28.equity.json` (no trades, no finite Sharpe, no trade sidecar counts).

## Scoreboard

![Summary](latest-run-assets/summary.svg)

| Metric | Value |
|---|---:|
| `passed` | true |
| `diagnostic_only` | false |
| `wf_3cut_sharpe_mean` | 0.646 |
| `spy_sharpe_mean` | 1.081 |
| `strategy_minus_spy_sharpe_mean` | -0.435 |
| `wf_3cut_apy_mean` | 4.54% |
| `spy_apy_mean` | 16.94% |
| `strategy_minus_spy_apy_mean` | -12.40% |
| `n_positive_cuts` | 3.000 |
| `n_cuts_beat_spy_sharpe` | 0.000 |
| `n_cuts_beat_spy_apy` | 0.000 |
| `real_ic` | 0.035 |
| `sanity_placebo_ic` | 0.040 |
| `trade_contract_passed` | true |
| `trade_monotonicity_passed` | false |
| `alpha_economics_passed` | true |
| `run_at` | 2026-05-30T16:36:38.364731 |
| `sanity_regime_ic_passed` | false |
| `sanity_shuffled_ic` | 0.004 |
| `wf_3cut_sharpe_std` | 0.749 |
| `wf_reason` | PASS: absolute Sharpe floor met and SPY benchmark met; SPY mean Sharpe +1.081, ΔSharpe -0.435, beat SPY Sharpe 0/3, beat SPY APY 0/3; benchmark-lag regimes=['HIGH_CALM', 'LOW_SPIKED'] |

## Walk-Forward Cuts

![Cut performance](latest-run-assets/cuts.svg)

| Cut | Sharpe | SPY Sharpe | APY | SPY APY | Buys | Sells |
|---|---:|---:|---:|---:|---:|---:|
| 2024-01-02 to 2024-12-31 | 1.501 | 1.778 | 10.62% | 24.11% | 8.000 | 4.000 |
| 2024-07-01 to 2025-06-30 | 0.334 | 0.715 | 2.60% | 13.47% | 13.000 | 11.000 |
| 2025-04-01 to 2026-03-28 | 0.103 | 0.749 | 0.41% | 13.26% | 12.000 | 12.000 |

## Regime View

![Regime performance](latest-run-assets/regimes.svg)

| Regime | Sharpe | SPY Sharpe | APY | SPY APY |
|---|---:|---:|---:|---:|
| HIGH_CALM | 0.802 | 1.264 | 5.52% | 18.68% |
| LOW_SPIKED | 0.334 | 0.715 | 2.60% | 13.47% |

## Trade Diagnostics

![Trade diagnostics](latest-run-assets/trades.svg)

| Group | Count |
|---|---:|
| `trade_buy_source_counts_total.JointPortfolioQPJob` | 33.000 |
| `trade_sell_exit_reason_counts_total.qp_close` | 1.000 |
| `trade_sell_exit_reason_counts_total.qp_sell` | 5.000 |
| `trade_sell_exit_reason_counts_total.single_day_loss` | 3.000 |
| `trade_sell_exit_reason_counts_total.stop_loss` | 5.000 |
| `trade_sell_exit_reason_counts_total.trailing_stop` | 13.000 |
| `trade_buy_regime_counts_total.BULL_CALM` | 33.000 |
| `trade_sell_regime_counts_total.BEAR` | 5.000 |
| `trade_sell_regime_counts_total.BULL_CALM` | 11.000 |
| `trade_sell_regime_counts_total.BULL_VOLATILE` | 1.000 |
| `trade_sell_regime_counts_total.CHOPPY` | 10.000 |

_Refresh with `make latest-report`._
