# Source Map From Monorepo

Initial source commit:
`8f3e08d8d1ae1e402a78f4815efb59e3c7c66aa8`.

Backtesting code should be ported in reviewed slices from:

- `backtesting/renquant_104/main.py`
- `backtesting/renquant_101/`
- `backtesting/renquant_102/`
- `backtesting/renquant_103/`
- `scripts/export_lean*.py`
- `scripts/analyze_backtest.py`
- walk-forward simulation and trade-forensics scripts

Data files from `backtesting/data/` must be represented by manifests or DVC/LFS
pointers, not copied into normal Git.

## Lift progress (2026-05-27, branch feature/lift-bt-forensics)

Lifted (functional-lift, copy-not-move, tested, boundary-clean):
- `forensics/` — risk_metrics, sim_smoke, trade_score_diagnostics (from kernel/).
- `forensics/metrics/` — deflated_sharpe (DSR), pbo (CSCV), perf_summary
  (compute_perf_triple), block_bootstrap, hac_se (verbatim leaf package).
- `lean/export.py` — `build_daily_lines` LEAN deci-cent daily format core.

Remaining (consumers — need base-data/pipeline wiring, not clean leaves):
- LEAN orchestration `export_symbol` (umbrella-path-coupled → rewire to base-data
  manifests) + `export_lean_watchlist` (reads strategy configs + fetch_ohlcv).
- The sim adapter execution (drives the pinned renquant-pipeline via the
  `simulation.py` `runner` contract) + `analyze_backtest`.
