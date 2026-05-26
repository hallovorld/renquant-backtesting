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
