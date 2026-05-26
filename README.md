# renquant-backtesting

Backtesting, LEAN assembly, simulation, and decision-forensics repository for
RenQuant.

This repo validates decision quality using the same pipeline contracts as live
runtime. It does not own live broker credentials or model training
implementation.

## Pipeline Rule

Backtest and simulation workflows are `renquant-common` Task/Job/Pipeline
chains.

## Initial Split Source

`hallovorld/RenQuant` commit
`8f3e08d8d1ae1e402a78f4815efb59e3c7c66aa8`.

## Local Test

```bash
PYTHONPATH=../renquant-common/src:src python -m pytest -q
```
