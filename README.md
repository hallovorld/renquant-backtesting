# renquant-backtesting

Backtesting, LEAN assembly, simulation, and decision-forensics repository for
RenQuant.

Operating model: https://github.com/hallovorld/RenQuant/blob/main/doc/arch/subrepo-operating-model.md

Repository map: [RENQUANT_REPOS.md](RENQUANT_REPOS.md)

Local automation:

```bash
make test
make doctor
make latest-report
```

This repo validates decision quality using the same pipeline contracts as live
runtime. It does not own live broker credentials or model training
implementation.

## Runtime Parity

Simulation must reuse runtime decision tasks where practical. The
`simulate_panel_scoring_decisions()` adapter runs the shared `renquant-pipeline`
panel-scoring contract for a simulation bar, including feature-contract checks,
`blocked_by`, decision trace rows, and attributed order intents.

## Pipeline Rule

Backtest and simulation workflows are `renquant-common` Task/Job/Pipeline
chains.

## Latest Run Dashboard

`make latest-report` writes [docs/latest-run.md](docs/latest-run.md) plus SVG
charts under `docs/latest-run-assets/`. The generator scans local report roots
and the sibling umbrella strategy artifacts for the newest JSON with
walk-forward, simulation, or equity metrics.

## Initial Split Source

`hallovorld/RenQuant` commit
`8f3e08d8d1ae1e402a78f4815efb59e3c7c66aa8`.

## Local Test

```bash
make test
```
