# CLAUDE.md

Canonical operating model:
https://github.com/hallovorld/RenQuant/blob/main/doc/arch/subrepo-operating-model.md

Local repo map: `RENQUANT_REPOS.md`.

Branch policy: `main` is the stable interface consumed by other repos and
automation. Experiments, optimizations, and large upgrades happen on feature
branches, then merge back only after tests and integration checks pass.

## Repo Role

`renquant-backtesting` owns simulation, LEAN assembly, walk-forward validation,
and decision forensics.

## Hard Boundaries

- Sim/live parity is mandatory: tests should compare shared pipeline contracts,
  not parallel hand-written logic.
- Consume data/model manifests; do not invent local source paths silently.
- Do not store live broker credentials or submit broker orders.
- Do not train production models here.
- Large validation-method changes use a feature branch.
- Do not delete or empty the source umbrella repo at
  `/Users/renhao/git/github/RenQuant`.

## Required Evidence

Backtest output should include benchmark comparison, trade-level attribution,
decision trace, gross/tax/net decomposition, and config/data/model fingerprints.

## Workflow

```bash
make test
make doctor
```
