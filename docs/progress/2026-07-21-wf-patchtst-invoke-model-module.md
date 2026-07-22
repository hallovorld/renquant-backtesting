# WF PatchTST driver: invoke the model repo as a module (unblock fresh WF training)

DATE: 2026-07-21
GOAL: GOAL-2 fresh-PatchTST — make walk-forward PatchTST training RUN from a
clean/pinned checkout so the Modal-ized full WF re-score can proceed.

## Root cause

`wf_gate/train_walkforward_patchtst.py` was bulk-copied from the umbrella
(`7decced` "Phase 1 — full sweep copy") and only had its manifest/loader
imports localized (`77493b5`). It kept the umbrella's *old* invocation
shape: `TRAIN_SCRIPT = REPO_ROOT/"scripts"/"patchtst_hf.py"` (and the
matching calibrator script). In the umbrella, `__file__.parent.parent`
resolves to the repo root where `scripts/patchtst_hf.py` exists; swept into
`src/renquant_backtesting/wf_gate/`, that same expression resolves to the
*package* dir, and neither `scripts/patchtst_hf.py` nor a `scripts/` dir has
ever existed in the renquant-backtesting checkout (no git history of one).
So every fold died at the subprocess with:

```
can't open file '.../renquant-backtesting/src/renquant_backtesting/scripts/patchtst_hf.py': No such file or directory
```

Meanwhile the training internals were refactored into **renquant-model** as
importable modules (`renquant_model_patchtst.hf_trainer`,
`renquant_model_patchtst.fit_calibrator`), and the umbrella's own copy of the
driver was already updated to `python -m renquant_model_patchtst.hf_trainer`.
The subrepo copy was left on the dead script path. A second latent blocker
sat right behind it: `REPO_ROOT/"backtesting"/"renquant_104"` and the relative
`data/*.parquet` paths also assumed the umbrella layout, so even with the
script found, the dataset and artifact root would not resolve in the subrepo.

## Fix (repo-boundary correct)

Training internals stay in renquant-model; this driver only *orchestrates*.

- Invoke the model repo **as a module**: `python -m
  renquant_model_patchtst.hf_trainer` / `... .fit_calibrator` (mirrors the
  umbrella's evolved driver and `wf_gate/runner.py`'s `-m sim_driver`
  pattern). No `scripts/*.py` path, nothing hacked into the runtime.
- Wire the subprocess `PYTHONPATH` (and the outer `sys.path`) to every
  existing sibling-repo `src` tree, discovered relative to the checkout
  (`.subrepo_runtime/repos/<repo>` in the pinned runtime, `~/git/github/<repo>`
  in a dev checkout) so `renquant_model_patchtst` imports regardless of cwd.
- Resolve the umbrella data/artifact root explicitly via
  `renquant_backtesting.repo_root.resolve_repo_root` (`--repo-root` /
  `$RENQUANT_REPO_ROOT` / cwd), matching `wf_gate/check_active_scorer.py`.
  Fixes the second blocker; artifacts + manifest land under
  `<repo-root>/backtesting/<strategy>/artifacts/…`.
- Calibrator invocation gains the module's panel args
  (`--panel/--raw-label-panel/--label-col/--min-rows`); new CLI:
  `--repo-root`, `--strategy`, `--raw-label-panel`, `--calibrator-min-rows`.
  Local `renquant_backtesting.walk_forward.{loader,manifest}` imports are
  preserved (the `77493b5` localization decision is intact).

No production caller binds this driver's CLI (only the umbrella script is
referenced as a producer label in inventory manifests), so the added/renamed
flags are back-compat-safe.

## Proof — 2-fold staged run (isolated scratch, CPU)

`python -m renquant_backtesting.wf_gate.train_walkforward_patchtst
--start-date 2023-06-01 --end-date 2023-08-15 --cadence-days 45
--repo-root <scratch> --dataset <scratch>/subset_transformer_40t_2021.parquet
--device cpu --epochs 2 --seq-len 16 --d-model 32 --n-heads 2 --n-layers 1
--skip-calibrators`

(Dataset is a 40-ticker / from-2021 subset of `transformer_v4_wl200_clean.parquet`,
built in scratch to keep the CPU smoke fast; identical schema.) Exit 0:

```
train cutoff=2023-06-01 done in 22.8s
train cutoff=2023-07-16 done in 24.3s
Wrote PatchTST manifest with 2/2 retrains -> .../artifacts/walkforward_patchtst_manifest.json
```

Per-fold artifacts written (each ~62 KB `.pt` + metadata sidecar):

- `walkforward_patchtst/2023-06-01/hf_patchtst_all_seed44_model.pt` (+ `.metadata.json`)
- `walkforward_patchtst/2023-07-16/hf_patchtst_all_seed44_model.pt` (+ `.metadata.json`)

Manifest `retrains` = 2/2, each with `cutoff_date`, `trained_date`, and
`effective_train_cutoff_date` (2023-03-08 / 2023-04-21) read back from the
model-repo sidecar contract. Outputs written only under the scratch
repo-root — no umbrella/production path touched.

Regression guard: `tests/wf_gate/test_train_walkforward_patchtst_command.py`
asserts the module invocation (and that `scripts/patchtst_hf.py` never comes
back), the calibrator panel args, the sibling-`src` PYTHONPATH wiring, and
repo-root-driven artifact/manifest paths.
