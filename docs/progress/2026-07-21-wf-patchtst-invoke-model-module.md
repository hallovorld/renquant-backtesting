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
- Pin the subprocess `PYTHONPATH` to the required `<repo>/src` trees of a
  SINGLE subrepo assembly (see Round 2 for the fail-closed hardening).
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
back), the calibrator panel args, and repo-root-driven artifact/manifest paths.

## Round 2 (codex review — pinned-assembly correctness)

Codex CHANGES_REQUESTED (sound): the first cut's subprocess PYTHONPATH could
silently import arbitrary developer checkouts — a `~/git/github` fallback plus
an ad-hoc two-root sibling scan — so a full WF run could derive artifacts from
branches outside the pinned assembly.

1. **Single pinned assembly, fail closed.** New `resolve_subrepo_root()`
   honors `$RENQUANT_SUBREPO_ROOT` (the standard injection point) and otherwise
   defaults to the assembly THIS driver was loaded from — the parent of the
   renquant-backtesting checkout (`.subrepo_runtime/repos` in the pinned
   runtime). No `~/git/github` fallback, no sibling globbing. `subprocess_env`
   pins `PYTHONPATH` to `required_subrepo_src_paths()`, which RAISES if the
   assembly is missing any required repo (`renquant-model`, `renquant-common`,
   `renquant-base-data`, `renquant-artifacts`, `renquant-pipeline`) rather than
   letting the import fall through to another checkout.
2. **Pinned-assembly subprocess import smoke test** — a fresh interpreter with
   the driver's `subprocess_env` imports both `renquant_model_patchtst.hf_trainer`
   and `.fit_calibrator` (proves the argv the driver launches resolves against
   the pinned assembly). Plus resolver tests: env honored / no home fallback,
   and fail-closed on an incomplete assembly.
3. **Calibrator-leg fold test** (`test_calibrator_leg_produces_artifacts_and_
   provenance`, opt-in via `$RENQUANT_WF_TEST_DATASET` + `$RENQUANT_WF_TEST_RAW_
   LABEL`) runs ONE fold WITHOUT `--skip-calibrators` and asserts the production
   calibration/provenance path: per-fold model `.pt` + model sidecar +
   calibrator sidecar, and a manifest entry carrying BOTH `calibrator_uri` and
   the model sidecar's `trained_date` / `effective_train_cutoff_date`.

Verified: 8/8 in the file pass (calibrator-leg run ~24 s on a CPU subset). A
2-fold `--epochs 2` run WITH the calibrator leg produced, per fold,
`hf_patchtst_all_seed44_model.pt` + `.metadata.json` + `hf_patchtst-calibration.json`,
and a 2/2 manifest with `calibrator_uri` set on every entry.
