# Progress: Modal-scaled walk-forward PatchTST re-score (GOAL-2 AC2/AC3)

Date: 2026-07-21
PR: renquant-backtesting `feat/wf-patchtst-modal-rescore` (base
`fix/wf-patchtst-invoke-model-module` / PR #74)
Reviewer: Codex (haorensjtu-dev)
Goal: GOAL-2 (fresh PatchTST) AC2/AC3 — produce a FRESH, provenance-stamped
2nd-expert PatchTST corpus for the GOAL-4 ensemble by running the walk-forward
training driver on Modal cloud GPUs, one fold per pod, in parallel.

## What changed (single durable record)

New subpackage `src/renquant_backtesting/wf_gate/modal/` that runs the reviewed
#74 walk-forward PatchTST driver on Modal cloud GPUs, one fold per pod:

- **`executor.py`** (modal-free) — the driver side: the single-source `IMAGE_SPEC`
  + `image_spec_fingerprint()`, the run-level `recipe_id` (mirrors the WF-gate
  `recipe_match` `sha256:<16hex>` convention), fold-window computation (delegated
  to the #74 `compute_retrain_dates`), staged-subset selection, code-bundle +
  data staging to a Modal Volume, dispatch orchestration, artifact collection,
  manifest assembly (via the reviewed `walk_forward.manifest.write_manifest`),
  the provenance sidecar, a `modal_readiness()` precheck, and the CLI.
- **`app.py`** — the module-scope `modal.App` + the `@app.function` GPU worker
  `train_fold_remote`. The worker sets up the Volume-staged assembly and calls
  the #74 driver's `train_one_cutoff` for ONE cutoff, then returns the fold's
  `.pt` (gzip+base64), calibrator JSON, metadata sidecar, the manifest-entry
  dict, and per-pod provenance. Image literals re-declared here to match
  `IMAGE_SPEC` (a test asserts lockstep).
- **`tests/test_modal_wf_patchtst.py`** — 14 unit tests, no cloud calls (a fake
  `modal` is injected into `sys.modules`). Covers image-spec lockstep,
  gpu/timeout/retries decoration-time baking, recipe-id determinism, the 43-fold
  target window, staged selection, dispatch fan-out + collection + manifest +
  provenance, partial-failure handling, and the readiness precheck.

## Architecture — why each piece is where it is

This follows the multi-repo / pipeline architecture (RENQUANT_REPOS.md,
`RenQuant/doc/arch/multirepo-sop.md`):

- **Model-training internals stay in `renquant-model`.** The Modal worker never
  reimplements training. It runs the #74 driver's `train_one_cutoff`, which shells
  out to `python -m renquant_model_patchtst.hf_trainer` /
  `renquant_model_patchtst.fit_calibrator`. No training code is added or copied
  into backtesting or a Modal file.
- **The WF driver stays in `renquant-backtesting`.** The per-fold unit of work is
  the reviewed driver (PR #74, incl. the `0744d14` fail-closed subrepo-assembly
  hardening). The Modal layer *wraps* it — it does not fork the fold logic.
- **The Modal orchestration is homed in `renquant-backtesting`**, next to the rest
  of the WF-gate infra (`wf_gate/`), because the walk-forward gate is
  backtesting's subject. It is deliberately NOT in `renquant-orchestrator`:
  orchestrator consumes backtesting, never the reverse, so putting a
  WF-training cloud path there would invert the dependency. The orchestrator's
  `cloud/` sweep executor is a *different workload* (cloud backtests of the daily
  kernel); this is the WF gate's own cloud *training* path. The two-file split
  (modal-free executor + module-scope app) and the Volume-staged code-bundle +
  `IMAGE_SPEC`-fingerprint + fake-modal test strategy mirror that proven pattern
  **without importing it** (no orchestrator dependency edge is introduced).

## Provenance (the AC2/AC3 stamps)

Each run writes, alongside the standard WF manifest, a
`<manifest>.provenance.json` sidecar carrying:

- `provenance_schema_version` (`1.0`), `recipe_id`, and the full `recipe`.
- Per-fold `effective_train_cutoff_date` (read from each pod's model metadata
  sidecar `training_contract`) + `trained_date` + artifact/calibrator URIs.
- `modal`: `image_spec_sha256`, `gpu`, `volume_name`, `volume_commit_id`, and the
  per-repo git HEADs of the staged code bundle.
- Per-pod facts (`worker_id`, `code_image_id`, `elapsed_seconds`, `device`,
  `result_checksum`) and any `failed_folds`.

The standard manifest itself is written by the reviewed
`walk_forward.manifest.write_manifest` (unchanged), which validates the leakage
invariant (`trained_date >= cutoff_date`, `effective <= cutoff`) — so the fresh
corpus is gate-consumable and its entries carry `effective_train_cutoff_date`.

## Config / plan

Target = the XGB expert's sim history: **43 folds, 21-day cadence, 2023-10-02 →
2026-03-02**, dataset `data/transformer_v4_wl200_clean.parquet` (full wl200
universe), calibrator raw-label panel
`data/alpha158_291_fundamental_dataset_rawlabel.parquet`. `--staged N` runs the N
most-recent folds first for a directional read before committing to the full 43.
The calibrator leg (`fit_calibrator`) runs by default (the staged 2-fold CPU
proof in #74 had skipped it).

## Run status (this session)

Modal infrastructure was validated with bounded smoke tests:

- **Auth + fan-out dispatch: CONFIRMED.** `~/.modal.toml` (profile
  `renhao-overflow`) authenticates; a trivial `app.run()` + `.map([1,2,3])`
  dispatched 3 real pods (task ids returned).
- **GPU grant: CONFIRMED.** A `gpu="T4"` pod returned `Tesla T4, 15360 MiB` from
  `nvidia-smi`.

A real staged single-fold run (`--staged 1 --gpu T4 --execute`) against an
isolated scratch repo-root (data symlinked read-only; artifacts written to
scratch, never the live umbrella or a committed corpus) exercises the full path
(bundle → Volume upload of the two panels → GPU image build → fold train +
calibrate → collect → manifest + provenance). See the PR body for the observed
outcome of that run. The full 43-fold corpus is a follow-up dispatch (heavier:
~1.2 GB panel upload + 43 GPU pods) once the staged read is reviewed.

## Guardrails honored

- No writes to production paths / the live umbrella tree — the real run uses an
  isolated scratch root with symlinked read-only inputs.
- The #74 driver's fail-closed subrepo assembly is respected: the worker sets
  `RENQUANT_SUBREPO_ROOT` to the Volume-staged bundle so all training imports are
  pinned to that single assembly.
- No branch-protection bypass; PR carries this progress doc; not self-merged.
