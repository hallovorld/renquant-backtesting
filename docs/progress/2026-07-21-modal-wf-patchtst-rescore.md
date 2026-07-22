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

Real staged single-fold runs (`--staged 1 --gpu T4 --execute`) against an
isolated scratch repo-root (data symlinked read-only; artifacts to scratch,
never the live umbrella or a committed corpus) surfaced two container-side bugs
that only appear on a real dispatch — both now fixed:

1. **Container-load `ModuleNotFoundError: renquant_common`.** Modal imports the
   worker's *defining module* before any function body runs; a module under the
   `renquant_backtesting` package dragged its heavy `__init__` at load time. Fix:
   the worker is a standalone top-level module `wf_patchtst_modal_app`
   (`os + modal` import surface only); the pinned bundle is put on `sys.path`
   inside the body.
2. **Stale bundled driver.** `bundle_code` searched `repo_root.parent` first, and
   the scratch root's sibling was an old pre-#74 checkout, so the pod trained
   against the removed `scripts/patchtst_hf.py` path. Fix: bundle from the
   reviewed checkout this executor runs from (`_EXECUTOR_CHECKOUT_ROOT.parent`) +
   a fail-closed `_assert_fresh_driver()` staleness guard.

After both fixes the pod loads cleanly and runs the correct command
(`python -m renquant_model_patchtst.hf_trainer … --device cuda --save-model`) —
real GPU training. The full/larger multi-fold corpus is a `.map` fan-out over the
recent 8–12 folds first (real universe, calibrator leg on), then the full 43,
reusing the now-cached image. See the PR thread for the landed fold count.

## Guardrails honored

- No writes to production paths / the live umbrella tree — the real run uses an
  isolated scratch root with symlinked read-only inputs.
- The #74 driver's fail-closed subrepo assembly is respected: the worker sets
  `RENQUANT_SUBREPO_ROOT` to the Volume-staged bundle so all training imports are
  pinned to that single assembly.
- No branch-protection bypass; PR carries this progress doc; not self-merged.

---

## Round 2 — 2026-07-22: codex PR #76 review fixes + cost sizing

Codex requested changes on #76 (three blockers). All addressed in `executor.py`
+ tests; no model/pipeline internals touched (still a pure wrapper of the #74
driver). Blockers, each with the fix:

1. **Arbitrary-checkout contamination (was reintroduced).** `bundle_code` used a
   list of `code_roots` and *silently skipped* missing repos, and `main()` fell
   back to `~/git/github` — so an incomplete reviewed assembly leaked in an
   ambient checkout per-repo (the exact thing #74's `resolve_subrepo_root`
   removed). Fix: `bundle_code(bundle_dir, code_root, *, assembly_lock=None)` now
   takes ONE explicit pinned root; **fails closed** if any `BUNDLE_REPOS` src is
   missing, if any staged repo has no resolvable git HEAD (unpinned), or (when an
   `--assembly-lock` JSON is given) if any staged commit drifts from the reviewed
   lock. `main()` uses only `_EXECUTOR_CHECKOUT_ROOT.parent` (or `--code-root`) —
   no home fallback.
2. **Weak `volume_commit_id` provenance.** It hashed remote filenames + local
   file *sizes*, so same-size code/data shared a provenance id. Fix: the commit
   id is now a digest of every staged file's **content** (streamed SHA-256), and
   the two leakage-relevant DATA panels get explicit per-file content digests in
   `provenance.modal.data_digests`. The resolved, immutable Modal image ids the
   pods actually ran are recorded in `provenance.modal.resolved_image_ids` (a
   stronger dep lock than the spec fingerprint).
3. **Partial/unverified corpus was promotable.** `collect_and_write` wrote the
   canonical serving name `walkforward_patchtst_manifest.json` for ANY nonzero
   fold count and `main` exited 0. Fix: every run is **quarantined** under
   `artifacts/walkforward_patchtst_runs/<run_id>/` (auto `--run-id`
   `wf-pt-<recipe8>-<utc>`); the executor **refuses** to write the canonical
   serving manifest (`_assert_not_canonical_manifest`); provenance carries
   `promotion_ready` (True only when every requested fold succeeded) +
   `quarantined`; a partial run **exits nonzero**. Promotion to the serving
   manifest is a separate reviewed step.

Also fixed a latent run-killer (found while wrapping): the trainer's
`build_config_contract()` reads `renquant-strategy-104/configs/strategy_config.json`
at the END of a fit, but only `<repo>/src` was bundled → every fold would die
with `FileNotFoundError` AFTER a full train. `EXTRA_BUNDLE_SUBDIRS` now bundles
that `configs/` dir and `_assert_strategy_config` fails closed pre-dispatch. The
container path lines up: trainer's `GITHUB_ROOT` = `/data/app/repos`, so it
resolves `/data/app/repos/renquant-strategy-104/configs/strategy_config.json`.

Tests: `tests/test_modal_wf_patchtst.py` grew 14 → **28** (fail-closed bundling,
missing-repo, unpinned, lock-drift/lock-match, strategy-config assert, content-vs
-size digest, run-namespace quarantine, canonical-manifest refusal,
promotion-ready). All 28 pass; the #74 driver regression suite still green.

### Cost sizing (T4, decision input) — [VERIFIED prices / GUESS wall-time]

GPU requested = **T4** (app default + `DEFAULT_GPU`). Modal published rate
(modal.com/pricing, 2026-07): **T4 $0.000164/s** (~$0.59/hr); A10 $0.000306/s;
A100-40GB $0.000583/s. Billing is per-GPU-second summed over pods, so the `.map`
fan-out cuts wall-clock but NOT total $.

Per-fold wall-time on T4 [GUESS]: baseline ~34 min train on local MPS. The model
is tiny (d_model=64, 2 layers, seq_len 32, 5 epochs) so it is overhead-/data-bound,
not FLOP-bound → T4 ≈ MPS (adjustment ~1.0×, plausible 0.7–1.2×). Add the
default calibrator leg (~6 min) + container/torch/Volume overhead (~3 min) →
**~43 min/fold ≈ 2,580 s** base case. GPU $/fold = 2,580 × $0.000164 = **$0.42**;
+~12% CPU/mem ≈ **$0.48/fold**.

| Run | Folds | GPU-hours | Base $ (T4) | Range (0.7–1.2×) |
|-----|------:|----------:|------------:|------------------|
| **Staged** (2025-11-01→2026-03-28) | 8 | ~5.7 | **~$3.8** | ~$3–$5 |
| **Full**   (2023-10-02→2026-03-02) | 43 | ~30.8 | **~$20** | ~$14–$24 |

Image build is one-time CPU-builder work (~$0.05–0.20, then cached). **Staged-8
on T4 is well under $10.** **Full-43 on T4 is ~$18–20 — OVER $10** (even the
optimistic 0.7× case ≈ $12–14 stays ≥ $10). A10G is *more* expensive here (~1.87×
rate, only ~1.3× faster for this small model), so T4 is cost-optimal. Recommend:
run staged-8 first for a directional read; full-43 needs a separate >$10 sign-off.
Launch remains gated by the "no Modal until clear plan" rule + operator decision.
