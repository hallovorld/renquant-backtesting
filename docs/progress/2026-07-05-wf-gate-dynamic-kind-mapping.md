# fix(wf-gate): dynamic kind→config mapping survives lineup swaps

DATE: 2026-07-05
PR: #69

## What shipped (round 1)

Replaced the hardcoded `PROD_REFERENCE_BY_KIND` constant with
`_resolve_prod_reference_by_kind()`, which scans `strategy_config.json` and
`strategy_config.shadow.json` and maps scorer kind → filename from each
config's own declared `ranking.panel_scoring.kind` — so a primary/shadow
lineup swap (e.g. the 06-23 operator reversal that put XGB back on primary)
no longer requires a code change to `select_prod_reference_for_candidate()`.

## Round 2 (codex review)

STATUS: fixed
WHAT: `select_prod_reference_for_candidate()` was fixed in round 1, but the
new selector was never actually called anywhere in this file.
`wf_config_builder.main()` still took `--prod-config` (default
`strategy_config.json`) and used that path directly for both derivation and
the parity check — so a caller invoking the CLI without already knowing to
pass the kind-matched config would still compare a candidate against the
wrong reference after a lineup swap, the exact regression this PR claims to
fix.
WHY-DIR: codex correctly identified this as dead code from the caller's
perspective — the fix to the function was real, but the fix to the *system*
(the CLI a human/scheduler actually invokes) hadn't landed. The two other
callers of `select_prod_reference_for_candidate` in this repo
(`runner.py::main()`, `pipelines.py::prod_strategy_config_path()`) already
wired it correctly; only this module's own `main()` had not been updated —
likely because those two share one call site region while `wf_config_builder`
was edited standalone.
EVIDENCE:
- `main()` now: loads the candidate artifact's declared `kind` (via
  `renquant_backtesting.wf_gate.artifact_loader.load_artifact_payload`, the
  same canonical loader `runner.py` uses — not a new/duplicated reader) when
  `--candidate-artifact` is given, and calls
  `select_prod_reference_for_candidate()` for the default (no
  `--prod-config`) path — mirroring the exact pattern already proven correct
  in `runner.py:3177-3188`.
- `--prod-config` is retained as an explicit override, but if a candidate kind
  is also known, the override's own declared kind is validated against the
  candidate and fails closed (`SystemExit`) on mismatch — the same fail-closed
  posture `select_prod_reference_for_candidate`'s own `RENQUANT_STRATEGY_CONFIG`
  env-override path already uses. An explicit override still cannot smuggle a
  wrong reference past parity.
- `main()` gained an optional `argv` parameter (matching the convention
  already used by several other `wf_gate` scripts, e.g.
  `check_active_scorer.py`) so it can be driven end-to-end from tests without
  subprocessing.
- Added `test_main_survives_swapped_lineup_without_explicit_prod_config`:
  candidate kind=xgb, primary config kind=hf_patchtst, shadow config kind=xgb
  — drives the REAL `main()` entrypoint with no `--prod-config` flag and
  asserts the derived config's kind is `xgb` (the shadow reference), not
  `hf_patchtst` (the primary). Added
  `test_main_explicit_prod_config_mismatch_fails_closed` for the override
  validation. Both confirmed to fail against the pre-fix code (`TypeError:
  main() takes 0 positional arguments but 1 was given`, since `main()` had no
  `argv` parameter pre-fix — proving the tests exercise the actual fix, not a
  pre-existing capability).
- Full `wf_gate` suite: 185/185 passed. Full repo suite: 327 passed, 6 skipped
  (pre-existing, unrelated).
NEXT: none — companion umbrella PR (renquant-orchestrator#452 /
`fix/wf-promote-dynamic-gbdt-config`) tracks the paired umbrella-side fix.
