# Frozen-input-bundle guard enforced through sim_driver (model#79 round-3)

STATUS:    delivered
WHAT:      New `wf_gate/input_bundle_guard.py` — a generalized port of the
           model#79 amendment-1 checker (`verify_g4_input_bundle` v1) —
           plus its ENFORCED wiring into `wf_gate/sim_driver.py`, the
           actual launcher of assembled 104 sims.
           Guard module:
           - `verify_input_bundle(bundle_dir, target_root,
             frozen_root_digest) -> list[str]` returns "VOID ..." mismatch
             lines (empty = OK). Checks: (1) sha256 of the bundle's
             `MANIFEST.sha256` equals the frozen root digest (mismatch
             SHORT-CIRCUITS — an untrusted manifest voids per-file
             results); (2) every manifest-listed file exists under the
             target with a matching sha256; (3) bidirectional membership —
             any target file inside a covered group but absent from the
             manifest is `VOID extra file not in manifest`.
           - Covered groups are BUNDLE-DERIVED, not hardcoded (the v1
             checker's fixed COVERED_GROUPS list generalized away): for
             every manifest relpath, its parent-directory path truncated
             to the top 2 path levels (`data/ohlcv/AAPL.parquet` ->
             `data/ohlcv`; `models/m.bin` -> `models`); root-level entries
             contribute no group. A manifest listing ANY file under a
             two-level prefix claims that ENTIRE prefix. The v1 separate
             derived-config check is subsumed (it is a listed file).
           - Manifest format `"sha256  size  relpath"`; meta rows
             `MANIFEST.sha256`/`ROOT_DIGEST` excluded. Deterministic,
             stdlib-only, read-only, no network. CLI: `python -m
             renquant_backtesting.wf_gate.input_bundle_guard <bundle>
             <target> --frozen-root <hex>` -> exit 0 "VERIFY OK: ..." or
             exit 4 with every VOID line + "PREFLIGHT FAILED: N
             mismatch(es)".
           sim_driver enforcement (through the launched command):
           - New optional flags `--input-bundle <dir>` +
             `--input-bundle-root <sha256>`; one without the other is an
             argparse error. Flags absent = byte-identical legacy
             behavior (guard import is lazy inside the enabled path).
           - PRECONDITION: guard runs before ANY config/data loading or
             scoring; mismatches print all VOID lines +
             "INPUT BUNDLE PREFLIGHT FAILED: N mismatch(es)" and exit 4
             — `run_backtest` is never reached.
           - POSTCONDITION: after the sim completes AND all requested
             outputs (equity/trade/report, golden comparison) are
             written, the guard runs AGAIN; mismatch = "INPUT BUNDLE
             POST-RUN FAILED (execution-time input mutation): N
             mismatch(es)" + exit 6, distinct from 4 so wrappers can tell
             "never ran" from "ran on mutated inputs" (round-2 P1: a
             refetch/re-copy during execution reintroduces an unfrozen
             data source while the initial check stays valid).
           - Both verdicts (OK lines included) go to STDOUT so the
             launching wrapper's tee captures them verbatim.
WHY/DIR:   codex round-3 CHANGES_REQUESTED on renquant-model#79 (G4 XGB
           rerun prereg amendment 1): the checker under
           `renquant-model/doc/research/evidence` "violates the
           multi-repo boundary. ... the actual launcher is
           `renquant_backtesting.wf_gate.sim_driver` ... 'The launcher
           MUST invoke this exact file' is a manual convention, not an
           enforced precondition, which is precisely how batch 1 froze an
           invocation that had not been exercised. ... move the checker
           plus its mandatory before/after-seed enforcement into the repo
           that owns the driver ..., expose it through the launched
           command, and advance/freeze that runtime pin." And: "This is
           not a request to put model-training logic downstream: it is
           run-integrity plumbing alongside the sim driver, where it can
           actually prevent an unverified execution."
EVIDENCE:  artifact:      `wf_gate/input_bundle_guard.py` +
           `wf_gate/sim_driver.py` (this repo, this PR).
           prod or exp:   experiment — feature branch, not yet merged or
           pinned into any G4 rerun / daily-run surface.
           existing data: no prior ENFORCED checker existed at the
           sim-driver layer; the only precedent is v1
           (`verify_g4_input_bundle`) living under
           `renquant-model/doc/research/evidence`, which codex round-3
           flagged as an unenforced manual convention (see WHY/DIR) —
           grepping this repo's history shows no earlier guard module.
           best-known?:   yes for this repo — supersedes v1 as the
           enforced variant; v1 is referenced only as the port source,
           not as a live evidence path going forward.
           scope:         this is an ENGINEERING correctness claim (test
           suite pass/fail + CLI smoke), not a model/IC/Sharpe number —
           `PYTHONPATH=<common src>:<base-data src>:<artifacts src>:
           <pipeline src>:src /Users/renhao/git/github/RenQuant/.venv/bin/
           python -m pytest -q` (full repo suite) -> 406 passed, 8
           skipped, 0 failed (389 baseline + 17 new, no regressions).
           New `tests/wf_gate/test_input_bundle_guard.py` (12 tests):
           clean-ok / missing / mutated digest / extra-in-covered-group /
           extra-outside-groups-ignored / extra-in-sibling-subdir-of-
           group-flagged / bad-root-digest short-circuit / meta-row
           exclusion / missing manifest / covered-group derivation rule /
           CLI exit 0 / CLI exit 4 with all VOID lines. New
           `tests/wf_gate/test_sim_driver_input_bundle.py` (5 tests, fake
           `sim.runner` capture pattern from
           `test_sim_driver_seed_plumb.py`): preflight mismatch exits 4
           with `run_backtest` NEVER called; post-run mutation (fake
           run_backtest mutates a covered file) exits 6 AFTER exactly one
           `run_backtest` call; clean bundle echoes both OK verdicts;
           lone `--input-bundle` or lone `--input-bundle-root` is an
           argparse error (exit 2). CLI smoke on a real tiny bundle:
           exit 0 VERIFY OK, then exit 4 with digest-mismatch + extra-
           file VOID lines after tampering.
NEXT:      (1) model#79 amendment references THIS module + merged revision
           as the mandatory checker and replaces the evidence-script
           convention; (2) the G4 rerun worktree advances/freezes its
           backtesting pin to the revision carrying the guard (currently
           frozen at #78); (3) a fresh end-to-end enforced smoke —
           `sim_driver --input-bundle <bundle> --input-bundle-root
           de72ca...62df8` against the frozen G4 bundle — before seed 101
           (the seed-999 smoke predates the enforcement path and cannot
           prove it).
