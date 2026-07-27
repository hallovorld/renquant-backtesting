# Frozen-input-bundle guard enforced through sim_driver (model#79 round-3)

STATUS:    delivered
WHAT:      New `wf_gate/input_bundle_guard.py` — a generalized port of the
           model#79 amendment-1 checker (`verify_g4_input_bundle` v1) —
           plus its ENFORCED wiring into `wf_gate/sim_driver.py`, the
           actual launcher of assembled 104 sims.
           Guard module:
           - `verify_input_bundle(bundle_dir, target_root,
             frozen_root_digest, covered_roots) -> list[str]` returns
             "VOID ..." mismatch lines (empty = OK). Checks: (1) sha256
             of the bundle's `MANIFEST.sha256` equals the frozen root
             digest (mismatch SHORT-CIRCUITS — an untrusted manifest
             voids per-file results); (2) every manifest-listed file
             exists under the target with a matching sha256, regardless
             of covered roots; (3) bidirectional membership — any target
             file inside an EXPLICIT covered root but absent from the
             manifest is `VOID extra file not in manifest`.
           - Covered roots are EXPLICIT and caller-frozen (repeatable
             `--covered-root <relpath>`, at least one required): the
             membership sweep runs ONLY over the given target-relative
             directories, recursively. Manifest entries OUTSIDE all
             covered roots (singletons) are digest-checked individually
             with no sweep of their parent. FIELD-TEST FINDING (honest
             record): the first revision derived the covered set from
             manifest relpaths (parent dirs truncated to the top 2 path
             levels). Testing against the REAL G4 bundle + worktree
             falsified that rule with 735 false VOIDs — a root-level
             singleton (`data/sec_fundamentals_daily.parquet`, parent
             `data`) claimed all of `data/`, and 3-level-deep artifact
             dirs claimed all of `backtesting/renquant_104/`, so
             unrelated code files AND the sim's own outputs
             (`data/wf_provenance/*.jsonl`, `data/sim_runs_*.db`) were
             flagged as extras: the post-run check would have failed on
             EVERY successful sim. Replaced with explicit frozen roots
             pinned in the launch command next to the root digest. The
             v1 separate derived-config check remains subsumed (it is a
             listed file).
           - Manifest format `"sha256  size  relpath"`; meta rows
             `MANIFEST.sha256`/`ROOT_DIGEST` excluded. Deterministic,
             stdlib-only, read-only, no network. CLI: `python -m
             renquant_backtesting.wf_gate.input_bundle_guard <bundle>
             <target> --frozen-root <hex> --covered-root <relpath>
             [--covered-root ...]` -> exit 0 "VERIFY OK: ..." or exit 4
             with every VOID line + "PREFLIGHT FAILED: N mismatch(es)".
           sim_driver enforcement (through the launched command):
           - New optional flags `--input-bundle <dir>` +
             `--input-bundle-root <sha256>` + repeatable
             `--input-bundle-covered-root <relpath>`; any partial
             combination (including missing covered roots) is an
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
           flagged as an unenforced manual convention (see WHY/DIR).
           This PR's own first revision (bundle-DERIVED covered roots)
           is now also superseded: the real-tree field test below
           falsified it.
           best-known?:   yes for this repo — supersedes v1 as the
           enforced variant; v1 is referenced only as the port source,
           not as a live evidence path going forward.
           scope:         this is an ENGINEERING correctness claim (test
           suite pass/fail + real-tree CLI verification), not a
           model/IC/Sharpe number.
           REAL-TREE VALIDATION (the enforcement path exercised against
           the actual frozen G4 bundle + rerun worktree, 6 covered
           roots: data/ohlcv, data/earnings_surprise,
           data/news_sentiment_alpaca, backtesting/renquant_104/
           artifacts/walkforward_gbdt_prod_recipe_v2, .../artifacts/sim/
           walkforward_calibrators, .../models):
           `VERIFY OK: 4429 files verified, membership clean,
           root=8072ca771d0cab732687efdbca929dbacae34a0b72cb26ad423ccac6
           ade8aea1` (exit 0, ~1.6s). The same invocation under the
           derived-roots revision produced 735 false VOIDs — that is the
           measurement that forced the redesign.
           `PYTHONPATH=<common src>:<base-data src>:<artifacts src>:
           <pipeline src>:src /Users/renhao/git/github/RenQuant/.venv/bin/
           python -m pytest -q` (full repo suite) -> 409 passed, 8
           skipped, 0 failed (389 baseline + 20 new, no regressions).
           New `tests/wf_gate/test_input_bundle_guard.py` (12 tests):
           clean-ok / missing / mutated digest / extra-in-covered-root /
           extras-outside-roots-ignored (incl. the sim-output false-VOID
           class) / singleton-outside-roots-still-digest-verified /
           bad-root-digest short-circuit / meta-row exclusion / missing
           manifest / CLI exit 0 / CLI exit 4 with all VOID lines / CLI
           requires >=1 --covered-root. New
           `tests/wf_gate/test_sim_driver_input_bundle.py` (8 tests, fake
           `sim.runner` capture pattern from
           `test_sim_driver_seed_plumb.py`): preflight mismatch exits 4
           with `run_backtest` NEVER called; post-run mutation (fake
           run_backtest mutates a covered file) exits 6 AFTER exactly one
           `run_backtest` call; clean bundle echoes both OK verdicts;
           sim outputs written OUTSIDE covered roots do NOT void the
           post-run check (the field-test scenario); 4 partial-flag
           combinations (incl. --input-bundle + --input-bundle-root
           without a covered root) are argparse errors (exit 2).
NEXT:      (1) model#79 amendment references THIS module + merged revision
           as the mandatory checker — pinning the frozen root digest AND
           the six covered roots in the launched command — and replaces
           the evidence-script convention; (2) the G4 rerun worktree
           advances/freezes its backtesting pin to the revision carrying
           the guard (currently frozen at #78); (3) a fresh end-to-end
           enforced smoke — `sim_driver --input-bundle <bundle>
           --input-bundle-root 8072ca...aea1 --input-bundle-covered-root
           <6 roots>` — before seed 101 (the seed-999 smoke predates the
           enforcement path and cannot prove it).
