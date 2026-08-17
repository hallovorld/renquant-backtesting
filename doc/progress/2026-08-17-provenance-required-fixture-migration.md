# Provenance-required cutover — migrate the runtime-parity manifest fixture

STATUS:    delivered. Test-only. No src/ change, no behaviour change, nothing
           deployed, no live path touched. Unblocks CI on main, which has been
           red for every push since 2026-08-15.

WHAT:      `tests/test_runtime_parity.py`, one manifest fixture:
           * added `"provenance": {"kind": "none"}` — the field is now REQUIRED
             on every candidate manifest;
           * `"promotion_status": "prod"` -> `"candidate"` — `kind="none"`
             cannot be combined with `prod`.
           Plus a comment recording why the fixture, not the guard, was wrong.

WHY/DIR:   `renquant-artifacts` sets `PROVENANCE_REQUIRED_AFTER = date(2026,8,15)`;
           `provenance_required()` returns True unconditionally on/after that
           date (one-way, no env override). This is a scheduled cutover that
           arrived, NOT a regression: main's last green CI is 2026-08-10, i.e.
           before the date. Every push since fails in `verify_artifact_provenance`.

           The fixture claimed `promotion_status="prod"` while carrying no
           lineage at all. The guard rejecting that is the guard working: a real
           prod artifact must resolve a canonical, registry-published lineage
           (`publication_record_digest` + registry bindings). A synthetic test
           double is not a published production artifact, so `"prod"` was a
           fiction the cutover exposed — not a capability it removed.

           Rejected alternative: keep `"prod"` and satisfy the canonical path.
           That would mean fabricating a registry publication record for a test
           double — manufacturing exactly the evidence the guard exists to
           demand. Measured (see EVIDENCE): that path is the ONLY way to keep
           `"prod"`, which is what makes dropping the claim the honest fix.

EVIDENCE:
  artifact:       tests/test_runtime_parity.py (fixture only)
  prod or exp:    neither — test fixture; no production path reads it
  existing data:  main CI history (last green 2026-08-10, before the cutover);
                  PR #112 CI run 31970614447 fails inside
                  `verify_artifact_provenance` — same root cause
  best-known?:    yes — validator behaviour measured directly, not inferred:
                    kind=none      + prod       -> REJECT ("cannot be combined")
                    kind=none      + candidate  -> PASS
                    kind=canonical + prod       -> REJECT (missing
                                                   publication_record_digest,
                                                   registry bindings)
                    kind=canonical + candidate  -> PASS
                  `promotion_status` has no consumer in
                  renquant-backtesting/src, so the test's assertions do not
                  depend on the changed value.
                  `"candidate"` is the value 8 other manifest fixtures in these
                  repos already use; the migrated one was the only `"prod"`.
  scope:          this repo's fixture only. The identical cutover breaks
                  renquant-orchestrator; that is a separate PR in that repo.

VERIFICATION:
  Run from a SIBLING worktree — `[tool.pytest.ini_options] pythonpath` uses
  `../renquant-*/src`, so a worktree outside `git/github/` cannot resolve the
  siblings and fails with unrelated ModuleNotFoundErrors.

  pre-fix  (clean origin/main): tests/test_runtime_parity.py  2 failed
  post-fix:                     tests/test_runtime_parity.py  2 passed
  post-fix  full suite:         3 failed, 716 passed, 1 skipped

  The 3 remaining failures are PRE-EXISTING on clean origin/main (verified by
  running the same three node-ids with the fix stashed: 3 failed) and are
  local-only: `test_b1_lift.py` / `test_import_lift.py` byte-diff against
  `parents[3]/"RenQuant"/...` and `pytest.skip` when the umbrella is not a
  sibling, which is the case in CI. They fail here because this machine's
  umbrella tree has drifted from the subrepo source. Out of scope; recorded so
  the number is not read as "this PR left three tests red".

NEXT:      merge unblocks main. PR #112 (APPROVED, red on exactly this failure)
           then updates from main and goes green on its own. The orchestrator
           counterpart is filed separately.
