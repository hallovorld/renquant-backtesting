# 2026-08-04 — Stage-2 scoring stamp WIRED (operator sign-off executed)

STATUS:    the reviewed wiring change bt#100's severability note promised
WHAT:      the operator signed stage 2 at 2026-08-04 ~10:20 PT ("GOAL-6
           Stage-2 签字，go"; durable record on #94). runner.py gains
           _attempt_stage2_stamp — a never-raise setup envelope that
           locates the committed run-001 extension bundle in the PINNED
           model checkout (env-overridable for tests), reads the DECLARED
           content pin from RUN_CLAIM.json (never recomputed from the
           manifest bytes), loads the SAME sanity panel the battery uses
           (same feat_cols/label/dataset derivation), and calls bt#100's
           attempt_lineage_scoring_stamp. The gate output carries
           lineage_stage2 as a SIBLING stamp beside lineage_stage1.
           Within the signed scope and nothing more: stamps cross the
           vintage seam only as the module's two separately-pooled
           segments; ADMISSION IS BYTE-IDENTICAL — no pass/fail path
           consults the stamp (the replacement guard test asserts no
           branch on lineage_stage2 anywhere in the runner).
GUARD SWAP: bt#100's runner-free source guard
           (test_runner_carries_NO_reference_to_stage2_until_signoff) is
           deleted CONSCIOUSLY in this change, exactly as its own
           docstring prescribed, and replaced by
           test_runner_wires_stage2_as_a_sibling_stamp_only (wiring
           present + RUN_CLAIM pin source + no admission consumption).
ROLLBACK:  revert this PR — the stamp disappears, nothing else changes
           (the signed rollback clause).
EVIDENCE:  lineage_stage2 suite 20 passed; runner/lineage selection 71
           passed; full suite 624 passed with the SAME 2 machine-local
           umbrella-byte-equivalence failures as clean main (control run
           performed; none introduced).
NEXT:      first scheduled gate run stamps lineage_stage2 for real; the
           design's later "Stage 2 — conjunction" (admission gating)
           remains a separate operator-authorized transition.
