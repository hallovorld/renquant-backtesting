# 2026-08-04 — clf corpus/recipe evidence relocated here per orch#718's ruling

STATUS:    relocation of an UNREPRODUCED HISTORICAL measurement
WHAT:      orch PR #718 was closed out-of-scope with the destination
           "clf 语料/recipe 比对器 → renquant-model + renquant-backtesting".
           The archive lands corpus-side (this repo owns the folds and the
           _recipe_projection criterion used as the match definition);
           doc/evidence/2026-08-01-clf-corpus-recipe-match/ carries the
           byte-verbatim corpus (claims doc, clf_match.json, run.log,
           evaluator, guard test), a provenance manifest (source OID
           21ece5db + per-file sha256), the unreproduced-historical label
           with the honest inventory (fingerprint prefixes + machine
           paths, live growing corpus, machine-local runner kept out of
           CI), and the placement note inviting a model-side POINTER
           rather than duplication.
WHY/DIR:   headline preserved: 0 of 85 folds (2026-08-01 snapshot)
           matched the certified clf recipe by the gate's own criterion —
           the GOAL-6 "no out-of-sample corpus for the certified recipe"
           anchor was CORRECT, settled by criterion not folder name.
           Relocate-don't-close + the #778/#202 archive discipline
           (OID + hash manifest + no production-behavior wording), baked
           in from round 1 this time.
EVIDENCE:  manifest hashes recorded before any source-branch deletion;
           source branch protected until this merges.
NEXT:      after merge: delete goal6/wf-corpus-capability-census +
           worktree (the last non-GOAL-4 stranded branch).
