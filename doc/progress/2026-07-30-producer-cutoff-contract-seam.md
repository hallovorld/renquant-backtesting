# The seam between the trainer that stamps a cutoff and the gate that reads it   (PR pending)

STATUS:    delivered
WHAT:      Tests the composition of three separately-merged changes end to end, and
           closes a **fail-open** the seam test found: a NUMERIC `effective_train_cutoff_date`
           resolved to a valid-looking cutoff near the epoch.
WHY/DIR:   GOAL-6. Static evaluation of a candidate needs orch#620 (the trainer stamps
           the cutoff at the artifact's top level), #86 (static sanity resolves the
           contract wherever it is stamped) and #87 (the window derives from that
           cutoff). Each was tested in its own repo; **the seam between them was not**,
           and this programme's expensive defects live exactly there.
EVIDENCE:  §1.
NEXT:      The deployed artifacts predate orch#620 and still carry no cutoff, so today's
           production artifact remains statically unevaluable. That is a deployment
           question, not a code one.

## §1 EVIDENCE — a fail-open in the guard whose job is proving label separation

`_effective_artifact_cutoff` did `pd.Timestamp(value)` on whatever the field held.
**`pd.Timestamp(-1)` is `1969-12-31T23:59:59.999999999` and `pd.Timestamp(0)` is
`1970-01-01`** `[VERIFIED — asserted in test_the_epoch_hazard_is_real_and_not_hypothetical]`
— pandas reads a bare integer as nanoseconds since the epoch.

So a malformed **integer** stamp did not fail. It resolved to a plausible `Timestamp`
near 1970, which makes `safe_last_label` ~1970, which means **every `eval_start` after
1970 satisfies the OOS contract**. A garbage stamp bought *admission* rather than a
refusal, in the one guard whose entire purpose is proving that an artifact's labels do
not overlap its evaluation window.

Found only because the seam test fed the resolver deliberately malformed input. Neither
repo's own tests covered it: the producer tests that it writes an ISO string, and the
consumer tested the happy path.

**Fixed:** numeric values (including `bool`, which is an `int` subclass) are refused
outright, `NaT` is refused, and a date expressed as a date — `str`, `date`, `datetime`,
`Timestamp` — still resolves. Seven numeric variants and six malformed ones are pinned,
each with the negative case proving the refusal comes from the value's type and not from
the resolver having become unconditionally strict.

## §2 The seam itself

`test_a_stamped_artifact_flows_all_the_way_to_a_PASSING_oos_contract` walks the whole
chain: producer field name → `_effective_artifact_cutoff` → `derive_static_eval_start`
in cutoff mode → `_validate_static_sanity_oos_contract` returning `passed: True`. Its
paired negative removes the stamp and asserts the refusal returns **with its real
reason** — *"trained_date is wall-clock metadata"* — not a generic failure.

The producer's field name is **hardcoded on purpose**. Importing the orchestrator here
would make this repo depend on a sibling's internals, which is the boundary violation
orch#623's registry exists to prevent. If the producer renames the field this test must
fail — a silent rename is exactly the seam breaking.

## §3 Suite

| tree | result |
|---|---|
| `origin/main` @ 15006d2, separate worktree | 3 failed, 464 passed, 8 skipped, 242 warnings in 10.35s |
| this branch | 3 failed, 487 passed, 8 skipped, 242 warnings in 10.36s |

`[VERIFIED — python3 -m pytest -q in both worktrees, sibling checkouts on PYTHONPATH]`

## §4 Live-surface impact

The resolver change is strictly a **tightening**: values it used to accept and now
refuses were all producing a near-epoch cutoff, i.e. admitting artifacts it should have
refused. No currently deployed artifact carries this field at all, so no production
verdict changes.
