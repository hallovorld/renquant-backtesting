# The static eval window ignores the artifact it is out-of-sample FOR   (PR #87)

STATUS:    delivered  (opt-in; default behaviour byte-identical)
WHAT:      Extracts the static sanity eval-window derivation into
           `derive_static_eval_start()`, adds an opt-in `artifact_cutoff` mode that
           derives the window from the artifact's own declared cutoff, and records
           BOTH candidate values in `sanity_meta` on every run so the A/B is measured
           for free. Adds `common_eval_start()` for the comparability hazard the new
           mode introduces. Default = the historical fixed-80% cut, unchanged.
WHY/DIR:   #84's remaining half, and the last thing between GOAL-6 and a breadth
           retrain that can actually be evaluated. The window was cut at a fixed 80%
           of the panel's dates **with no reference to the artifact being scored**,
           so the OOS contract refuses any artifact whose cutoff sits later than that
           fraction no matter how much usable panel tail remains.
EVIDENCE:  §1 — the measured A/B, the default-equivalence proof, and a limitation
           that changes the sequencing.
NEXT:      §3. This change alone does NOT unblock today's production artifacts, and
           the reason is a producer-side gap, not this one.

## §1 EVIDENCE

Panel `2014-01-02 … 2026-05-01`, 3217 distinct business days, `lookahead_days = 60`
`[VERIFIED — derive_static_eval_start on pd.bdate_range, this session]`:

| artifact cutoff | `safe_last_label` | fixed-80% verdict | `artifact_cutoff` start | eval dates |
|---|---|---|---|---|
| 2023-08-22 | 2023-11-14 | admits (start 2023-11-15) | 2023-11-15 | 643 |
| 2024-06-03 | 2024-08-26 | **REFUSES** | 2024-08-27 | 439 |
| 2025-06-02 | 2025-08-25 | **REFUSES** | 2025-08-26 | 179 |
| 2026-02-27 | 2026-05-22 | REFUSES | **no window** | 0 |

The fixed rule's start is **2023-11-15 for every one of them**, because it never looks
at the artifact. So the latest cutoff it can admit is **2023-08-22**, which discards
**703 business days ≈ 2.79 years** of otherwise-usable training data, and the gap
widens every day the panel grows
`[DERIVED — np.busday_count(2023-08-22, 2026-05-01) = 703; 703/252 = 2.79]`.

The last row is not a defect: a cutoff whose 60-day forward window ends past the
panel's last date genuinely has no out-of-sample dates, and inventing some would be
leakage. Both modes correctly return nothing, with a reason recorded.

### The default does not move

`derive_static_eval_start` in default mode reproduces the replaced expression exactly
for panel sizes 2, 5, 10, 47, 100, 251, 1205 and 3217
`[VERIFIED — tests/test_eval_window_derivation.py::test_default_reproduces_the_historical_expression]`,
and passing an artifact does not change the default answer. This matters more than it
sounds: the window decides which dates are scored, so it decides the IC, so it decides
which fold walk-forward selection picks. A silent change here would move the gate.
An unrecognised value of `RQ_WF_EVAL_WINDOW_MODE` falls back to the historical mode
rather than selecting a third behaviour.

### The comparability hazard, handled rather than mentioned

Under `artifact_cutoff`, two artifacts with different cutoffs get different windows —
and comparing arms on different date samples is exactly the era confound that voided a
study on this programme (lag-0 IC by score-date quartile ran
`+0.0493 / +0.0032 / +0.0672 / +0.0043`
`[VERIFIED — prior work, 2026-07-29 PatchTST closure retraction]`). `common_eval_start()`
takes the **latest** of the arms' starts — the only choice that keeps every arm
out-of-sample *and* on the same rows — and returns `None` if any arm has no window,
because a comparison missing an arm is not a comparison and falling back to the
survivor's window would score a candidate against nothing. Measured: arms at
2024-08-27 and 2025-08-26 resolve to a common 2025-08-26.

## §2 A hypothesis I formed and disproved before shipping it

`_effective_artifact_cutoff` reads six keys at the artifact **root** and looks at
neither `training_contract` nor `metadata`. Since orch#620 nests the contract under
`metadata` (forced by common#38) and `retrain_patchtst.py` reads
`effective_train_cutoff_date` *inside* a contract dict, I expected a two-level miss —
the cutoff being stamped somewhere the resolver cannot see.

**That is not what the artifacts show.** On the live production and shadow artifacts:
no cutoff key at the root, no `training_contract` at the root, none under `metadata`,
only wall-clock `trained_date` (2026-05-18 and 2026-06-25 respectively)
`[VERIFIED — json inspection of RenQuant/data/panel-ltr-prod-alpha158-fund-fwd60d.json
and .../shadow_analyst/panel-ltr-shadow-baseline-noan-fwd60d.json, read-only]`.
So the resolver's `None` is **correct**, and its refusal message — *"trained_date is
wall-clock metadata and cannot prove OOS label separation"* — is accurate.

## §3 What this does NOT unblock, and the sequencing that follows

**Today's production artifact still cannot be statically evaluated**, and this PR
cannot fix that. The binding constraint is not the window: it is that the artifact
declares no cutoff at all, so `artifact_cutoff` mode has nothing to derive from. A
test pins that case with the real symptom in its assertion so it is not rediscovered.

The order therefore is: a producer stamps the effective cutoff on the artifact
(orch#620's direction, not yet reflected in the deployed artifacts — both were trained
before it merged), **then** this mode makes the window satisfiable, **then** a breadth
retrain can be evaluated against its own panel. Flipping this mode first would change
nothing for the artifact that matters, which is exactly why it ships opt-in with the
A/B recorded rather than switched on.

Also still out of scope, per #84: the config-reference choice.

## §4 Live-surface impact

None. Default mode is the historical one, proven equivalent by test; the opt-in is an
environment variable that nothing in the run surface sets. No thresholds move, no
decision rule changes. `sanity_meta` gains descriptive keys only.
