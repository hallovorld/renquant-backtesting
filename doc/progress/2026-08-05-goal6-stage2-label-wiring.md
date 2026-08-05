# 2026-08-05 — GOAL-6: the Stage-2 lane now scores AGAINST something

## The gap

MEASURED 2026-08-04: the Stage-2 lineage stamp scored **124 of 125 windows** and
carried **`label_summary: null`**. Not because the summary was broken — bt#107
built it and bt#108 made its absence explicit — but because **the gate's call
site passed no `labels_by_date`**.

A lane that ranks candidates and cannot say whether its scores relate to
outcomes ranks them on nothing. The machinery existed; the wire did not.

## What lands

`runner.py`'s Stage-2 call site now builds the labels from **the panel the lane
already scored on** — the same frame, the same label column the artifact
declares. No second data source is introduced, so none can disagree with the
first. `summarize_lineage_scores` still takes labels from its caller and never
derives them; this is a helper *for* the caller.

## The helper's contract is the stamp's contract

The Stage-2 stamp's whole promise is that it **cannot break admission**, so
`labels_by_date_from_panel` **never raises** — including on a panel that raises
when indexed. Every way it can fail is its **own stated reason**:

| condition | reason |
|---|---|
| no panel supplied | `no sanity panel was supplied` |
| panel present but empty | `the sanity panel is empty` |
| column absent | `panel has no '<col>' column` |
| every label null | `no non-null '<col>' rows` |
| anything else | `<ExceptionType>: <message>` |

"Absent" and "empty" are kept apart deliberately — one catch-all would read as
*"the caller did not ask for labels"*, and after this change that is **false**.
For the same reason, when the builder comes back empty the call site **replaces**
the stamp's `label_summary_absent_because` with *"the caller HAS a label contract
but could not build it: …"* rather than letting it inherit the old wording.

The keys are `pd.Timestamp`, because `summarize_lineage_scores` looks them up as
`labels_by_date.get(pd.Timestamp(d))` — string keys would match nothing, every
date would be skipped, and the result would be **indistinguishable from having
no labels at all**. A test pins that.

`label_source` provenance rides on the stamp either way: the label column, the
source, and the row/date counts actually used.

## Behaviour invariance `[VERIFIED — this session]`

Full suite before and after the change: **the same three failures**
(`test_b1_lift`, `test_import_lift`, `test_REAL_run001_…` — all pre-existing on
`main`), and **673 → 682 passed**, the +9 being this PR's own tests. Admission
is untouched: the stamp is still `unavailable`-on-anything and the label path
adds no new raise.

## Not claimed

This does not say any candidate now has evidence. It says the lane can now
**report** whether its scores track the label, per regime — which was the
prerequisite. The regime split still needs `regime_by_date` from the production
chain, which this call site does not yet supply; the summary's own contract
already reports that absence as absence rather than as an empty split.
