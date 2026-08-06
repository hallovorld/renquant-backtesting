# GOAL-6: the Stage-2 lane now scores AGAINST something   (PR #109)

STATUS:   delivered — code + tests complete; behaviour-invariance check run
          before and after. Not merged (Codex review pending).

WHAT:     `runner.py`'s Stage-2 call site now builds `labels_by_date` from
          **the panel the lane already scored on** — the same frame, the
          same label column the artifact declares — and passes it into
          `summarize_lineage_scores`. No second data source is introduced,
          so none can disagree with the first. New helper
          `labels_by_date_from_panel` (`lineage_stage2.py`) does the
          building; `summarize_lineage_scores` itself still takes labels
          from its caller and never derives them — the helper is *for* the
          caller, not a replacement for the summarizer's contract.

WHY/DIR:  MEASURED 2026-08-04: the Stage-2 lineage stamp scored **124 of 125
          windows** and carried **`label_summary: null`**. Not because the
          summary was broken — bt#107 built it and bt#108 made its absence
          explicit — but because **the gate's call site passed no
          `labels_by_date`**. A lane that ranks candidates and cannot say
          whether its scores relate to outcomes ranks them on nothing; the
          machinery existed, the wire did not. This PR is that wire.

          The Stage-2 stamp's whole promise is that it cannot break
          admission, so `labels_by_date_from_panel` never raises —
          including on a panel that raises when indexed. Every way it can
          fail is its own stated reason (no panel supplied / panel empty /
          column absent / every label null / any other exception), each a
          distinct message. "Absent" and "empty" are kept apart on purpose:
          one catch-all would read as "the caller did not ask for labels",
          and after this change that is false. When the builder comes back
          empty, the call site replaces the stamp's
          `label_summary_absent_because` with "the caller HAS a label
          contract but could not build it: …" rather than letting it
          inherit the old wording. Keys are `pd.Timestamp` because
          `summarize_lineage_scores` looks them up as
          `labels_by_date.get(pd.Timestamp(d))` — string keys would match
          nothing, every date would be skipped, and the result would be
          indistinguishable from having no labels at all; a test pins that.
          `label_source` provenance rides on the stamp either way: the
          label column, the source, and the row/date counts actually used.

EVIDENCE:
artifact:      `src/renquant_backtesting/wf_gate/runner.py` (Stage-2 call
               site, ~line 3438 — the only production call site, per
               `tests/test_stage2_label_evaluation_visibility.py:8`),
               `src/renquant_backtesting/wf_gate/lineage_stage2.py`
               (`labels_by_date_from_panel`, line 340),
               `tests/test_lineage_stage2_label_wiring.py` (new, 9 tests)
prod or exp:   WF-gate lane code (the walk-forward promotion gate's own
               scoring path, not the live order-placing pipeline). This is
               the call site `attempt_lineage_scoring_stamp(...)` runs from
               during real WF-gate evaluations.
existing data: the 124/125-windows / `label_summary: null` gap was measured
               and recorded in the prior PRs bt#107/bt#108
               (`doc/progress/2026-08-05-goal6-stage2-scored-but-never-evaluated.md`)
               against the live 2026-08-04 staging stamp
               `[VERIFIED — prior work, bt#107/bt#108]`.
best-known?:   yes — this closes the one remaining gap those two PRs
               identified but did not wire (no `labels_by_date` at the call
               site).
scope:         Stage-2 lineage scoring/label-summary path only. Admission
               is untouched: the stamp is still `unavailable`-on-anything
               and the label path adds no new raise. The regime split still
               needs `regime_by_date` from the production chain, which this
               call site does not yet supply — see NEXT.

          Tests: full suite before and after the change shows the same
          three pre-existing failures on `main` (`test_b1_lift`,
          `test_import_lift`, `test_REAL_run001_…`), and **673 → 682
          passed**, the +9 being this PR's own tests
          `[VERIFIED — this session]`.

NEXT:     The regime split (per-regime IC, not just an aggregate) still
          needs `regime_by_date` wired from the production chain into the
          same call site — this PR only wires the label, not the regime
          map. The summary's own contract already reports that absence as
          absence rather than as an empty split, so nothing downstream is
          misled in the meantime.

## Not claimed

This does not say any candidate now has evidence. It says the lane can now
**report** whether its scores track the label, per regime — which was the
prerequisite. The regime split still needs `regime_by_date` from the
production chain, which this call site does not yet supply; the summary's
own contract already reports that absence as absence rather than as an
empty split.
