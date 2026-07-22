# Wire the AC7 freshness/coverage gate into the WF paths — GOAL-5   (PR #77)

STATUS:    delivered
WHAT:      Wires the canonical `renquant_common.training_freshness.assess_training_panel_freshness`
           contract (shipped in the paired renquant-common PR #34) into BOTH WF
           trainers, fail-closed BEFORE dispatching folds when the training panel
           does not cover the window the folds need:
           - PatchTST: `train_walkforward_patchtst.py` (`required_through_date`,
             `resolve_dataset_path`, `assert_training_panel_fresh`, called in
             `main()` after the dry-run early-return) + the Modal executor
             (`_assert_panel_fresh_or_report`, run on the LOCAL panel before
             Volume staging, exit code 2 + printed reasons on breach) — the
             executor needs its own check since it runs `train_one_cutoff`
             directly, never the driver's `main()`.
           - XGB/GBDT: `train_walkforward_panel.py` mirrors the same
             `data_end_for_cutoff` / `required_through_date` /
             `resolve_dataset_path` / `assert_training_panel_fresh` shape
             (label `fwd_60d_excess`, horizon 60 BDay, confirmed
             `train_production_model.py:60`). No Modal mirror — the XGB WF
             path trains via local subprocess only.
           Same 4 CLI flags on both (`--min-tickers-per-day 20`,
           `--min-rows 0`, `--max-gap-days 5`, `--max-staleness-days` off by
           default); COVERAGE is always enforced, no blanket bypass.
WHY/DIR:   GOAL-5 AC7 (training-pipeline reliability track). Each fold slices
           the panel with `date < data_end`; the per-fold trainer only
           rejected an EMPTY post-cutoff slice, so a stale-but-nonempty
           parquet that stops short of `max(data_end)` silently trained the
           most-recent folds on a truncated window. This closes that gap at
           dispatch time for both trainers, using the one shared contract
           (no drift between PatchTST and XGB checks).
EVIDENCE:  n/a (WF-gate wiring + unit tests only, no model/data claim).
           `PYTHONPATH=<renquant-common src>:<renquant-backtesting src> python3 -m pytest
           tests/wf_gate/test_train_walkforward_freshness_gate.py
           tests/wf_gate/test_train_walkforward_panel_freshness_gate.py`
           -> 13 passed (7 PatchTST + 6 XGB), all on small fixture parquets
           (never the 400MB production panel); existing driver + modal
           executor suites unaffected (13 + 35 pass, unchanged).
NEXT:      AC8 (auto-correct / panel refresh) and AC9 (retry/self-heal) are
           separate follow-ups. Depends on renquant-common PR #34 (the
           contract) merging first.
