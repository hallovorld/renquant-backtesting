# Progress: AC-1 sidecar 176-column consumer evidence (wf_gate)

Date: 2026-07-18
Scope: test-only companion to the renquant-base-data RFC
`doc/design/2026-07-18-rawlabel-sidecar-sentiment-reconciliation.md` (AC-1)
and its evidence appendix (base-data PR carries the inventory + the decisive
sanity-contract scan result).

## What this PR adds

- `tests/wf_gate/test_sidecar_176_contract.py` — executable evidence for the
  ONE consumer whose sidecar disposition a textual sweep cannot prove:
  `wf_gate/runner.py::_load_sanity_panel` reads the served
  `alpha158_291_fundamental_dataset_rawlabel.parquet` with a bare
  `pd.read_parquet` and resolves columns dynamically from the artifact's
  sanity contract (`feature_cols`). Pinned behaviors:
  - sentiment-free contract (169 features) → DIRECT path at 176 columns;
  - prod-shape contract (172 features INCLUDING the 3 sentiment columns,
    no `training_contract.dataset` — the live `panel-ltr.alpha158_fund.json`
    shape) → direct path at today's 179-column serving, but FLIPS to the
    supplement/merge path (`feature_panel_merge: True`, sentiment supplied
    by the training panel) at 176 columns;
  - all sanity/calibrator label columns survive the 176-column contract.
- `tests/wf_gate/rawlabel_sidecar_columns_176.json` — embedded export of
  base-data `RAWLABEL_SIDECAR_COLUMNS` (main `b72dd92`); the drift guard for
  every embedded copy is base-data
  `tests/test_rawlabel_sidecar_schema_export.py`.

## What this PR does NOT do

No behavior change, no migration, no served-file mutation. The (x)/(y)
disposition decision on the flip is made in the base-data appendix review —
the scan there found the ACTIVE prod scorer and 98 other active/candidate
contracts name the sentiment columns, so the flip case is the live
population, not an edge case.
