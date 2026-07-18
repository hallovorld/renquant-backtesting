"""AC-1 executable consumer evidence: ``_load_sanity_panel`` vs the 176-col sidecar.

Companion to renquant-base-data
``doc/design/2026-07-18-rawlabel-sidecar-sentiment-reconciliation.md`` (AC-1)
and its evidence appendix. The WF gate reads the served
``alpha158_291_fundamental_dataset_rawlabel.parquet`` with a bare
``pd.read_parquet`` and resolves the needed columns DYNAMICALLY from the
candidate artifact's sanity contract (``feature_cols``), so no textual sweep
can prove its 176-column disposition — these tests pin it executably:

- a contract that does NOT name the sentiment columns takes the DIRECT path
  against a 176-column sidecar (``feature_panel_merge: False``);
- a contract that DOES name them (the live prod XGB shape: 172 features
  including sentiment, no ``training_contract.dataset``) takes the direct
  path against today's 179-column serving but FLIPS to the supplement/merge
  path (``feature_panel_merge: True``) against the migrated 176-column file
  — the exact provenance-semantics change AC-1 (x)/(y) is about.

Fixture provenance: ``rawlabel_sidecar_columns_176.json`` is an export of
renquant-base-data ``rawlabel_sidecar.RAWLABEL_SIDECAR_COLUMNS`` at main
``b72dd92``; base-data ``tests/test_rawlabel_sidecar_schema_export.py`` is
the drift guard for every embedded copy (this file is named there).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from renquant_backtesting.wf_gate import runner as wf_runner

SIDECAR_COLUMNS = json.loads(
    (Path(__file__).parent / "rawlabel_sidecar_columns_176.json").read_text()
)
SENTIMENT_COLS = ["sentiment_pos_share", "mean_sentiment", "n_articles_log"]
KEYS = ["ticker", "date"]
NON_FEATURES = set(
    KEYS + ["split_label", "fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess",
            "fwd_60d_excess_raw"]
)
#: The 169 feature columns the sidecar carries (alpha158 + fund family).
SIDECAR_FEATURES = [c for c in SIDECAR_COLUMNS if c not in NON_FEATURES]
#: The live prod XGB sanity-contract shape: 172 features INCLUDING sentiment
#: (verified against artifacts/prod/panel-ltr.alpha158_fund.json, 2026-07-18).
PROD_LIKE_CONTRACT = SIDECAR_FEATURES + SENTIMENT_COLS


def _sidecar_frame(n_dates: int = 6) -> pd.DataFrame:
    """A tiny sidecar frame with the EXACT 176-column contract."""
    assert len(SIDECAR_COLUMNS) == 176
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    rows = [(t, d) for t in ("AAA", "BBB") for d in dates]
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({
        "ticker": pd.array([t for t, _ in rows], dtype="string"),
        "date": [d for _, d in rows],
    })
    for col in SIDECAR_COLUMNS:
        if col in KEYS:
            continue
        if col == "split_label":
            frame[col] = pd.array(["train"] * len(rows), dtype="string")
        else:
            frame[col] = rng.normal(size=len(rows))
    return frame.loc[:, SIDECAR_COLUMNS]


def _write_panels(tmp_path, *, sidecar_cols_extra=(), with_training_panel=False):
    data = tmp_path / "data"
    data.mkdir()
    sidecar = _sidecar_frame()
    rng = np.random.default_rng(11)
    for col in sidecar_cols_extra:
        sidecar[col] = rng.normal(size=len(sidecar))
    sidecar.to_parquet(data / "alpha158_291_fundamental_dataset_rawlabel.parquet")
    if with_training_panel:
        train = sidecar[KEYS].copy()
        for col in SENTIMENT_COLS:
            train[col] = rng.normal(size=len(train))
        train.to_parquet(data / "alpha158_291_fundamental_dataset.parquet")
    return sidecar


def test_direct_path_at_176_when_contract_names_no_sentiment(tmp_path, monkeypatch):
    _write_panels(tmp_path)
    monkeypatch.setattr(wf_runner, "REPO", tmp_path)

    panel, meta = wf_runner._load_sanity_panel(SIDECAR_FEATURES, "fwd_60d_excess")

    assert meta["feature_panel_merge"] is False
    assert "alpha158_291_fundamental_dataset_rawlabel.parquet" in meta["sanity_feature_panel"]
    assert len(SIDECAR_FEATURES) == 169
    assert not set(SENTIMENT_COLS) & set(panel.columns)


def test_prod_shape_contract_takes_direct_path_at_todays_179(tmp_path, monkeypatch):
    """BEFORE-migration behavior: served file carries sentiment -> direct path."""
    _write_panels(tmp_path, sidecar_cols_extra=SENTIMENT_COLS)
    monkeypatch.setattr(wf_runner, "REPO", tmp_path)

    panel, meta = wf_runner._load_sanity_panel(PROD_LIKE_CONTRACT, "fwd_60d_excess")

    assert len(PROD_LIKE_CONTRACT) == 172
    assert meta["feature_panel_merge"] is False
    assert "alpha158_291_fundamental_dataset_rawlabel.parquet" in meta["sanity_feature_panel"]


def test_prod_shape_contract_flips_to_merge_path_at_176(tmp_path, monkeypatch):
    """AFTER-migration behavior: the silent direct->merge provenance flip.

    The run still completes (the training panel supplies the three columns)
    but ``feature_panel_merge`` flips True and the sentiment features are
    now sourced from a DIFFERENT file than the rest — the exact semantics
    change the RFC's AC-1 requires disposing of explicitly, affecting every
    active/candidate contract that names the sentiment columns.
    """
    _write_panels(tmp_path, with_training_panel=True)
    monkeypatch.setattr(wf_runner, "REPO", tmp_path)

    panel, meta = wf_runner._load_sanity_panel(PROD_LIKE_CONTRACT, "fwd_60d_excess")

    assert meta["feature_panel_merge"] is True
    assert sorted(meta["feature_cols_supplied_by_feature_panel"]) == sorted(SENTIMENT_COLS)
    assert meta["supplement_only_missing"] is True
    assert "alpha158_291_fundamental_dataset.parquet" in meta["sanity_feature_panel"]
    assert "alpha158_291_fundamental_dataset_rawlabel.parquet" in meta["sanity_label_panel"]
    for col in SENTIMENT_COLS:
        assert panel[col].notna().all()


def test_sanity_labels_survive_at_176():
    """The z-scored labels + raw label the sanity/calibrator paths need all
    remain in the 176-column contract (only the 3 sentiment columns go)."""
    for label in ("fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess", "fwd_60d_excess_raw"):
        assert label in SIDECAR_COLUMNS
    assert not set(SENTIMENT_COLS) & set(SIDECAR_COLUMNS)
