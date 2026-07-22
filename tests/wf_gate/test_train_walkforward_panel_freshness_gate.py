"""AC7 (GOAL-5) — the XGB/GBDT walk-forward panel driver must fail-closed
BEFORE dispatching cutoffs when the training panel does not cover the window the
folds need, and proceed when it does.

Sibling of ``test_train_walkforward_freshness_gate.py`` (the PatchTST driver
gate). The XGB driver subprocesses the umbrella ``train_production_model.py``,
which only rejects an EMPTY post-cutoff slice; a stale-but-nonempty panel that
stops short of ``max(data_end)`` silently trains on a truncated window. These
tests exercise the pre-dispatch gate against small fixture parquets — never the
production ``alpha158_291_fundamental_dataset.parquet``.
"""
from __future__ import annotations

import pandas as pd
import pytest

from renquant_backtesting.wf_gate import train_walkforward_panel as twp

BDAY = pd.offsets.BDay


def _panel(dates: list[pd.Timestamp], n_tickers: int = 25) -> pd.DataFrame:
    rows = [
        {"date": d, "ticker": f"T{i:03d}"}
        for d in dates
        for i in range(n_tickers)
    ]
    return pd.DataFrame(rows)


def _args(tmp_path, dataset, **extra):
    argv = [
        "--start-date", "2024-01-01",
        "--end-date", "2024-06-01",
        "--cadence-days", "21",
        "--repo-root", str(tmp_path),
        "--dataset", str(dataset),
    ]
    for k, v in extra.items():
        argv += [f"--{k}", str(v)]
    return twp.parse_args(argv)


def _dates_and_required(args):
    dates = twp.compute_retrain_dates(
        pd.Timestamp(args.start_date), pd.Timestamp(args.end_date),
        int(args.cadence_days),
    )
    return dates, twp.required_through_date(dates, args.label)


# ── required_through_date is the MAX fold data_end ─────────────────────────

def test_required_through_date_is_max_data_end(tmp_path) -> None:
    args = _args(tmp_path, tmp_path / "ph.parquet")
    dates, required = _dates_and_required(args)
    expected = max(
        pd.Timestamp(twp.data_end_for_cutoff(c, args.label)) for c in dates
    )
    assert required == expected
    # It is the LAST cutoff's data_end (latest window end), not the first.
    assert required == pd.Timestamp(
        twp.data_end_for_cutoff(dates[-1], args.label)
    )
    # GBDT recipe horizon is 60 business days (fwd_60d_excess), matching PatchTST.
    assert twp.infer_label_lookahead_days(args.label) == 60


# ── PASS: the panel covers max(data_end) ───────────────────────────────────

def test_gate_passes_when_panel_covers(tmp_path) -> None:
    args = _args(tmp_path, tmp_path / "ph.parquet")
    dates, required = _dates_and_required(args)
    full = list(pd.bdate_range(required - BDAY(120), required + BDAY(3)))
    ds = tmp_path / "good.parquet"
    _panel(full, n_tickers=25).to_parquet(ds)
    args.dataset = str(ds)
    # Must not raise.
    twp.assert_training_panel_fresh(args, dates)


# ── FAIL-CLOSED: the panel stops short of max(data_end) ────────────────────

def test_gate_aborts_when_panel_short(tmp_path) -> None:
    args = _args(tmp_path, tmp_path / "ph.parquet")
    dates, required = _dates_and_required(args)
    short = list(pd.bdate_range(required - BDAY(120), required - BDAY(20)))
    ds = tmp_path / "short.parquet"
    _panel(short, n_tickers=25).to_parquet(ds)
    args.dataset = str(ds)
    with pytest.raises(RuntimeError, match="FAIL-CLOSED"):
        twp.assert_training_panel_fresh(args, dates)


def test_gate_aborts_on_thin_ticker_day(tmp_path) -> None:
    args = _args(tmp_path, tmp_path / "ph.parquet")
    dates, required = _dates_and_required(args)
    full = list(pd.bdate_range(required - BDAY(120), required + BDAY(3)))
    panel = _panel(full, n_tickers=25)
    thin_day = full[10]
    panel = panel[~((panel["date"] == thin_day) & (panel["ticker"] >= "T003"))]
    ds = tmp_path / "thin.parquet"
    panel.reset_index(drop=True).to_parquet(ds)
    args.dataset = str(ds)
    with pytest.raises(RuntimeError, match="min_tickers_per_day"):
        twp.assert_training_panel_fresh(args, dates)


def test_gate_missing_dataset_raises(tmp_path) -> None:
    args = _args(tmp_path, tmp_path / "does_not_exist.parquet")
    dates, _ = _dates_and_required(args)
    with pytest.raises(FileNotFoundError):
        twp.assert_training_panel_fresh(args, dates)


def test_floors_are_flaggable_off(tmp_path) -> None:
    # A thin (3-ticker) but fully-covering panel PASSES once the ticker floor
    # is set to 0 — the thresholds are CLI-flaggable, coverage stays enforced.
    args = _args(tmp_path, tmp_path / "ph.parquet",
                 **{"min-tickers-per-day": "0"})
    dates, required = _dates_and_required(args)
    full = list(pd.bdate_range(required - BDAY(120), required + BDAY(3)))
    ds = tmp_path / "thin_ok.parquet"
    _panel(full, n_tickers=3).to_parquet(ds)
    args.dataset = str(ds)
    twp.assert_training_panel_fresh(args, dates)  # no raise
