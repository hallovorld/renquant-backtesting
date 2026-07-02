"""End-to-end CLI test for --dump-predictions (#59 review point 5).

Drives ``analyze_manifest_sanity_placebo.main()`` through the real argv path
with only the heavy scoring inputs stubbed (artifact payload, sanity panel,
manifest scorer, regime chain), and proves:

  * the dump corresponds EXACTLY to the scored val index — the fake scorer
    returns a shuffled strict subset of the panel rows, and every dumped
    (date, name, score) must match it one-for-one;
  * the regime join cardinality is one-to-one (no row multiplication, one
    regime per date);
  * the sidecar manifest + canonical content hash verify after reload;
  * canonical production output paths are refused BEFORE any scoring runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

import renquant_backtesting.analysis.analyze_manifest_sanity_placebo as amsp
from renquant_backtesting.analysis.pick_table import (
    canonical_table_content_hash,
    sha256_file,
    verify_pick_table,
)

LABEL = "fwd_60d_excess"
N_DATES = 15
TICKERS = [f"T{i}" for i in range(8)]


def _make_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=N_DATES)
    rows = []
    for d in dates:
        for j, t in enumerate(TICKERS):
            rows.append({
                "date": d,
                "ticker": t,
                "f1": float(j) + d.day * 0.1,
                "f2": float(j) * 2.0,
                LABEL: 0.001 * j - 0.002 * (d.day % 3),
            })
    return pd.DataFrame(rows)


def _mu_value(ticker: str, date: pd.Timestamp) -> float:
    """Deterministic, per-date-distinct fake score, reconstructible in asserts."""
    return float(int(ticker[1:])) + float(pd.Timestamp(date).day) * 0.01


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Patch the four heavy inputs; record exactly what the scorer emitted."""
    panel = _make_panel()
    scored: dict = {}

    def fake_load_artifact_payload(path):
        return {
            "kind": "panel_ltr_xgboost",
            "feature_cols": ["f1", "f2"],
            "label_col": LABEL,
            "lookahead_days": 60,
        }

    def fake_load_sanity_panel(feat_cols, label, dataset_path=None):
        assert feat_cols == ["f1", "f2"] and label == LABEL
        return panel.copy(), {"panel_source": "synthetic"}

    def fake_score_manifest_sanity(val, feat_cols, manifest_path,
                                   candidate_artifact_path, candidate_artifact,
                                   panel_history=None):
        # shuffled STRICT SUBSET: drops 3 rows and randomizes index order, so
        # positional (mis)alignment or silent row loss cannot pass the asserts
        keep = val.sample(frac=1.0, random_state=7).index[:-3]
        sub = val.loc[keep]
        mu = pd.Series(
            [_mu_value(t, d) for t, d in zip(sub["ticker"], sub["date"])],
            index=keep,
        )
        scored["mu"] = mu
        scored["rows"] = {
            (pd.Timestamp(d).normalize(), t)
            for d, t in zip(sub["date"], sub["ticker"])
        }
        return mu, {"scorer": "synthetic"}

    def fake_build_regime_series(dates, *, strategy_dir=None):
        uniq = sorted({pd.Timestamp(d).normalize() for d in dates})
        regimes = ["BULL_CALM", "BEAR", "CHOPPY", "BULL_VOLATILE"]
        return pd.DataFrame({
            "date": uniq,
            "regime": [regimes[i % len(regimes)] for i in range(len(uniq))],
            "confidence": [0.9] * len(uniq),
            "source": ["gmm"] * len(uniq),
        })

    monkeypatch.setattr(amsp, "_load_artifact_payload", fake_load_artifact_payload)
    monkeypatch.setattr(amsp, "_load_sanity_panel", fake_load_sanity_panel)
    monkeypatch.setattr(amsp, "_score_manifest_sanity", fake_score_manifest_sanity)
    monkeypatch.setattr(amsp, "build_regime_series", fake_build_regime_series)

    artifact = tmp_path / "candidate_artifact.json"
    artifact.write_text(json.dumps({"kind": "panel_ltr_xgboost"}))
    manifest = tmp_path / "walkforward_manifest.json"
    manifest.write_text(json.dumps({"retrains": [{"artifact_uri": "x"}]}))
    return {
        "tmp_path": tmp_path,
        "artifact": artifact,
        "manifest": manifest,
        "scored": scored,
        "monkeypatch": monkeypatch,
    }


def _run_cli(harness, dump_path: Path, extra: list[str] | None = None) -> Path:
    tmp_path = harness["tmp_path"]
    out_dir = tmp_path / "out"
    harness["monkeypatch"].setattr(sys, "argv", [
        "analyze_manifest_sanity_placebo",
        "--artifact", str(harness["artifact"]),
        "--manifest", str(harness["manifest"]),
        "--label", LABEL,
        "--output-dir", str(out_dir),
        "--dump-predictions", str(dump_path),
        *(extra or []),
    ])
    amsp.main()
    return out_dir


def test_cli_dump_matches_scored_val_index_exactly(harness):
    dump = harness["tmp_path"] / "exp" / "pick_table.parquet"
    out_dir = _run_cli(harness, dump)

    table = pd.read_parquet(dump)
    mu = harness["scored"]["mu"]

    # exact correspondence with the scored val index: same row count, same
    # (date, name) set, and every score attached to the RIGHT (date, name)
    assert len(table) == len(mu) == 2 * len(TICKERS) - 3
    dumped = {(pd.Timestamp(d), n) for d, n in zip(table["date"], table["name"])}
    assert dumped == harness["scored"]["rows"]
    for row in table.itertuples(index=False):
        assert row.score == _mu_value(row.name, row.date)

    # regime join cardinality: one-to-one per date — no row multiplication,
    # exactly one regime value per date, no missing regimes
    assert not table.duplicated(subset=["date", "name"]).any()
    per_date = table.groupby("date")["regime"].nunique(dropna=False)
    assert (per_date == 1).all()
    assert table["regime"].notna().all()

    # the diagnostic result JSON records the dump + reload verification
    result = json.loads((out_dir / "candidate_artifact.json").read_text())
    assert result["dump_predictions"]["parquet"] == str(dump)
    assert result["dump_predictions"]["verified_after_reload"] is True


def test_cli_sidecar_manifest_and_hash_verify_after_reload(harness):
    dump = harness["tmp_path"] / "exp" / "pick_table.parquet"
    _run_cli(harness, dump)

    sidecar_path = dump.parent / "pick_table.manifest.json"
    assert sidecar_path.exists()
    sidecar = json.loads(sidecar_path.read_text())

    # independent reload + recompute must match the stamped content hash
    table = pd.read_parquet(dump)
    assert (
        canonical_table_content_hash(table, label=LABEL)
        == sidecar["output"]["output_content_sha256"]
    )
    verified = verify_pick_table(dump)  # default sidecar discovery
    assert verified["content_verified"] and verified["counts_verified"]
    assert verified["parquet_transport_match"] is True

    # provenance: input hashes are real content hashes of the input files
    recipe = sidecar["recipe"]
    assert recipe["label"] == LABEL
    assert recipe["manifest_input_sha256"] == sha256_file(harness["manifest"])
    assert recipe["reference_artifact_sha256"] == sha256_file(harness["artifact"])
    assert len(recipe["generator_sha256"]) == 64
    assert recipe["val_start"] <= recipe["val_end"]
    assert recipe["val_cut"] < recipe["val_start"]

    # temporal semantics: realized labels stamped, research-only
    ts = sidecar["temporal_semantics"]
    assert ts["research_only"] is True
    assert ts["label_lookahead_days"] == 60


def test_cli_refuses_canonical_data_tree_before_scoring(harness):
    calls = {"n": 0}
    original = amsp._load_artifact_payload

    def counting_loader(path):
        calls["n"] += 1
        return original(path)

    harness["monkeypatch"].setattr(amsp, "_load_artifact_payload", counting_loader)
    dump = harness["tmp_path"] / "data" / "pick_table.parquet"
    with pytest.raises(ValueError, match="refusing"):
        _run_cli(harness, dump)
    assert calls["n"] == 0  # refused before any input was even loaded
    assert not dump.exists()


def test_cli_allows_data_exp_research_path(harness):
    dump = harness["tmp_path"] / "data" / "exp" / "pick_table.parquet"
    _run_cli(harness, dump)
    assert dump.exists()
    assert (dump.parent / "pick_table.manifest.json").exists()


def test_cli_override_flag_bypasses_research_only_guard(harness):
    dump = harness["tmp_path"] / "data" / "pick_table.parquet"
    _run_cli(harness, dump, extra=["--allow-production-path"])
    assert dump.exists()
    assert (dump.parent / "pick_table.manifest.json").exists()
