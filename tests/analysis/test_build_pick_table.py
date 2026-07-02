"""Unit + end-to-end tests for build_pick_table (--dump-predictions, the
durable OOS pick table) and its evidence-provenance manifest.

Covers Codex's round-1 review on renquant-backtesting#59: the decile bug on
common small cross-sections, index-misalignment risk, the durable-artifact
manifest contract (ported from the already-merged RenQuant#430), the
research-only path guard, and CLI-level end-to-end provenance verification.
"""
import json

import numpy as np
import pandas as pd
import pytest

from renquant_backtesting.analysis.analyze_manifest_sanity_placebo import (
    ResearchOnlyOutputPathError,
    _decile_rank,
    _guard_research_only_output_path,
    build_pick_table,
    build_pick_table_manifest,
    canonical_pick_table_content_hash,
)


def _fixture():
    dates = pd.to_datetime(["2025-01-02"] * 10 + ["2025-01-03"] * 10)
    tickers = [f"T{i}" for i in range(10)] * 2
    val = pd.DataFrame({
        "date": dates,
        "ticker": tickers,
        "fwd_60d_excess": [0.01 * i for i in range(10)] * 2,
    })
    mu = pd.Series([float(i) for i in range(10)] + [float(9 - i) for i in range(10)],
                   index=val.index)
    regimes = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
        "regime": ["BULL_CALM", "BEAR"],
    })
    return val, mu, regimes


def test_columns_and_row_count():
    val, mu, regimes = _fixture()
    out = build_pick_table(val, mu, "fwd_60d_excess", regimes)
    assert list(out.columns) == ["date", "ticker", "fwd_60d_excess", "score",
                                 "regime", "decile_rank"]
    assert len(out) == 20


def test_decile_rank_exactly_10_distinct_is_balanced_0_through_9():
    """The bug case: exactly 10 distinct scores must populate every decile
    0..9 exactly once, not 1..9 with decile 0 absent and decile 9 doubled
    (the rank(pct=True)*10 + clip(upper=9) failure mode)."""
    val, mu, regimes = _fixture()
    out = build_pick_table(val, mu, "fwd_60d_excess", regimes)
    for d, g in out.groupby("date"):
        counts = g["decile_rank"].value_counts().sort_index()
        assert list(counts.index) == list(range(10)), (
            f"date {d}: decile buckets not 0..9, got {list(counts.index)}")
        assert (counts == 1).all(), f"date {d}: expected 1 name per decile, got {counts.to_dict()}"
        top = g.loc[g["score"].idxmax()]
        assert top["decile_rank"] == 9
    d1 = out[out["date"] == "2025-01-02"].nlargest(1, "score")["ticker"].iloc[0]
    d2 = out[out["date"] == "2025-01-03"].nlargest(1, "score")["ticker"].iloc[0]
    assert d1 != d2


def test_decile_rank_tied_scores():
    # 7 distinct values among 10 observations (with ties) -> capped at
    # n_unique=7 bins (0..6), not padded to 9; deterministic, no crash.
    #
    # Ties are broken by rank(method="first") (original array position)
    # BEFORE qcut, exactly like RenQuant#430's reference implementation —
    # this keeps bin sizes balanced/deterministic, but it does NOT guarantee
    # every tied observation lands in the same bucket (a 3-way tie split
    # across a bin boundary can land two in one bucket and the third in the
    # next). That's the real, intentional behavior being ported here, not a
    # bug: assert the deterministic per-position result exactly, not a
    # "ties always share a bucket" property the reference method doesn't
    # actually provide.
    scores = pd.Series([1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    out = _decile_rank(scores)
    assert out.min() == 0
    assert out.max() == 6
    assert len(out.unique()) == 7
    assert len(out) == 10
    assert out.tolist() == [0, 0, 1, 2, 3, 3, 4, 5, 6, 6]


def test_decile_rank_fewer_than_10_names():
    scores = pd.Series([1.0, 2.0, 3.0])
    out = _decile_rank(scores)
    assert out.min() == 0
    assert out.max() == 2  # only 3 distinct names -> 3 bins, not padded to 9
    assert len(out) == 3


def test_decile_rank_single_distinct_value():
    scores = pd.Series([5.0, 5.0, 5.0])
    out = _decile_rank(scores)
    assert (out == 0).all()


def test_decile_rank_more_than_10_names_still_10_buckets():
    scores = pd.Series([float(i) for i in range(37)])
    out = _decile_rank(scores)
    assert out.min() == 0
    assert out.max() == 9
    assert len(out.unique()) == 10


def test_regime_joined_and_nan_scores_dropped():
    val, mu, regimes = _fixture()
    mu2 = mu.copy()
    mu2.iloc[0] = float("nan")
    out = build_pick_table(val, mu2, "fwd_60d_excess", regimes)
    assert len(out) == 19
    assert set(out["regime"].unique()) == {"BULL_CALM", "BEAR"}


def test_empty_regimes_yields_none_column():
    val, mu, _ = _fixture()
    out = build_pick_table(val, mu, "fwd_60d_excess", pd.DataFrame())
    assert "regime" in out.columns
    assert out["regime"].isna().all()


def test_shuffled_mu_index_attaches_correct_scores_not_positional():
    """Regression for the .to_numpy() index-discard bug: a mu series with
    the SAME labels as val but in a DIFFERENT order must still attach each
    score to its correct ticker/date row, not whatever row happens to sit
    at the same positional offset."""
    val, mu, regimes = _fixture()
    shuffled_order = list(val.index)
    rng = np.random.RandomState(42)
    rng.shuffle(shuffled_order)
    mu_shuffled = mu.loc[shuffled_order]
    assert list(mu_shuffled.index) != list(mu.index)  # sanity: genuinely shuffled

    out_original = build_pick_table(val, mu, "fwd_60d_excess", regimes)
    out_shuffled = build_pick_table(val, mu_shuffled, "fwd_60d_excess", regimes)

    merged = out_original.merge(
        out_shuffled, on=["date", "ticker"], suffixes=("_orig", "_shuf"))
    assert len(merged) == len(out_original)
    assert np.allclose(merged["score_orig"], merged["score_shuf"]), (
        "shuffling mu's index order changed which score attached to which "
        "ticker -- index alignment is broken")


def test_missing_index_in_mu_raises():
    val, mu, regimes = _fixture()
    mu_missing = mu.drop(mu.index[0])
    with pytest.raises((ValueError, KeyError)):
        build_pick_table(val, mu_missing, "fwd_60d_excess", regimes)


def test_duplicate_index_in_mu_raises():
    val, mu, regimes = _fixture()
    mu_dup = pd.concat([mu, mu.iloc[[0]]])
    with pytest.raises((ValueError, KeyError)):
        build_pick_table(val, mu_dup, "fwd_60d_excess", regimes)


# --------------------------------------------------------------------------
# canonical_pick_table_content_hash / build_pick_table_manifest
# --------------------------------------------------------------------------


def test_content_hash_order_independent():
    val, mu, regimes = _fixture()
    table = build_pick_table(val, mu, "fwd_60d_excess", regimes)
    h1 = canonical_pick_table_content_hash(table, "fwd_60d_excess")
    h2 = canonical_pick_table_content_hash(
        table.sample(frac=1.0, random_state=7).reset_index(drop=True), "fwd_60d_excess")
    assert h1 == h2


def test_content_hash_sensitive_to_score_change():
    val, mu, regimes = _fixture()
    table = build_pick_table(val, mu, "fwd_60d_excess", regimes)
    h1 = canonical_pick_table_content_hash(table, "fwd_60d_excess")
    table2 = table.copy()
    table2.loc[table2.index[0], "score"] += 0.0001
    h2 = canonical_pick_table_content_hash(table2, "fwd_60d_excess")
    assert h1 != h2


def test_manifest_stamps_research_only_and_temporal_semantics(tmp_path):
    val, mu, regimes = _fixture()
    table = build_pick_table(val, mu, "fwd_60d_excess", regimes)
    out_path = tmp_path / "pick_table.parquet"
    table.to_parquet(out_path, index=False)
    manifest = build_pick_table_manifest(table, label="fwd_60d_excess", output_path=out_path)
    assert manifest["research_only"] is True
    assert "fwd_60d_excess" in manifest["temporal_semantics"]
    assert "REALIZED" in manifest["temporal_semantics"]["fwd_60d_excess"]
    assert "output_content_sha256" in manifest["output"]
    assert manifest["output"]["output_parquet_sha256"] is not None
    assert manifest["counts"]["n_rows"] == len(table)


def test_manifest_content_hash_matches_reload(tmp_path):
    """The manifest's output_content_sha256 must verify after a fresh
    reload of the written parquet -- round-trip integrity, not just
    'the file was written'."""
    val, mu, regimes = _fixture()
    table = build_pick_table(val, mu, "fwd_60d_excess", regimes)
    out_path = tmp_path / "pick_table.parquet"
    table.to_parquet(out_path, index=False)
    manifest = build_pick_table_manifest(table, label="fwd_60d_excess", output_path=out_path)

    reloaded = pd.read_parquet(out_path)
    reloaded_hash = canonical_pick_table_content_hash(reloaded, "fwd_60d_excess")
    assert reloaded_hash == manifest["output"]["output_content_sha256"]


# --------------------------------------------------------------------------
# research-only output-path guard
# --------------------------------------------------------------------------


def test_guard_rejects_live_path(tmp_path):
    with pytest.raises(ResearchOnlyOutputPathError):
        _guard_research_only_output_path(tmp_path / "live" / "picks.parquet")


def test_guard_rejects_production_path(tmp_path):
    with pytest.raises(ResearchOnlyOutputPathError):
        _guard_research_only_output_path(tmp_path / "production_artifacts" / "picks.parquet")


def test_guard_allows_research_path(tmp_path):
    _guard_research_only_output_path(tmp_path / "data" / "exp" / "picks.parquet")


def test_guard_bypass_flag_allows_flagged_path(tmp_path):
    _guard_research_only_output_path(
        tmp_path / "live" / "picks.parquet", allow_production=True)


# --------------------------------------------------------------------------
# CLI end-to-end
# --------------------------------------------------------------------------


def _padded_panel_and_mu():
    """A panel with 10 'padding' dates (older, excluded from the 80% val
    cutoff) followed by 2 real dates that fall in the val partition -- lets
    the CLI-level test exercise the actual val_cut logic in analyze_manifest
    without an empty validation partition. mu is built AFTER the val cutoff
    so its index genuinely matches whatever analyze_manifest computes,
    rather than assuming positions."""
    pad_dates = pd.date_range("2024-12-01", periods=10, freq="D")
    pad_rows = pd.DataFrame({
        "date": np.repeat(pad_dates, 10),
        "ticker": [f"T{i}" for i in range(10)] * 10,
        "fwd_60d_excess": 0.0,
    })
    real_val, real_mu, regimes = _fixture()
    panel = pd.concat([pad_rows, real_val], ignore_index=True)
    panel["f1"] = 0.0

    distinct = sorted(panel["date"].unique())
    val_cut = pd.Timestamp(distinct[int(len(distinct) * 0.8)])
    expected_val = panel[panel["date"] > val_cut].copy()
    assert set(expected_val["date"].unique()) == set(real_val["date"].unique()), (
        "fixture construction bug: padding did not correctly isolate the "
        "real dates into the val partition")

    mu = pd.Series(
        [float(i % 10) for i in range(len(expected_val))],
        index=expected_val.index,
    )
    return panel, mu, regimes


def test_cli_dump_predictions_writes_manifest_and_verifies(tmp_path, monkeypatch):
    """Drive the actual --dump-predictions CLI path (not just internal
    functions) and prove: output rows correspond exactly to the scored val
    index, regime-join cardinality is one-to-one, and the manifest's
    output_content_sha256 verifies against a fresh reload."""
    panel, mu, regimes = _padded_panel_and_mu()

    import renquant_backtesting.analysis.analyze_manifest_sanity_placebo as mod

    monkeypatch.setattr(mod, "_load_artifact_payload", lambda p: {"feature_cols": ["f1"]})
    monkeypatch.setattr(mod, "_sanity_model_label_col", lambda a: "fwd_60d_excess")
    monkeypatch.setattr(mod, "_load_sanity_panel", lambda feat_cols, label: (panel.copy(), {}))
    monkeypatch.setattr(
        mod, "_score_manifest_sanity",
        lambda val_df, feat_cols, manifest_path, artifact_path, artifact, panel_history=None: (mu, {}))
    monkeypatch.setattr(mod, "build_regime_series", lambda dates, strategy_dir=None: regimes)
    monkeypatch.setattr(mod, "summarize_ic", lambda *a, **k: {})
    monkeypatch.setattr(mod, "shift_diagnostics", lambda *a, **k: [])
    monkeypatch.setattr(mod, "regime_diagnostics", lambda *a, **k: {})
    monkeypatch.setattr(mod, "regime_shift_diagnostics", lambda *a, **k: {})

    dump_path = tmp_path / "exp" / "pick_table.parquet"
    mod.analyze_manifest(
        artifact_path=tmp_path / "fake_artifact.json",
        manifest_path=tmp_path / "fake_manifest.json",
        label="fwd_60d_excess",
        strategy_dir=tmp_path,
        shifts=[60],
        min_names=1,
        dump_predictions=dump_path,
    )
    assert dump_path.exists()
    manifest_path_out = dump_path.with_suffix(".manifest.json")
    assert manifest_path_out.exists()
    manifest_out = json.loads(manifest_path_out.read_text())

    reloaded = pd.read_parquet(dump_path)
    # every (date, ticker) row present exactly once -- one-to-one regime join,
    # no fan-out duplication
    assert reloaded.duplicated(subset=["date", "ticker"]).sum() == 0
    # exactly the scored val index's (date, ticker) pairs, no more/fewer
    expected_pairs = set(zip(
        pd.to_datetime(mu.index.map(lambda i: panel.loc[i, "date"])).astype(str),
        panel.loc[mu.index, "ticker"],
    ))
    actual_pairs = set(zip(reloaded["date"].astype(str), reloaded["ticker"]))
    assert actual_pairs == expected_pairs
    reloaded_hash = canonical_pick_table_content_hash(reloaded, "fwd_60d_excess")
    assert reloaded_hash == manifest_out["output"]["output_content_sha256"]
    assert manifest_out["research_only"] is True


def test_cli_dump_predictions_live_path_raises(tmp_path, monkeypatch):
    import renquant_backtesting.analysis.analyze_manifest_sanity_placebo as mod

    panel, mu, regimes = _padded_panel_and_mu()
    monkeypatch.setattr(mod, "_load_artifact_payload", lambda p: {"feature_cols": ["f1"]})
    monkeypatch.setattr(mod, "_sanity_model_label_col", lambda a: "fwd_60d_excess")
    monkeypatch.setattr(mod, "_load_sanity_panel", lambda feat_cols, label: (panel.copy(), {}))
    monkeypatch.setattr(
        mod, "_score_manifest_sanity",
        lambda val_df, feat_cols, manifest_path, artifact_path, artifact, panel_history=None: (mu, {}))
    monkeypatch.setattr(mod, "build_regime_series", lambda dates, strategy_dir=None: regimes)
    monkeypatch.setattr(mod, "summarize_ic", lambda *a, **k: {})
    monkeypatch.setattr(mod, "shift_diagnostics", lambda *a, **k: [])

    with pytest.raises(mod.ResearchOnlyOutputPathError):
        mod.analyze_manifest(
            artifact_path=tmp_path / "fake_artifact.json",
            manifest_path=tmp_path / "fake_manifest.json",
            label="fwd_60d_excess",
            strategy_dir=tmp_path,
            shifts=[60],
            min_names=1,
            dump_predictions=tmp_path / "live" / "pick_table.parquet",
        )
