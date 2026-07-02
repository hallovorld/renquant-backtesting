"""Unit tests for the pick-table export contract (--dump-predictions).

Covers the #59 review points:
  1. decile_rank uses the deterministic qcut/rank algorithm shared with
     RenQuant#430 — ties, fewer-than-10 names, exactly 10, and balance.
  2. build_pick_table aligns scores strictly BY INDEX and fails on
     missing/duplicate indexes (shuffled-index regression included).
  3. canonical content hash is byte-compatible with RenQuant#430's
     (golden hash computed with #430's implementation verbatim) and the
     sidecar manifest verifies after reload.
  4. realized-label temporal semantics are stamped and canonical production
     output paths are refused.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from renquant_backtesting.analysis.pick_table import (
    ResearchOnlyOutputPathError,
    build_pick_table,
    build_pick_table_manifest,
    canonical_table_content_hash,
    decile_rank,
    default_sidecar_path,
    pick_table_columns,
    refuse_production_output_path,
    sha256_file,
    verify_pick_table,
)

LABEL = "fwd_60d_excess"


def _fixture(n: int = 10):
    dates = pd.to_datetime(["2025-01-02"] * n + ["2025-01-03"] * n)
    tickers = [f"T{i}" for i in range(n)] * 2
    val = pd.DataFrame({
        "date": dates,
        "ticker": tickers,
        LABEL: [0.01 * i for i in range(n)] * 2,
    })
    # day 2 reverses the ranking so deciles must be per-date
    mu = pd.Series(
        [float(i) for i in range(n)] + [float(n - 1 - i) for i in range(n)],
        index=val.index,
    )
    regimes = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
        "regime": ["BULL_CALM", "BEAR"],
    })
    return val, mu, regimes


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

def test_schema_columns_and_canonical_sort():
    val, mu, regimes = _fixture()
    out = build_pick_table(val, mu, LABEL, regimes)
    assert list(out.columns) == pick_table_columns(LABEL) == [
        "date", "name", "score", "decile_rank", LABEL, "regime",
    ]
    assert len(out) == 20
    # canonical (date, name) sort — the #430 on-disk order
    assert out.equals(out.sort_values(["date", "name"]).reset_index(drop=True))


def test_regime_join_values_and_per_date_deciles():
    val, mu, regimes = _fixture()
    out = build_pick_table(val, mu, LABEL, regimes)
    d1 = out[out["date"] == "2025-01-02"]
    d2 = out[out["date"] == "2025-01-03"]
    assert set(d1["regime"]) == {"BULL_CALM"} and set(d2["regime"]) == {"BEAR"}
    # mu reverses on day 2: the top-decile name must flip between the dates
    assert d1.loc[d1["score"].idxmax(), "name"] != d2.loc[d2["score"].idxmax(), "name"]
    for _, g in out.groupby("date"):
        assert g.loc[g["score"].idxmax(), "decile_rank"] == 9
        assert g.loc[g["score"].idxmin(), "decile_rank"] == 0


def test_empty_regimes_yields_none_column():
    val, mu, _ = _fixture()
    out = build_pick_table(val, mu, LABEL, pd.DataFrame())
    assert "regime" in out.columns
    assert out["regime"].isna().all()


# ---------------------------------------------------------------------------
# review point 1 — decile correctness (ties, <10, exactly 10, balance)
# ---------------------------------------------------------------------------

def test_exactly_ten_distinct_scores_is_balanced_0_to_9():
    """Regression for the rank(pct=True)*10 bug: with exactly 10 distinct
    scores the old construction produced no decile 0 and a doubled decile 9."""
    scores = pd.Series([float(i) for i in range(10)])
    deciles = decile_rank(scores)
    assert sorted(deciles) == list(range(10))  # each decile exactly once


def test_decile_balance_thirty_names():
    scores = pd.Series([float(i) for i in range(30)])
    counts = decile_rank(scores).value_counts()
    assert sorted(counts.index) == list(range(10))
    assert set(counts.values) == {3}  # exactly balanced


def test_decile_fewer_than_ten_names_falls_back_monotonically():
    scores = pd.Series([3.0, 1.0, 2.0, 0.0, 5.0, 4.0])
    deciles = decile_rank(scores)
    # 6 distinct values -> 6 buckets 0..5, monotone in score
    assert sorted(deciles) == list(range(6))
    ordered = deciles[scores.sort_values().index].tolist()
    assert ordered == sorted(ordered)


def test_decile_ties_deterministic_and_near_balanced():
    scores = pd.Series([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0])
    a = decile_rank(scores)
    b = decile_rank(scores.copy())
    assert a.tolist() == b.tolist()  # deterministic under ties
    counts = a.value_counts()
    assert counts.max() - counts.min() <= 1  # near-balanced (12 rows, 10 bins)
    # monotone: a higher score never gets a lower decile
    ordered = a[scores.sort_values(kind="stable").index].tolist()
    assert ordered == sorted(ordered)


def test_decile_ties_exact_per_position_result():
    """Ties are broken by position (rank method='first'), so a 3-way tie can
    split across a bin boundary — assert the exact deterministic per-position
    result, not a 'ties always share a bucket' property the #430 reference
    method deliberately does not provide (bin sizes stay balanced instead)."""
    scores = pd.Series([1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert decile_rank(scores).tolist() == [0, 0, 1, 2, 3, 3, 4, 5, 6, 6]


def test_decile_constant_scores_single_bucket():
    scores = pd.Series([1.0] * 7)
    assert decile_rank(scores).tolist() == [0] * 7


# ---------------------------------------------------------------------------
# review point 2 — strict index alignment
# ---------------------------------------------------------------------------

def test_shuffled_mu_index_regression():
    """Same labels, different order: scores must attach to the right ticker
    (the original to_numpy() implementation attached them positionally)."""
    val, mu, regimes = _fixture()
    shuffled = mu.sample(frac=1.0, random_state=7)
    assert list(shuffled.index) != list(mu.index)  # genuinely shuffled
    out = build_pick_table(val, shuffled, LABEL, regimes)
    truth = {
        (d, t): m for d, t, m in zip(val["date"], val["ticker"], mu)
    }
    for row in out.itertuples(index=False):
        assert row.score == truth[(row.date, row.name)]


def test_duplicate_val_index_raises():
    val, mu, regimes = _fixture()
    val2 = val.copy()
    val2.index = [0] + list(val.index[:-1])
    with pytest.raises(ValueError, match="duplicate index"):
        build_pick_table(val2, mu, LABEL, regimes)


def test_duplicate_mu_index_raises():
    val, mu, regimes = _fixture()
    mu2 = pd.concat([mu, mu.iloc[:1]])
    with pytest.raises(ValueError, match="duplicate index"):
        build_pick_table(val, mu2, LABEL, regimes)


def test_missing_mu_labels_raises():
    val, mu, regimes = _fixture()
    with pytest.raises(ValueError, match="missing scores"):
        build_pick_table(val, mu.iloc[:-2], LABEL, regimes)


def test_nan_score_raises_not_silently_dropped():
    val, mu, regimes = _fixture()
    mu2 = mu.copy()
    mu2.iloc[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        build_pick_table(val, mu2, LABEL, regimes)


def test_duplicate_date_name_rows_raise():
    val, mu, regimes = _fixture()
    val2 = pd.concat([val, val.iloc[:1]], ignore_index=True)
    mu2 = pd.concat([mu, mu.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match=r"duplicate \(date, name\)"):
        build_pick_table(val2, mu2, LABEL, regimes)


def test_regime_join_must_be_one_to_one():
    val, mu, regimes = _fixture()
    dup = pd.concat([regimes, regimes.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="one-to-one"):
        build_pick_table(val, mu, LABEL, dup)


# ---------------------------------------------------------------------------
# review point 3 — durable contract: canonical hash + sidecar verification
# ---------------------------------------------------------------------------

# Computed with RenQuant#430's scripts/regen_oos_pick_table.py::
# canonical_table_content_hash VERBATIM on the GOLDEN table below — pins this
# repo's implementation to the already-merged contract mechanically.
GOLDEN_430_HASH = "5bb03dd21a2d0a6ce5c9d9e23ff73b7bc7fae0b82b60ce3cae0b18fbd7c17510"

GOLDEN = pd.DataFrame({
    "date": pd.to_datetime(["2025-01-03", "2025-01-02", "2025-01-02"]),
    "name": ["BBB", "AAA", "BBB"],
    "score": [0.25, -0.1, 1.5],
    "decile_rank": [0, 0, 9],
    "fwd_60d_excess": [0.01, -0.02, 0.03],
    "regime": ["BEAR", "BULL_CALM", "BULL_CALM"],
})


def test_canonical_hash_matches_renquant_430_golden():
    assert canonical_table_content_hash(GOLDEN, label=LABEL) == GOLDEN_430_HASH


def test_canonical_hash_order_independent_and_content_sensitive():
    shuffled = GOLDEN.sample(frac=1.0, random_state=3)
    assert canonical_table_content_hash(shuffled, label=LABEL) == GOLDEN_430_HASH
    tampered = GOLDEN.copy()
    tampered.loc[0, "score"] = 0.250001
    assert canonical_table_content_hash(tampered, label=LABEL) != GOLDEN_430_HASH


def _write_dump(tmp_path: Path):
    val, mu, regimes = _fixture()
    table = build_pick_table(val, mu, LABEL, regimes)
    parquet = tmp_path / "pick_table.parquet"
    table.to_parquet(parquet, index=False)
    manifest_input = tmp_path / "wf_manifest.json"
    manifest_input.write_text('{"retrains": []}')
    reference_artifact = tmp_path / "artifact.json"
    reference_artifact.write_text('{"kind": "test"}')
    sidecar = build_pick_table_manifest(
        table,
        label=LABEL,
        generator="tests/analysis/test_build_pick_table.py",
        generator_path=Path(__file__),
        manifest_input=manifest_input,
        reference_artifact=reference_artifact,
        val_cut="2025-01-01",
        val_start="2025-01-02",
        val_end="2025-01-03",
        label_lookahead_days=60,
        output_content_sha256=canonical_table_content_hash(table, label=LABEL),
        output_parquet_sha256=sha256_file(parquet),
    )
    sidecar_path = default_sidecar_path(parquet)
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    return table, parquet, sidecar_path, sidecar


def test_sidecar_verifies_after_reload(tmp_path):
    _, parquet, sidecar_path, sidecar = _write_dump(tmp_path)
    result = verify_pick_table(parquet, sidecar_path)
    assert result["content_verified"] and result["counts_verified"]
    assert result["parquet_transport_match"] is True
    # review point 3: manifest/artifact/generator hashes + label + window
    recipe = sidecar["recipe"]
    for key in ("generator_sha256", "contract_module_sha256",
                "manifest_input_sha256", "reference_artifact_sha256"):
        assert len(recipe[key]) == 64
    assert recipe["label"] == LABEL
    assert recipe["val_cut"] == "2025-01-01"
    assert recipe["val_start"] == "2025-01-02"
    assert recipe["val_end"] == "2025-01-03"
    assert sidecar["counts"]["n_rows"] == 20


def test_verify_detects_tampered_content(tmp_path):
    table, parquet, sidecar_path, _ = _write_dump(tmp_path)
    tampered = table.copy()
    tampered.loc[0, "score"] = tampered.loc[0, "score"] + 1.0
    tampered.to_parquet(parquet, index=False)
    with pytest.raises(ValueError, match="content hash mismatch"):
        verify_pick_table(parquet, sidecar_path)


# ---------------------------------------------------------------------------
# review point 4 — temporal semantics + production-path refusal
# ---------------------------------------------------------------------------

def test_sidecar_stamps_realized_label_semantics(tmp_path):
    _, _, _, sidecar = _write_dump(tmp_path)
    ts = sidecar["temporal_semantics"]
    assert ts["research_only"] is True
    assert ts["label_lookahead_days"] == 60
    assert "REALIZED" in ts["label_realized"]
    assert "NEVER" in ts["not_a_live_input"]


@pytest.mark.parametrize("bad", [
    "/repo/data/pick.parquet",
    "/repo/data/ohlcv/pick.parquet",
    "/repo/backtesting/renquant_104/artifacts/sim/pick.parquet",
])
def test_refuses_canonical_production_paths(bad):
    with pytest.raises(ResearchOnlyOutputPathError, match="refusing"):
        refuse_production_output_path(Path(bad))


@pytest.mark.parametrize("bad", [
    "/repo/research/live_mirror/pick.parquet",
    "/repo/prod_dumps/pick.parquet",
    "/repo/outputs/production/pick.parquet",
])
def test_refuses_live_prod_marker_paths(bad):
    """Name-based heuristic layered on the structural rules: any resolved
    path component containing live/prod/production is refused."""
    with pytest.raises(ResearchOnlyOutputPathError, match="refusing"):
        refuse_production_output_path(Path(bad))


def test_allow_production_flag_bypasses_guard():
    refuse_production_output_path(
        Path("/repo/data/pick.parquet"), allow_production=True)
    refuse_production_output_path(
        Path("/repo/live/pick.parquet"), allow_production=True)


def test_allows_research_and_scratch_paths(tmp_path):
    refuse_production_output_path(Path("/repo/data/exp/pick.parquet"))
    refuse_production_output_path(Path("/repo/data/exp/sub/pick.parquet"))
    refuse_production_output_path(tmp_path / "pick.parquet")
