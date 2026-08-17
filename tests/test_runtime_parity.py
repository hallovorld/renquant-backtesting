from __future__ import annotations

import pytest

from renquant_backtesting.runtime_parity import simulate_panel_scoring_decisions


def _inputs() -> dict:
    return {
        "strategy_config": {
            "watchlist": ["AAPL", "MSFT"],
            "sector_map": {"AAPL": "TECH", "MSFT": "TECH"},
            "ranking": {"panel_scoring": {"enabled": True, "buy_floor": 0.5}},
            "execution": {"default_quantity": 1},
        },
        "data_manifest": {
            "dataset_id": "daily-fixture",
            "schema_version": "fixture-v1",
            "fingerprint": "sha256:data",
            "uri": "object://renquant-data/daily-fixture.parquet",
            "asset_class": "equity",
        },
        "artifact_manifest": {
            "artifact_id": "panel-ltr-prod",
            "model_family": "gbdt-panel-ltr",
            "strategy": "renquant_104",
            "fingerprint": "sha256:model",
            "uri": "object://renquant-artifacts/panel-ltr-prod.json",
            # renquant-artifacts made provenance mandatory on 2026-08-15
            # (PROVENANCE_REQUIRED_AFTER) and rejects kind='none' combined with
            # promotion_status='prod' -- correctly: a real prod artifact must carry
            # a canonical, registry-published lineage (publication_record_digest +
            # registry bindings). This fixture is a synthetic test double, not a
            # published production artifact, so claiming 'prod' was a fiction the
            # new guard exposed rather than a capability the guard removed.
            # Nothing under test reads promotion_status (no consumer in
            # renquant-backtesting/src), and 'candidate' is the value the other
            # manifest fixtures in these repos already use.
            "promotion_status": "candidate",
            "provenance": {"kind": "none"},
            "feature_cols": ["alpha_1", "alpha_2"],
            "metrics": {"accepted": True},
        },
        "market_snapshot": {
            "as_of": "2026-05-25",
            "feature_frame": {
                "AAPL": {"alpha_1": 1.0, "alpha_2": 0.5},
                "MSFT": {"alpha_1": -1.0, "alpha_2": 0.1},
            },
            "panel_scores": {"AAPL": 0.72, "MSFT": 0.21},
        },
    }


def test_backtesting_uses_shared_panel_scoring_contract_for_trade_admission() -> None:
    result = simulate_panel_scoring_decisions(**_inputs(), emit_orders=True)

    assert result["ok"] is True
    assert result["scores"] == {"AAPL": pytest.approx(0.72), "MSFT": pytest.approx(0.21)}
    assert [row["ticker"] for row in result["accepted_candidates"]] == ["AAPL"]
    assert result["blocked_by"] == {"MSFT": "panel_score_below_buy_floor"}
    assert result["order_intents"][0]["ticker"] == "AAPL"
    assert result["order_intents"][0]["attribution"]["source_job"] == "PanelScoringJob"


def test_backtesting_shared_contract_fails_closed_on_missing_feature() -> None:
    inputs = _inputs()
    inputs["market_snapshot"]["feature_frame"]["AAPL"] = {"alpha_1": 1.0}
    inputs["market_snapshot"]["feature_frame"]["MSFT"] = {"alpha_1": 0.2}

    result = simulate_panel_scoring_decisions(**inputs, emit_orders=True)

    assert result["buy_blocked"] is True
    assert result["order_intents"] == []
    assert result["blocked_by"] == {
        "AAPL": "feature_contract_missing:alpha_2",
        "MSFT": "feature_contract_missing:alpha_2",
    }
