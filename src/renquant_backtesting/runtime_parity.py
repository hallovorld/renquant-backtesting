"""Backtesting adapters for the shared runtime decision contract."""
from __future__ import annotations

from typing import Any


def simulate_panel_scoring_decisions(
    *,
    strategy_config: dict[str, Any],
    data_manifest: dict[str, Any],
    artifact_manifest: dict[str, Any],
    market_snapshot: dict[str, Any],
    account_snapshot: dict[str, Any] | None = None,
    emit_orders: bool = False,
) -> dict[str, Any]:
    """Run the shared panel-scoring contract for a simulation bar."""
    from renquant_pipeline import InferenceContext, PanelScoringJob, RuntimeInferencePipeline

    ctx = InferenceContext(
        strategy_config=strategy_config,
        data_manifest=data_manifest,
        artifact_manifest=artifact_manifest,
        market_snapshot=market_snapshot,
        account_snapshot=account_snapshot or {},
    )
    result = RuntimeInferencePipeline([PanelScoringJob(emit_orders=emit_orders)]).run(ctx)
    return {
        "ok": result.ok,
        "scores": dict(ctx.scores),
        "accepted_candidates": list(getattr(ctx, "accepted_candidates", []) or []),
        "blocked_by": dict(ctx.blocked_by),
        "buy_blocked": bool(ctx.buy_blocked),
        "decision_trace": list(ctx.decision_trace),
        "order_intents": list(ctx.order_intents),
        "steps": [record.job_name for record in result.steps],
    }
