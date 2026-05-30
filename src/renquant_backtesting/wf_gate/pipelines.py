"""Task/Job/Pipeline scaffold for ``wf_gate.runner`` — Phase 2 progress.

The 2525-line ``runner.py`` is procedural with 5 implicit stages. This module
exposes the same flow as a Pipeline of 5 Jobs whose Tasks delegate (today) to
the existing runner.py functions. Phase 2 work moves implementations into the
Tasks incrementally; Phase 5 flips callers to use this Pipeline.

Stages (mapped to ``runner.main()`` linear flow):
    1. ConfigJob          : load artifact + derive/parity strategy config
    2. RecipeMatchJob     : resolve manifest, validate recipe fingerprint
    3. WfSimJob           : run 3 sim cuts (sequential or pooled via --jobs)
    4. TradeGateJob       : run trade-contract + trade-monotonicity gates
    5. SanityJob          : run §5.2 sanity battery (shuffled + time-shift)
    6. StampJob           : write wf_gate_metadata + emit PASS/FAIL verdict

The scaffold returns the live ``PipelineResult`` so callers can inspect skipped
stages, per-step elapsed time, and audit each Job at runtime. Existing
``runner.main()`` is unchanged; it is still the authoritative live entry point.
This module is **import-only** for now and is intended to grow into the
production path during Phase 5.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from renquant_common import Job, Pipeline, Task


@dataclass
class WfGateContext:
    """State threaded through the wf_gate Pipeline."""

    artifact_path: Path
    strategy_config: str
    strategy_dir: Path | None = None  # umbrella's backtesting/<strategy>/ (for manifest URI resolution)
    artifact: dict | None = None
    # Stage 1
    base_config: dict | None = None
    config_parity_result: dict | None = None
    # Stage 2
    manifest_path: Path | None = None
    recipe_usage: dict | None = None
    # Stage 3
    trace_dir: Path | None = None
    wf_result: dict | None = None
    # Stage 4
    trade_contract_result: dict | None = None
    trade_gate_result: dict | None = None
    # Stage 5
    sanity_result: dict | None = None
    # Stage 6
    wf_meta: dict = field(default_factory=dict)
    overall_pass: bool = False
    # CLI knobs that gate stages skip themselves
    skip_wf: bool = False
    skip_sanity: bool = False
    skip_config_parity: bool = False
    skip_trade_gates: bool = False
    derive_config_from_prod: bool = False
    preserve_experiment_overrides: bool = False
    jobs: int = 1


# ─── Stage 1: ConfigJob ──────────────────────────────────────────────────────

class LoadArtifactTask(Task):
    """Load the candidate artifact JSON or sequence-checkpoint sidecar.

    Phase 3a: implementation lifted out of runner.py into wf_gate.artifact_loader
    so the Task is independently testable (no need to load the 2525-line runner
    to exercise loading semantics).
    """

    def run(self, ctx: WfGateContext) -> bool | None:
        from .artifact_loader import load_artifact_payload  # noqa: PLC0415
        ctx.artifact = load_artifact_payload(ctx.artifact_path)
        return True


class DeriveConfigTask(Task):
    """Optionally derive a prod-semantic WF config (--derive-config-from-prod)."""

    def run(self, ctx: WfGateContext) -> bool | None:
        # Delegate kept thin: the legacy main() handles all the derive logic
        # inline; this Task is a placeholder so Phase 2 can lift it cleanly.
        return True


class CheckConfigParityTask(Task):
    """Run prod/WF decision-semantics parity check unless skipped.

    Phase 3c: calls ``wf_config_parity.evaluate_wf_config_parity`` directly with
    ``strategy_dir`` injected via context (overriding the module-level default
    which is wrong in the copied package — fine for the lift, runner copy keeps
    its own correct value).
    """

    def run(self, ctx: WfGateContext) -> bool | None:
        if ctx.skip_config_parity:
            ctx.config_parity_result = {"passed": True, "reason": "skipped (--skip-config-parity)"}
            return True
        if ctx.strategy_dir is None or ctx.artifact_path is None:
            ctx.config_parity_result = {
                "passed": True, "reason": "skipped (no strategy_dir or artifact)",
            }
            return True
        prod_cfg = ctx.strategy_dir / "strategy_config.json"
        wf_cfg = ctx.strategy_dir / ctx.strategy_config
        if not prod_cfg.exists() or not wf_cfg.exists():
            ctx.config_parity_result = {
                "passed": True,
                "reason": f"skipped (config not found: prod={prod_cfg.exists()} wf={wf_cfg.exists()})",
            }
            return True
        from .wf_config_parity import evaluate_wf_config_parity  # noqa: PLC0415
        ctx.config_parity_result = evaluate_wf_config_parity(
            prod_cfg, wf_cfg,
            candidate_artifact=ctx.artifact_path,
            strategy_dir=ctx.strategy_dir,
        )
        return True


class ConfigJob(Job):
    @property
    def tasks(self) -> list[Task]:
        return [LoadArtifactTask(), DeriveConfigTask(), CheckConfigParityTask()]


# ─── Stage 2: RecipeMatchJob ─────────────────────────────────────────────────

class ResolveManifestTask(Task):
    """Select the same-recipe WF manifest from the config + sim search dir."""

    def run(self, ctx: WfGateContext) -> bool | None:
        return True


class ValidateRecipeMatchTask(Task):
    """Run ``manifest_recipe_usage`` and refuse non-matching manifests.

    Phase 3b.2: uses the lifted ``recipe_match.manifest_recipe_usage`` directly
    (with ``ctx.strategy_dir`` for URI resolution) — no longer imports runner.
    """

    def run(self, ctx: WfGateContext) -> bool | None:
        if ctx.manifest_path is not None and ctx.strategy_dir is not None:
            from .recipe_match import manifest_recipe_usage  # noqa: PLC0415
            ctx.recipe_usage = manifest_recipe_usage(
                ctx.manifest_path, ctx.artifact_path,
                strategy_dir=ctx.strategy_dir,
            )
        return True


class RecipeMatchJob(Job):
    @property
    def tasks(self) -> list[Task]:
        return [ResolveManifestTask(), ValidateRecipeMatchTask()]


# ─── Stage 3: WfSimJob ───────────────────────────────────────────────────────

class RunWfSimTask(Task):
    """Run all 3 WF cuts (sequential or pooled per ``ctx.jobs``)."""

    def run(self, ctx: WfGateContext) -> bool | None:
        return True


class WfSimJob(Job):
    def should_skip(self, ctx: WfGateContext) -> bool:
        return bool(ctx.skip_wf)

    @property
    def tasks(self) -> list[Task]:
        return [RunWfSimTask()]


# ─── Stage 4: TradeGateJob ───────────────────────────────────────────────────

class RunTradeContractTask(Task):
    """Run ``run_trade_contract_gate``."""

    def run(self, ctx: WfGateContext) -> bool | None:
        return True


class RunTradeMonotonicityTask(Task):
    """Run ``run_trade_monotonicity_gate``."""

    def run(self, ctx: WfGateContext) -> bool | None:
        return True


class TradeGateJob(Job):
    def should_skip(self, ctx: WfGateContext) -> bool:
        return bool(ctx.skip_trade_gates) or bool(ctx.skip_wf)

    @property
    def tasks(self) -> list[Task]:
        return [RunTradeContractTask(), RunTradeMonotonicityTask()]


# ─── Stage 5: SanityJob ──────────────────────────────────────────────────────

class RunSanityBatteryTask(Task):
    """Run the §5.2 sanity battery (shuffled-label + time-shift placebos)."""

    def run(self, ctx: WfGateContext) -> bool | None:
        return True


class SanityJob(Job):
    def should_skip(self, ctx: WfGateContext) -> bool:
        return bool(ctx.skip_sanity)

    @property
    def tasks(self) -> list[Task]:
        return [RunSanityBatteryTask()]


# ─── Stage 6: StampJob ───────────────────────────────────────────────────────

class AssembleMetadataTask(Task):
    """Compose ``wf_gate_metadata`` from each stage's result.

    Phase 3h: collects ctx fields produced by the prior Jobs into a single dict
    suitable for stamping into the artifact. Mirrors what runner.main()'s
    inline ``wf_meta = {...}`` assembly does today.
    """

    def run(self, ctx: WfGateContext) -> bool | None:
        ctx.wf_meta = {
            # Stage 1
            "config_parity": ctx.config_parity_result,
            # Stage 2
            "recipe_usage": ctx.recipe_usage,
            # Stage 3
            "wf_result": ctx.wf_result,
            # Stage 4
            "trade_contract": ctx.trade_contract_result,
            "trade_monotonicity": ctx.trade_gate_result,
            # Stage 5
            "sanity": ctx.sanity_result,
            # Trace
            "wf_trade_trace_dir": str(ctx.trace_dir) if ctx.trace_dir else None,
        }
        # Drop None fields so the stamped dict is honest about what ran.
        ctx.wf_meta = {k: v for k, v in ctx.wf_meta.items() if v is not None}
        return True


class StampArtifactTask(Task):
    """Write ``wf_gate_metadata`` back into the artifact JSON or sidecar.

    Phase 3h: uses the lifted ``artifact_loader.write_artifact_payload`` so the
    Task no longer imports runner. Preserves the existing artifact payload by
    re-reading + merging the metadata.wf_gate_metadata key (the historical
    field name the preflight reads).
    """

    def run(self, ctx: WfGateContext) -> bool | None:
        if ctx.artifact is None:
            return True
        from .artifact_loader import write_artifact_payload  # noqa: PLC0415
        payload = dict(ctx.artifact)
        metadata = dict(payload.get("metadata") or {})
        metadata["wf_gate_metadata"] = ctx.wf_meta
        payload["metadata"] = metadata
        write_artifact_payload(ctx.artifact_path, payload)
        return True


class EmitVerdictTask(Task):
    """Compute the overall PASS/FAIL and set ``ctx.overall_pass``.

    A run passes when every stage that produced a result reports ``passed=True``.
    Stages that were skipped do not block the verdict.
    """

    def run(self, ctx: WfGateContext) -> bool | None:
        per_stage = (
            ctx.config_parity_result,
            ctx.recipe_usage,
            ctx.wf_result,
            ctx.trade_contract_result,
            ctx.trade_gate_result,
            ctx.sanity_result,
        )
        # Manifest recipe validation lives under "recipe_validated", others under "passed".
        def _ok(r: dict | None) -> bool:
            if not r:
                return True  # skipped / not produced — not a fail
            if "passed" in r:
                return bool(r.get("passed"))
            if "recipe_validated" in r:
                return bool(r.get("recipe_validated"))
            return True
        ctx.overall_pass = all(_ok(r) for r in per_stage)
        return True


class StampJob(Job):
    @property
    def tasks(self) -> list[Task]:
        return [AssembleMetadataTask(), StampArtifactTask(), EmitVerdictTask()]


# ─── Top-level Pipeline ──────────────────────────────────────────────────────

def build_wf_gate_pipeline() -> Pipeline:
    """The wf_gate Pipeline (5 functional Jobs + the StampJob output stage)."""
    return Pipeline(
        [ConfigJob(), RecipeMatchJob(), WfSimJob(), TradeGateJob(),
         SanityJob(), StampJob()],
        name="wf-gate",
    )
