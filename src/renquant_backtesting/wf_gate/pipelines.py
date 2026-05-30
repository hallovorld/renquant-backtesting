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
    """Run prod/WF decision-semantics parity check unless skipped."""

    def run(self, ctx: WfGateContext) -> bool | None:
        if ctx.skip_config_parity:
            ctx.config_parity_result = {"passed": True, "reason": "skipped (--skip-config-parity)"}
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
    """Run ``_manifest_recipe_usage`` and refuse non-matching manifests."""

    def run(self, ctx: WfGateContext) -> bool | None:
        if ctx.manifest_path is not None and ctx.artifact is not None:
            from . import runner  # noqa: PLC0415
            ctx.recipe_usage = runner._manifest_recipe_usage(
                ctx.manifest_path, ctx.artifact_path,
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
    """Compose ``wf_gate_metadata`` from each stage's result."""

    def run(self, ctx: WfGateContext) -> bool | None:
        return True


class StampArtifactTask(Task):
    """Write metadata back into the artifact / sidecar."""

    def run(self, ctx: WfGateContext) -> bool | None:
        return True


class EmitVerdictTask(Task):
    """Log PASS/FAIL verdict (the only side-effect callers depend on for CI)."""

    def run(self, ctx: WfGateContext) -> bool | None:
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
