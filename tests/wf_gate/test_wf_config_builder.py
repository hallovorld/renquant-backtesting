"""Tests for the converged WF-gate config-parity contract (active path).

The ONE contract (shared with umbrella ``scripts/run_wf_gate.py``): select the
production reference whose scorer kind MATCHES the candidate's declared kind
(read from artifact metadata, never a path suffix), and use that SAME selected
reference for BOTH derivation and the parity check. The builder must NOT mutate
``ranking.panel_scoring.kind`` to force parity — that would convert a genuine
prod-vs-candidate mismatch into a passing config.

Coverage:
  * the builder inherits ``kind`` unchanged from whatever prod config it is
    handed (no suffix inference, no mutation);
  * ``select_prod_reference_for_candidate`` picks the kind-matched reference,
    honors a validated env override, and FAILS CLOSED on unknown / mismatched
    kinds;
  * NEGATIVE: a GBDT candidate with NO GBDT reference selected (PatchTST prod)
    stays non-promotable — parity fails;
  * POSITIVE: a GBDT candidate WITH the GBDT/shadow reference selected passes,
    using the same selected reference for derivation and parity.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from renquant_backtesting.wf_gate.wf_config_builder import (
    build_wf_config_from_prod,
    select_prod_reference_for_candidate,
)
from renquant_backtesting.wf_gate.wf_config_parity import evaluate_wf_config_parity


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _prod_config(*, kind: str, artifact_path: str) -> dict:
    return {
        "ranking": {
            "panel_scoring": {
                "enabled": True,
                "kind": kind,
                "artifact_path": artifact_path,
                "buy_floor": "adaptive_mean_std",
            },
        },
    }


def _write_manifest(path: Path, artifact_uri: str) -> None:
    _write_json(path, {"retrains": [{"artifact_uri": artifact_uri}]})


def _gbdt_artifact(path: Path, cols: list[str]) -> Path:
    _write_json(path, {"kind": "panel_ltr_xgboost", "feature_cols": cols})
    return path


# ── builder no longer mutates kind ───────────────────────────────────────────


def test_builder_inherits_kind_unchanged_from_prod(tmp_path: Path) -> None:
    """The derived config carries the prod config's kind verbatim — no mutation.

    Even when the manifest artifact is a GBDT ``panel-ltr.json``, deriving from a
    PatchTST prod config keeps ``kind=hf_patchtst``. The builder must never infer
    a kind from the artifact suffix; selecting the correct reference is the
    caller's job.
    """
    gbdt_artifact = _gbdt_artifact(tmp_path / "artifacts" / "sim" / "panel-ltr.json", ["a", "b"])
    manifest = tmp_path / "artifacts" / "wf" / "manifest.json"
    _write_manifest(manifest, str(gbdt_artifact.relative_to(tmp_path)))

    prod_cfg = _prod_config(kind="hf_patchtst", artifact_path="artifacts/prod/model.pt")
    base_cfg = {"walkforward": {"manifest_path": str(manifest.relative_to(tmp_path))}}

    derived = build_wf_config_from_prod(
        prod_cfg,
        manifest_path=str(manifest.relative_to(tmp_path)),
        base_wf_config=base_cfg,
        strategy_dir=tmp_path,
    )

    panel = derived["ranking"]["panel_scoring"]
    assert panel["kind"] == "hf_patchtst"  # inherited, NOT mutated to xgb
    assert panel["artifact_path"].endswith("panel-ltr.json")


def test_builder_inherits_xgb_kind_from_gbdt_prod(tmp_path: Path) -> None:
    """Derived config inherits ``xgb`` when the prod config IS the GBDT reference."""
    gbdt_artifact = _gbdt_artifact(tmp_path / "artifacts" / "sim" / "panel-ltr.json", ["a", "b"])
    manifest = tmp_path / "artifacts" / "wf" / "manifest.json"
    _write_manifest(manifest, str(gbdt_artifact.relative_to(tmp_path)))

    prod_cfg = _prod_config(kind="xgb", artifact_path="artifacts/prod/panel-ltr.json")
    base_cfg = {"walkforward": {"manifest_path": str(manifest.relative_to(tmp_path))}}

    derived = build_wf_config_from_prod(
        prod_cfg,
        manifest_path=str(manifest.relative_to(tmp_path)),
        base_wf_config=base_cfg,
        strategy_dir=tmp_path,
    )

    assert derived["ranking"]["panel_scoring"]["kind"] == "xgb"


# ── candidate-matched reference selection ─────────────────────────────────────


def _seed_references(strategy_dir: Path) -> None:
    _write_json(
        strategy_dir / "strategy_config.json",
        _prod_config(kind="hf_patchtst", artifact_path="artifacts/prod/model.pt"),
    )
    _write_json(
        strategy_dir / "strategy_config.shadow.json",
        _prod_config(kind="xgb", artifact_path="artifacts/prod/panel-ltr.json"),
    )


def test_select_reference_maps_gbdt_to_shadow(tmp_path: Path) -> None:
    _seed_references(tmp_path)
    ref = select_prod_reference_for_candidate("panel_ltr_xgboost", strategy_dir=tmp_path)
    assert ref == (tmp_path / "strategy_config.shadow.json").resolve()


def test_select_reference_maps_patchtst_to_primary(tmp_path: Path) -> None:
    _seed_references(tmp_path)
    ref = select_prod_reference_for_candidate("hf_patchtst", strategy_dir=tmp_path)
    assert ref == (tmp_path / "strategy_config.json").resolve()


def test_select_reference_unknown_kind_fails_closed(tmp_path: Path) -> None:
    _seed_references(tmp_path)
    with pytest.raises(ValueError, match="no production reference"):
        select_prod_reference_for_candidate("some_unknown_scorer", strategy_dir=tmp_path)


def test_select_reference_empty_kind_fails_closed(tmp_path: Path) -> None:
    _seed_references(tmp_path)
    with pytest.raises(ValueError, match="no declared scorer kind"):
        select_prod_reference_for_candidate(None, strategy_dir=tmp_path)


def test_select_reference_env_override_validated(tmp_path: Path) -> None:
    """A validated env override matching the candidate kind is honored."""
    _seed_references(tmp_path)
    ref = select_prod_reference_for_candidate(
        "panel_ltr_xgboost",
        strategy_dir=tmp_path,
        env_override="strategy_config.shadow.json",
    )
    assert ref == (tmp_path / "strategy_config.shadow.json").resolve()


def test_select_reference_env_override_mismatch_fails_closed(tmp_path: Path) -> None:
    """An env override whose kind != candidate kind FAILS CLOSED (no smuggling)."""
    _seed_references(tmp_path)
    with pytest.raises(ValueError, match="does not match the candidate kind"):
        select_prod_reference_for_candidate(
            "panel_ltr_xgboost",
            strategy_dir=tmp_path,
            env_override="strategy_config.json",  # PatchTST primary, kind=hf_patchtst
        )


# ── same-selected-reference contract: negative then positive ──────────────────


def test_gbdt_candidate_against_patchtst_reference_stays_non_promotable(tmp_path: Path) -> None:
    """NEGATIVE: GBDT candidate vs PatchTST prod (no GBDT ref) → parity FAILS.

    This is the genuine mismatch the previous kind-mutation defeated. With the
    converged contract the builder inherits ``kind=hf_patchtst`` from the
    PatchTST primary, the candidate artifact is GBDT, and the SAME PatchTST
    reference is used for both derivation and parity, so the kind/artifact guard
    fires and the run is non-promotable.
    """
    gbdt_artifact = _gbdt_artifact(tmp_path / "artifacts" / "sim" / "panel-ltr.json", ["a", "b"])
    manifest = tmp_path / "artifacts" / "wf" / "manifest.json"
    _write_manifest(manifest, str(gbdt_artifact.relative_to(tmp_path)))
    _seed_references(tmp_path)

    # Caller WRONGLY selects the PatchTST primary for a GBDT candidate (or no
    # GBDT reference exists). The same reference is used for derivation + parity.
    prod_ref = tmp_path / "strategy_config.json"
    prod_cfg = json.loads(prod_ref.read_text())
    base_cfg = {"walkforward": {"manifest_path": str(manifest.relative_to(tmp_path))}}

    derived = build_wf_config_from_prod(
        prod_cfg,
        manifest_path=str(manifest.relative_to(tmp_path)),
        base_wf_config=base_cfg,
        strategy_dir=tmp_path,
    )
    assert derived["ranking"]["panel_scoring"]["kind"] == "hf_patchtst"

    wf_path = tmp_path / "wf_config.json"
    _write_json(wf_path, derived)

    result = evaluate_wf_config_parity(
        prod_ref,
        wf_path,
        candidate_artifact=gbdt_artifact,
        strategy_dir=tmp_path,
    )
    assert result["passed"] is False
    # The PatchTST-kind-vs-JSON-artifact guard fires on the derived config.
    assert any(
        i.get("path", "").endswith("ranking.panel_scoring.artifact_path")
        for i in result["issues"]
    ), result["issues"]


def test_gbdt_candidate_against_selected_gbdt_reference_passes(tmp_path: Path) -> None:
    """POSITIVE: GBDT candidate WITH the selected GBDT/shadow reference passes.

    Same selected reference for derivation and parity; no kind mutation needed
    because the GBDT/shadow prod config already declares ``kind=xgb``.
    """
    gbdt_artifact = _gbdt_artifact(tmp_path / "artifacts" / "sim" / "panel-ltr.json", ["a", "b"])
    manifest = tmp_path / "artifacts" / "wf" / "manifest.json"
    _write_manifest(manifest, str(gbdt_artifact.relative_to(tmp_path)))
    _seed_references(tmp_path)

    # The selector chooses the GBDT/shadow reference for the GBDT candidate.
    prod_ref = select_prod_reference_for_candidate(
        gbdt_artifact and json.loads(gbdt_artifact.read_text())["kind"],
        strategy_dir=tmp_path,
    )
    assert prod_ref == (tmp_path / "strategy_config.shadow.json").resolve()

    # Point the shadow reference at the candidate artifact so the feature
    # contract matches (the real shadow config points at the prod GBDT artifact).
    prod_cfg = json.loads(prod_ref.read_text())
    prod_cfg["ranking"]["panel_scoring"]["artifact_path"] = str(gbdt_artifact)
    _write_json(prod_ref, prod_cfg)

    base_cfg = {"walkforward": {"manifest_path": str(manifest.relative_to(tmp_path))}}
    derived = build_wf_config_from_prod(
        prod_cfg,
        manifest_path=str(manifest.relative_to(tmp_path)),
        base_wf_config=base_cfg,
        strategy_dir=tmp_path,
    )
    assert derived["ranking"]["panel_scoring"]["kind"] == "xgb"

    wf_path = tmp_path / "wf_config.json"
    _write_json(wf_path, derived)

    result = evaluate_wf_config_parity(
        prod_ref,  # SAME selected reference used for derivation
        wf_path,
        candidate_artifact=gbdt_artifact,
        strategy_dir=tmp_path,
    )
    assert result["passed"] is True, result["issues"]
