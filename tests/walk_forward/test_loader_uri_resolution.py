from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from renquant_backtesting.walk_forward.loader import WalkForwardModelLoader


def _install_fake_panel_scorer(monkeypatch, loaded: list[Path]):
    class FakePanelScorer:
        @staticmethod
        def load(path):
            loaded.append(Path(path))
            return {"path": str(path)}

    module = types.ModuleType("renquant_pipeline.kernel.panel_pipeline.panel_scorer")
    module.PanelScorer = FakePanelScorer
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_model_as_of_resolves_relative_artifact_uri_against_manifest_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_dir = tmp_path / "strategy" / "artifacts" / "sim"
    artifact = manifest_dir / "walkforward_v2" / "2024-01-01" / "panel-ltr.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")
    manifest = manifest_dir / "walkforward_manifest.json"
    manifest.write_text(
        json.dumps({
            "retrains": [{
                "cutoff_date": "2024-01-01",
                "trained_date": "2024-01-02",
                "artifact_uri": "walkforward_v2/2024-01-01/panel-ltr.json",
                "lookahead_days": 0,
            }],
        }),
        encoding="utf-8",
    )

    loaded: list[Path] = []
    _install_fake_panel_scorer(monkeypatch, loaded)

    scorer = WalkForwardModelLoader(manifest).model_as_of("2024-01-03")

    assert scorer == {"path": str(artifact)}
    assert loaded == [artifact]


def test_model_as_of_resolves_strategy_artifact_uri_without_sim_double_prefix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    strategy = tmp_path / "strategy"
    manifest_dir = strategy / "artifacts" / "sim"
    artifact = strategy / "artifacts" / "walkforward_v2" / "2024-01-01" / "panel-ltr.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")
    manifest = manifest_dir / "walkforward_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({
            "retrains": [{
                "cutoff_date": "2024-01-01",
                "trained_date": "2024-01-02",
                "artifact_uri": "artifacts/walkforward_v2/2024-01-01/panel-ltr.json",
                "lookahead_days": 0,
            }],
        }),
        encoding="utf-8",
    )

    loaded: list[Path] = []
    _install_fake_panel_scorer(monkeypatch, loaded)

    scorer = WalkForwardModelLoader(manifest).model_as_of("2024-01-03")

    assert scorer == {"path": str(artifact)}
    assert loaded == [artifact]
    assert "artifacts/sim/artifacts" not in str(loaded[0])


def test_model_as_of_cache_key_uses_resolved_uri(tmp_path: Path, monkeypatch) -> None:
    manifest_dir = tmp_path / "strategy" / "artifacts" / "sim"
    artifact = manifest_dir / "walkforward_v2" / "2024-01-01" / "panel-ltr.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")
    manifest = manifest_dir / "walkforward_manifest.json"
    manifest.write_text(
        json.dumps({
            "retrains": [{
                "cutoff_date": "2024-01-01",
                "trained_date": "2024-01-02",
                "artifact_uri": str(artifact),
                "lookahead_days": 0,
            }],
        }),
        encoding="utf-8",
    )

    calls = 0

    class FakePanelScorer:
        @staticmethod
        def load(path):
            nonlocal calls
            calls += 1
            return object()

    module = types.ModuleType("renquant_pipeline.kernel.panel_pipeline.panel_scorer")
    module.PanelScorer = FakePanelScorer
    monkeypatch.setitem(sys.modules, module.__name__, module)

    loader = WalkForwardModelLoader(manifest)
    first = loader.model_as_of("2024-01-03")
    second = loader.model_as_of("2024-01-04")

    assert first is second
    assert calls == 1
