"""Contracts for the lifted dashboard reporting CLI."""
from __future__ import annotations

from pathlib import Path

from renquant_backtesting.reporting import build_dashboard


def test_build_dashboard_uses_explicit_repo_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "umbrella"
    (repo_root / "doc").mkdir(parents=True)
    (repo_root / "backtesting" / "renquant_104").mkdir(parents=True)
    (repo_root / "doc" / "roadmap.md").write_text(
        "## P0\n\n### 1. Keep operations multirepo\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(build_dashboard, "REPO_ROOT", repo_root)
    out = repo_root / "doc" / "dashboard.md"
    md = build_dashboard.build("alpaca", out)

    assert out.exists()
    assert "# RenQuant Dashboard" in md
    assert "DB unavailable" in md
    assert "Keep operations multirepo" in md


def test_dashboard_model_path_uses_strategy_config_env(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "umbrella"
    strategy_dir = repo_root / "backtesting" / "renquant_104"
    strategy_dir.mkdir(parents=True)
    pinned = tmp_path / "runtime" / "renquant-strategy-104" / "configs" / "strategy_config.json"
    pinned.parent.mkdir(parents=True)
    pinned.write_text(
        '{"ranking": {"panel_scoring": {"artifact_path": "artifacts/prod/panel-ltr.env.json"}}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(build_dashboard, "REPO_ROOT", repo_root)
    monkeypatch.setenv("RENQUANT_STRATEGY_CONFIG", str(pinned))

    assert build_dashboard._resolve_prod_panel_path() == (
        strategy_dir / "artifacts" / "prod" / "panel-ltr.env.json"
    )
