"""Regression: relative manifest URIs must not double-resolve (artifacts/sim/artifacts/...)."""
from pathlib import Path
from renquant_backtesting.wf_gate import runner as r


def test_absolute_uri_passthrough(tmp_path):
    p = tmp_path / "x.json"; p.write_text("{}")
    assert r._manifest_uri_to_path(tmp_path / "m.json", str(p)) == p


def test_strategy_dir_relative_uri_does_not_double(tmp_path, monkeypatch):
    # manifest lives in <strat>/artifacts/sim, uri is strategy-dir-relative.
    strat = tmp_path / "strat"
    art = strat / "artifacts" / "walkforward_v2" / "2024-01-01" / "panel-ltr.json"
    art.parent.mkdir(parents=True); art.write_text("{}")
    monkeypatch.setattr(r, "STRATEGY_DIR", strat)
    manifest = strat / "artifacts" / "sim" / "m.json"
    manifest.parent.mkdir(parents=True)
    resolved = r._manifest_uri_to_path(manifest, "artifacts/walkforward_v2/2024-01-01/panel-ltr.json")
    assert resolved == art  # NOT artifacts/sim/artifacts/...
    assert "sim/artifacts" not in str(resolved)
