"""sim_driver frozen-input-bundle enforcement (model#79 round-3).

The guard must be enforced THROUGH the launched command, not via a
wrapper convention:

* precondition: a bundle mismatch aborts with exit 4 BEFORE
  ``run_backtest`` is ever called (before any data loading/scoring);
* postcondition: after the sim ran and outputs were written, a mutated
  covered input exits 6 (execution-time input mutation) — distinct
  from 4 so wrappers can tell "never ran" from "ran on mutated inputs";
* ``--input-bundle`` / ``--input-bundle-root`` / at least one
  ``--input-bundle-covered-root`` must be given together (argparse
  error otherwise);
* both guard verdicts are echoed to stdout for the wrapper's tee.

Reuses the fake ``sim.runner`` capture pattern from
``test_sim_driver_seed_plumb.py``.
"""
from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def _install_fake_run_backtest(monkeypatch, calls: list[dict], result,
                               side_effect=None):
    """Fake ``sim.runner`` whose run_backtest captures every kwarg."""

    def run_backtest(**kwargs):
        calls.append(dict(kwargs))
        if side_effect is not None:
            side_effect()
        return result

    module = types.ModuleType("sim.runner")
    module.run_backtest = run_backtest
    monkeypatch.setitem(sys.modules, module.__name__, module)


def _install_fake_fetch_ohlcv(monkeypatch):
    module = types.ModuleType("renquant_pipeline.kernel.data")
    module.fetch_ohlcv = lambda sym: pd.DataFrame()
    monkeypatch.setitem(sys.modules, module.__name__, module)


def _repo_root_with_bundle(tmp_path: Path) -> tuple[Path, Path, str]:
    """Tmp repo root (strategy dir + one frozen data file) + its bundle."""
    repo_root = tmp_path / "repo"
    strategy_dir = repo_root / "backtesting" / "renquant_104"
    strategy_dir.mkdir(parents=True)
    (strategy_dir / "strategy_config.json").write_text("{}")

    blob = b"frozen-prices"
    data_file = repo_root / "data" / "ohlcv" / "prices.csv"
    data_file.parent.mkdir(parents=True)
    data_file.write_bytes(blob)

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = (f"{hashlib.sha256(blob).hexdigest()}  {len(blob)}  "
                f"data/ohlcv/prices.csv\n")
    (bundle / "MANIFEST.sha256").write_text(manifest)
    root = hashlib.sha256(manifest.encode()).hexdigest()
    (bundle / "ROOT_DIGEST").write_text(root + "\n")
    return repo_root, bundle, root


def _bundle_argv(bundle: Path, root: str) -> list[str]:
    return ["--input-bundle", str(bundle),
            "--input-bundle-root", root,
            "--input-bundle-covered-root", "data/ohlcv"]


def _run_sim_driver(monkeypatch, repo_root: Path, argv: list[str]):
    from renquant_backtesting.wf_gate import sim_driver

    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(
        sys, "argv",
        ["sim_driver", "--repo-root", str(repo_root), *argv],
    )
    sim_driver.main()


def test_preflight_mismatch_exits_4_before_run_backtest(
    monkeypatch, tmp_path: Path, capsys,
) -> None:
    repo_root, bundle, root = _repo_root_with_bundle(tmp_path)
    (repo_root / "data" / "ohlcv" / "prices.csv").write_bytes(b"tampered")
    calls: list[dict] = []
    _install_fake_run_backtest(
        monkeypatch, calls, SimpleNamespace(print_summary=lambda: None),
    )
    _install_fake_fetch_ohlcv(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        _run_sim_driver(
            monkeypatch, repo_root,
            [*_bundle_argv(bundle, root), "--no-compare", "--no-persist"],
        )

    assert exc.value.code == 4
    assert calls == [], "run_backtest must never be called on a void bundle"
    out = capsys.readouterr().out
    assert "VOID digest mismatch: data/ohlcv/prices.csv" in out
    assert "INPUT BUNDLE PREFLIGHT FAILED: 1 mismatch(es)" in out


def test_post_run_mutation_exits_6_after_run_backtest(
    monkeypatch, tmp_path: Path, capsys,
) -> None:
    """Preflight passes, the sim itself mutates a covered input -> exit 6."""
    repo_root, bundle, root = _repo_root_with_bundle(tmp_path)
    calls: list[dict] = []
    _install_fake_run_backtest(
        monkeypatch, calls, SimpleNamespace(print_summary=lambda: None),
        side_effect=lambda: (repo_root / "data" / "ohlcv" / "prices.csv")
        .write_bytes(b"mutated during execution"),
    )
    _install_fake_fetch_ohlcv(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        _run_sim_driver(
            monkeypatch, repo_root,
            [*_bundle_argv(bundle, root), "--no-compare", "--no-persist"],
        )

    assert exc.value.code == 6
    assert len(calls) == 1, "the sim DID run; the postcondition caught it"
    out = capsys.readouterr().out
    assert "INPUT BUNDLE PREFLIGHT OK: root=" + root in out
    assert "VOID digest mismatch: data/ohlcv/prices.csv" in out
    assert "INPUT BUNDLE POST-RUN FAILED (execution-time input mutation): " \
           "1 mismatch(es)" in out


def test_clean_bundle_runs_and_echoes_both_verdicts(
    monkeypatch, tmp_path: Path, capsys,
) -> None:
    repo_root, bundle, root = _repo_root_with_bundle(tmp_path)
    calls: list[dict] = []
    _install_fake_run_backtest(
        monkeypatch, calls, SimpleNamespace(print_summary=lambda: None),
    )
    _install_fake_fetch_ohlcv(monkeypatch)

    _run_sim_driver(
        monkeypatch, repo_root,
        [*_bundle_argv(bundle, root), "--no-compare", "--no-persist"],
    )

    assert len(calls) == 1
    out = capsys.readouterr().out
    assert f"INPUT BUNDLE PREFLIGHT OK: root={root}" in out
    assert f"INPUT BUNDLE POST-RUN OK: root={root}" in out


def test_sim_outputs_outside_covered_roots_do_not_void_post_run(
    monkeypatch, tmp_path: Path, capsys,
) -> None:
    """The real-tree field-test scenario: a successful sim writes its own
    outputs (provenance JSONL, per-seed DB) OUTSIDE the covered roots;
    the post-run check must stay clean."""
    repo_root, bundle, root = _repo_root_with_bundle(tmp_path)

    def write_sim_outputs():
        prov = repo_root / "data" / "wf_provenance"
        prov.mkdir(parents=True)
        (prov / "wfsim-run.jsonl").write_text("{}\n")
        (repo_root / "data" / "sim_runs_101.db").write_bytes(b"sqlite")

    calls: list[dict] = []
    _install_fake_run_backtest(
        monkeypatch, calls, SimpleNamespace(print_summary=lambda: None),
        side_effect=write_sim_outputs,
    )
    _install_fake_fetch_ohlcv(monkeypatch)

    _run_sim_driver(
        monkeypatch, repo_root,
        [*_bundle_argv(bundle, root), "--no-compare", "--no-persist"],
    )

    assert len(calls) == 1
    assert f"INPUT BUNDLE POST-RUN OK: root={root}" in capsys.readouterr().out


@pytest.mark.parametrize("partial_argv", [
    ["--input-bundle", "somewhere"],
    ["--input-bundle-root", "0" * 64],
    ["--input-bundle-covered-root", "data/ohlcv"],
    ["--input-bundle", "somewhere", "--input-bundle-root", "0" * 64],
])
def test_bundle_flags_must_be_given_together(
    monkeypatch, tmp_path: Path, partial_argv: list[str],
) -> None:
    """Any partial bundle-flag combination — including --input-bundle +
    --input-bundle-root without a covered root — is an argparse error."""
    repo_root, _bundle, _root = _repo_root_with_bundle(tmp_path)
    calls: list[dict] = []
    _install_fake_run_backtest(
        monkeypatch, calls, SimpleNamespace(print_summary=lambda: None),
    )
    _install_fake_fetch_ohlcv(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        _run_sim_driver(
            monkeypatch, repo_root,
            [*partial_argv, "--no-compare", "--no-persist"],
        )

    assert exc.value.code == 2  # argparse error
    assert calls == []
