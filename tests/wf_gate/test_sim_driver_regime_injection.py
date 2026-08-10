"""sim_driver ``--regime-series`` seam (orch#962 B2) — driver contract.

The BEAR-exit prereg's confirmatory arms (orch
``doc/design/2026-08-08-bear-exit-prereg.md`` §3: 200-seed episode-permutation
placebos, +5/+10/+20d shifts) need book sims under an INJECTED regime series.
Pinned here, through the launched command:

* ``--regime-series`` installs the seam (RegimeFinalizeTask patched) and the
  equity-JSON metadata records {active, sha256, source, n_dates,
  label_set_validated}; the ACTIVE verdict is echoed to stdout for the
  wrapper's tee;
* a sim bar date not covered by the series aborts with exit 5 BEFORE
  ``run_backtest`` (fail-closed preflight; the per-bar guard backstops);
* the golden comparison leg runs under the SAME injected series (paired
  arms — one class-level install covers both ``run_backtest`` calls);
* flag absent = the frozen pre-feature golden: RegimeFinalizeTask untouched,
  the config handed to ``run_backtest`` byte-equal to the legacy decoration,
  the equity payload's key set unchanged, no injection line on stdout
  (behavior-invariance requirement of orch#962 B2).

Reuses the fake ``sim.runner`` capture pattern from
``test_sim_driver_seed_plumb.py``.
"""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from renquant_backtesting.wf_gate.regime_injection import (
    uninstall_regime_injection,
)

D1, D2, D3 = "2020-03-02", "2020-03-03", "2020-03-04"

# The pre-feature equity-JSON key set (sim_driver payload literal @ a2dcda9).
# The invariance test pins that a flag-absent run adds NOTHING to it.
GOLDEN_PAYLOAD_KEYS = {
    "config", "start", "end", "initial_cash", "final_value", "total_return",
    "apy", "sharpe", "event_level_apy", "event_level_sharpe",
    "event_level_tax_debited", "event_level_tax_estimate", "tax_cash_debited",
    "tax_cash_debit_mode", "annual_net_tax_estimate",
    "tax_overstatement_vs_annual_net", "annual_net_final_value",
    "annual_net_total_return", "annual_net_apy", "annual_net_sharpe",
    "annual_net_ann_vol", "annual_net_max_dd", "ann_vol", "max_dd", "equity",
}


@pytest.fixture(autouse=True)
def _always_uninstall():
    yield
    uninstall_regime_injection()


def _install_fake_run_backtest(monkeypatch, calls: list[dict], result):
    def run_backtest(**kwargs):
        calls.append(dict(kwargs))
        return result

    module = types.ModuleType("sim.runner")
    module.run_backtest = run_backtest
    monkeypatch.setitem(sys.modules, module.__name__, module)


def _install_fake_fetch_ohlcv(monkeypatch, spy_df: pd.DataFrame):
    module = types.ModuleType("renquant_pipeline.kernel.data")
    module.fetch_ohlcv = lambda sym: spy_df
    monkeypatch.setitem(sys.modules, module.__name__, module)


def _install_fake_task_regime(monkeypatch):
    module = types.ModuleType("kernel.pipeline.task_regime")

    class RegimeFinalizeTask:
        def run(self, ctx):
            ctx.regime = "BULL_CALM"
            return True

    module.RegimeFinalizeTask = RegimeFinalizeTask
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def _install_fake_kernel_config(monkeypatch):
    module = types.ModuleType("kernel.config")
    module.REGIMES = ["BULL_CALM", "BULL_VOLATILE", "CHOPPY", "BEAR"]
    monkeypatch.setitem(sys.modules, module.__name__, module)


def _spy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"close": [100.0, 101.0, 102.0]},
        index=pd.to_datetime([D1, D2, D3]),
    )


def _full_fake_result() -> SimpleNamespace:
    eq = pd.DataFrame(
        {"portfolio": [100_000.0, 100_500.0, 101_000.0]},
        index=pd.to_datetime([D1, D2, D3]),
    )
    nan = float("nan")
    return SimpleNamespace(
        print_summary=lambda: None,
        equity_df=eq,
        final_value=101_000.0,
        total_return=0.01,
        apy=0.05,
        sharpe=nan,
        event_level_tax_debited=0.0,
        event_level_tax_estimate=0.0,
        tax_cash_debited=0.0,
        tax_cash_debit_mode="event_level",
        annual_net_tax_estimate=0.0,
        tax_overstatement_vs_annual_net=0.0,
        annual_net_final_value_estimate=nan,
        annual_net_total_return_estimate=nan,
        annual_net_apy_estimate=nan,
        annual_net_sharpe_estimate=nan,
        annual_net_ann_vol_estimate=nan,
        annual_net_max_dd_estimate=nan,
        annual_net_equity_df_estimate=pd.DataFrame(),
        ann_vol=nan,
        max_dd=nan,
        win_rate=0.5,
        buys=[],
    )


def _strategy_dir(tmp_path: Path) -> Path:
    strategy_dir = tmp_path / "backtesting" / "renquant_104"
    strategy_dir.mkdir(parents=True)
    (strategy_dir / "strategy_config.json").write_text("{}")
    return strategy_dir


def _series_csv(tmp_path: Path, mapping: dict) -> tuple[Path, str]:
    body = "".join(f"{d},{label}\n" for d, label in mapping.items())
    p = tmp_path / "injected_series.csv"
    p.write_text(body)
    return p, hashlib.sha256(body.encode()).hexdigest()


def _run_sim_driver(monkeypatch, tmp_path: Path, argv: list[str]):
    from renquant_backtesting.wf_gate import sim_driver

    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(
        sys, "argv",
        ["sim_driver", "--repo-root", str(tmp_path),
         "--start", D1, "--end", D3, *argv],
    )
    sim_driver.main()


def test_injection_active_records_metadata_and_patches_seam(
    monkeypatch, tmp_path: Path, capsys,
) -> None:
    _strategy_dir(tmp_path)
    task_mod = _install_fake_task_regime(monkeypatch)
    _install_fake_kernel_config(monkeypatch)
    calls: list[dict] = []
    _install_fake_run_backtest(monkeypatch, calls, _full_fake_result())
    _install_fake_fetch_ohlcv(monkeypatch, _spy_df())
    series_path, sha = _series_csv(
        tmp_path, {D1: "BEAR", D2: "BULL_CALM", D3: "BEAR"})
    equity_json = tmp_path / "out" / "equity.json"

    _run_sim_driver(
        monkeypatch, tmp_path,
        ["--regime-series", str(series_path),
         "--equity-json", str(equity_json),
         "--no-compare", "--no-persist"],
    )

    assert len(calls) == 1
    assert getattr(
        task_mod.RegimeFinalizeTask.run, "__regime_injection__", False,
    ), "the seam must be installed on the class the pipeline will call"
    payload = json.loads(equity_json.read_text())
    assert payload["regime_injection"] == {
        "active": True,
        "sha256": sha,
        "source": str(series_path),
        "n_dates": 3,
        "label_set_validated": True,
    }
    out = capsys.readouterr().out
    assert f"REGIME INJECTION ACTIVE: sha256={sha} n_dates=3" in out


def test_uncovered_sim_date_exits_5_before_run_backtest(
    monkeypatch, tmp_path: Path, capsys,
) -> None:
    _strategy_dir(tmp_path)
    _install_fake_task_regime(monkeypatch)
    _install_fake_kernel_config(monkeypatch)
    calls: list[dict] = []
    _install_fake_run_backtest(monkeypatch, calls, _full_fake_result())
    _install_fake_fetch_ohlcv(monkeypatch, _spy_df())
    series_path, _sha = _series_csv(tmp_path, {D1: "BEAR", D2: "BULL_CALM"})

    with pytest.raises(SystemExit) as exc:
        _run_sim_driver(
            monkeypatch, tmp_path,
            ["--regime-series", str(series_path),
             "--no-compare", "--no-persist"],
        )

    assert exc.value.code == 5
    assert calls == [], "run_backtest must never run on an uncovered series"
    out = capsys.readouterr().out
    assert ("REGIME INJECTION PREFLIGHT FAILED: 1 sim date(s) not covered"
            in out)
    assert D3 in out


def test_golden_comparison_leg_runs_under_the_same_injection(
    monkeypatch, tmp_path: Path,
) -> None:
    strategy_dir = _strategy_dir(tmp_path)
    (strategy_dir / "strategy_config.golden.json").write_text("{}")
    task_mod = _install_fake_task_regime(monkeypatch)
    _install_fake_kernel_config(monkeypatch)
    calls: list[dict] = []
    _install_fake_run_backtest(monkeypatch, calls, _full_fake_result())
    _install_fake_fetch_ohlcv(monkeypatch, _spy_df())
    series_path, _sha = _series_csv(
        tmp_path, {D1: "BEAR", D2: "BULL_CALM", D3: "BEAR"})

    _run_sim_driver(
        monkeypatch, tmp_path,
        ["--regime-series", str(series_path), "--no-persist"],
    )

    assert len(calls) == 2, "candidate + golden comparison legs"
    assert getattr(
        task_mod.RegimeFinalizeTask.run, "__regime_injection__", False,
    ), "one process-wide install covers BOTH legs (paired arms, same series)"


def test_without_flag_behavior_matches_the_pre_feature_golden(
    monkeypatch, tmp_path: Path, capsys,
) -> None:
    """Behavior invariance (orch#962 B2): with the feature merely available
    but the flag absent, the driver's behavioral surface is the frozen
    pre-feature golden — no patch, legacy config decoration byte-equal, the
    payload key set unchanged, nothing injection-related on stdout."""
    strategy_dir = _strategy_dir(tmp_path)
    task_mod = _install_fake_task_regime(monkeypatch)
    original_run = task_mod.RegimeFinalizeTask.run
    calls: list[dict] = []
    _install_fake_run_backtest(monkeypatch, calls, _full_fake_result())
    _install_fake_fetch_ohlcv(monkeypatch, _spy_df())
    equity_json = tmp_path / "out" / "equity.json"

    _run_sim_driver(
        monkeypatch, tmp_path,
        ["--equity-json", str(equity_json), "--no-compare", "--no-persist"],
    )

    assert task_mod.RegimeFinalizeTask.run is original_run, \
        "no --regime-series => RegimeFinalizeTask must be untouched"
    assert len(calls) == 1
    # The exact legacy config decoration — byte-equal, nothing added.
    assert calls[0]["config"] == {
        "_strategy_dir": str(strategy_dir),
        "_strategy_config_name": "strategy_config.json",
        "initial_cash": 100_000,
        "backtest_start": D1,
        "backtest_end": D3,
        "data_freshness": {"enabled": False},
        "persistence": {"enabled": False},
    }
    payload = json.loads(equity_json.read_text())
    assert set(payload.keys()) == GOLDEN_PAYLOAD_KEYS
    assert "regime_injection" not in payload
    assert "REGIME INJECTION" not in capsys.readouterr().out
