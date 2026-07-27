"""WF provenance seed plumb (pipeline#215 §3, umbrella#531).

``sim.runner.run_backtest`` records its ``seed`` kwarg in the WF sim-time
provenance sink (one JSONL per sim run; ``sim_run_id`` is minted inside
``run_backtest`` — the drivers pass nothing else). These tests pin the
contract that BOTH wf_gate sim drivers forward their CLI seed into
``run_backtest`` and never drop it on the way:

* ``sim_driver`` (run_sim_104): ``--seed`` reaches the candidate leg AND
  the golden comparison leg (same seed → paired A/B).
* ``dump_walkforward_sim_metrics``: ``--seed`` reaches its single
  ``run_backtest`` call and is echoed into the metrics JSON.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def _install_fake_run_backtest(monkeypatch, calls: list[dict], result):
    """Fake ``sim.runner`` whose run_backtest captures every kwarg."""

    def run_backtest(**kwargs):
        calls.append(dict(kwargs))
        return result

    module = types.ModuleType("sim.runner")
    module.run_backtest = run_backtest
    monkeypatch.setitem(sys.modules, module.__name__, module)


def _install_fake_fetch_ohlcv(monkeypatch):
    module = types.ModuleType("renquant_pipeline.kernel.data")
    module.fetch_ohlcv = lambda sym: pd.DataFrame()
    monkeypatch.setitem(sys.modules, module.__name__, module)


def _install_fake_risk_metrics(monkeypatch):
    module = types.ModuleType("renquant_common.risk_metrics")
    module.compute_risk_metrics = lambda *a, **k: {}
    module.daily_returns_from_equity = lambda s: s
    module.geometric_sharpe_ratio = lambda *a, **k: float("nan")
    monkeypatch.setitem(sys.modules, module.__name__, module)


def _sim_driver_strategy_dir(tmp_path: Path) -> Path:
    strategy_dir = tmp_path / "backtesting" / "renquant_104"
    strategy_dir.mkdir(parents=True)
    (strategy_dir / "strategy_config.json").write_text("{}")
    return strategy_dir


def _run_sim_driver(monkeypatch, tmp_path: Path, argv: list[str]):
    from renquant_backtesting.wf_gate import sim_driver

    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(
        sys, "argv",
        ["sim_driver", "--repo-root", str(tmp_path), *argv],
    )
    sim_driver.main()


def test_sim_driver_passes_seed_to_run_backtest(monkeypatch, tmp_path: Path) -> None:
    _sim_driver_strategy_dir(tmp_path)
    calls: list[dict] = []
    _install_fake_run_backtest(
        monkeypatch, calls, SimpleNamespace(print_summary=lambda: None),
    )
    _install_fake_fetch_ohlcv(monkeypatch)

    _run_sim_driver(
        monkeypatch, tmp_path,
        ["--seed", "7", "--no-compare", "--no-persist"],
    )

    assert len(calls) == 1
    assert calls[0]["seed"] == 7


def test_sim_driver_default_seed_is_forwarded_as_none(
    monkeypatch, tmp_path: Path,
) -> None:
    """No --seed → run_backtest still receives an explicit seed=None."""
    _sim_driver_strategy_dir(tmp_path)
    calls: list[dict] = []
    _install_fake_run_backtest(
        monkeypatch, calls, SimpleNamespace(print_summary=lambda: None),
    )
    _install_fake_fetch_ohlcv(monkeypatch)

    _run_sim_driver(monkeypatch, tmp_path, ["--no-compare", "--no-persist"])

    assert len(calls) == 1
    assert "seed" in calls[0]
    assert calls[0]["seed"] is None


def test_sim_driver_golden_leg_gets_same_seed(monkeypatch, tmp_path: Path) -> None:
    strategy_dir = _sim_driver_strategy_dir(tmp_path)
    (strategy_dir / "strategy_config.golden.json").write_text("{}")
    fake_result = SimpleNamespace(
        print_summary=lambda: None, apy=0.1, win_rate=0.5, buys=[],
    )
    calls: list[dict] = []
    _install_fake_run_backtest(monkeypatch, calls, fake_result)
    _install_fake_fetch_ohlcv(monkeypatch)

    _run_sim_driver(monkeypatch, tmp_path, ["--seed", "7", "--no-persist"])

    assert len(calls) == 2, "candidate + golden comparison legs"
    assert [c["seed"] for c in calls] == [7, 7]


def test_dump_walkforward_sim_metrics_passes_seed(
    monkeypatch, tmp_path: Path,
) -> None:
    from renquant_backtesting.wf_gate import dump_walkforward_sim_metrics as dump

    strategy_dir = tmp_path / "backtesting" / "renquant_104"
    strategy_dir.mkdir(parents=True)
    (strategy_dir / "strategy_config.sim_baseline.json").write_text("{}")
    fake_result = SimpleNamespace(
        print_summary=lambda: None,
        equity_df=pd.DataFrame(),
        final_value=100_000.0,
        total_return=0.0,
        apy=0.0,
        sharpe=float("nan"),
        sortino=float("nan"),
        calmar=float("nan"),
        max_dd=float("nan"),
        ann_vol=float("nan"),
        dsr=float("nan"),
        pbo=float("nan"),
        n_trials=1,
        beta_vs_spy=float("nan"),
        alpha_vs_spy=float("nan"),
        information_ratio_vs_spy=float("nan"),
        buys=[],
        sells=[],
        win_rate=float("nan"),
        avg_hold=float("nan"),
        avg_pnl=float("nan"),
        total_tax=0.0,
        exit_reasons={},
        longest_no_trade_streak=0,
        first_trade_date=None,
        last_activity_date=None,
    )
    calls: list[dict] = []
    _install_fake_run_backtest(monkeypatch, calls, fake_result)
    _install_fake_fetch_ohlcv(monkeypatch)
    _install_fake_risk_metrics(monkeypatch)

    out_path = tmp_path / "metrics.json"
    monkeypatch.setattr(dump, "REPO", tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(
        sys, "argv",
        ["dump_walkforward_sim_metrics", "--seed", "11", "--out", str(out_path)],
    )
    dump.main()

    assert len(calls) == 1
    assert calls[0]["seed"] == 11
    # Persistence forced OFF is the documented §2.1 persisted:false path —
    # it must not eat the seed.
    assert calls[0]["config"]["persistence"] == {"enabled": False}
    assert json.loads(out_path.read_text())["seed"] == 11
