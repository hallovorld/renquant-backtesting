"""Regime-series injection seam (orch#962 B2) — unit contract.

The seam wraps the umbrella's ``kernel.pipeline.task_regime.RegimeFinalizeTask``
so a caller-supplied per-date label series REPLACES the computed ``ctx.regime``
before any downstream decision task reads it (BEAR-exit prereg placebo/shift
arms). Pinned here:

* loading: CSV + JSON, sha256 of the exact source bytes, structural defects
  are hard errors;
* injection replaces labels AND flips a downstream decision on a crafted
  fixture (the seam demonstrably reaches decision logic — the 5/15 no-op
  lesson);
* an injected series missing a bar date ERRORS instead of silently falling
  back to the computed regime (guards-validate-the-wrong-object lesson);
* install is fail-closed (no kernel module / no anchor class / double
  install / label outside kernel.config.REGIMES) and reversible;
* importing the module has no side effects — the invariance leg lives in
  ``test_sim_driver_regime_injection.py``.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from renquant_backtesting.wf_gate.regime_injection import (
    RegimeInjectionError,
    RegimeSeries,
    date_key,
    install_regime_injection,
    load_regime_series,
    required_sim_dates,
    uncovered_dates,
    uninstall_regime_injection,
)

D1, D2, D3 = "2020-03-02", "2020-03-03", "2020-03-04"


# ── fixtures ─────────────────────────────────────────────────────────────────


def _install_fake_task_regime(monkeypatch, computed: str = "BULL_CALM"):
    """Fake ``kernel.pipeline.task_regime`` whose finalize commits ``computed``."""
    module = types.ModuleType("kernel.pipeline.task_regime")

    class RegimeFinalizeTask:
        def run(self, ctx):
            ctx.regime = computed
            ctx.final_regime = computed
            counts = getattr(ctx, "regime_counts", None)
            if isinstance(counts, dict):
                counts[computed] = counts.get(computed, 0) + 1
            return True

    module.RegimeFinalizeTask = RegimeFinalizeTask
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def _install_fake_kernel_config(monkeypatch, regimes):
    module = types.ModuleType("kernel.config")
    module.REGIMES = list(regimes)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


class _Ctx:
    """Minimal InferenceContext stand-in for the bar loop."""

    def __init__(self, today):
        self.today = today
        self.regime = None
        self.final_regime = None
        self.regime_counts: dict = {}
        self._regime_evidence: dict = {}


class _PanelExitStub:
    """Downstream decision task keyed on ``ctx.regime`` (BEAR -> exit)."""

    def __init__(self):
        self.fired_on: list[str] = []

    def run(self, ctx):
        if ctx.regime == "BEAR":
            self.fired_on.append(ctx.today.isoformat())
        return True


def _bar_loop(task_mod, dates):
    """Run finalize + exit-stub over ``dates``; return (exit task, contexts)."""
    finalize = task_mod.RegimeFinalizeTask()
    exit_task = _PanelExitStub()
    contexts = []
    for d in dates:
        ctx = _Ctx(dt.date.fromisoformat(d))
        finalize.run(ctx)
        exit_task.run(ctx)
        contexts.append(ctx)
    return exit_task, contexts


def _series(mapping, sha: str = "f" * 64, source: str = "mem://series"):
    return RegimeSeries(by_date=dict(mapping), sha256=sha, source=source)


@pytest.fixture(autouse=True)
def _always_uninstall():
    yield
    uninstall_regime_injection()


# ── loading ──────────────────────────────────────────────────────────────────


def test_load_csv_series_with_header_and_sha256(tmp_path: Path) -> None:
    body = f"date,regime\n{D1},BEAR\n{D2},BULL_CALM\n"
    p = tmp_path / "series.csv"
    p.write_text(body)

    series = load_regime_series(p)

    assert series.by_date == {D1: "BEAR", D2: "BULL_CALM"}
    assert series.sha256 == hashlib.sha256(body.encode()).hexdigest()
    assert series.source == str(p)


def test_load_csv_series_without_header(tmp_path: Path) -> None:
    p = tmp_path / "series.csv"
    p.write_text(f"{D1},BEAR\n{D2},CHOPPY\n")

    assert load_regime_series(p).by_date == {D1: "BEAR", D2: "CHOPPY"}


def test_load_json_series_and_sha256(tmp_path: Path) -> None:
    body = json.dumps({D1: "BEAR", D3: "BULL_VOLATILE"})
    p = tmp_path / "series.json"
    p.write_text(body)

    series = load_regime_series(p)

    assert series.by_date == {D1: "BEAR", D3: "BULL_VOLATILE"}
    assert series.sha256 == hashlib.sha256(body.encode()).hexdigest()


@pytest.mark.parametrize("name,body", [
    ("dup.csv", f"{D1},BEAR\n{D1},CHOPPY\n"),
    ("baddate.csv", "not-a-date,BEAR\nalso-bad,CHOPPY\n"),
    ("badheader.csv", f"when,which\n{D1},BEAR\n"),
    ("threecol.csv", f"{D1},BEAR,extra\n"),
    ("empty.csv", ""),
    ("emptylabel.json", json.dumps({D1: ""})),
    ("nonobject.json", json.dumps([[D1, "BEAR"]])),
    ("badjson.json", "{not json"),
    ("empty.json", "{}"),
    ("wrongext.txt", f"{D1},BEAR\n"),
])
def test_load_rejects_structural_defects(
    tmp_path: Path, name: str, body: str,
) -> None:
    p = tmp_path / name
    p.write_text(body)
    with pytest.raises(RegimeInjectionError):
        load_regime_series(p)


def test_load_missing_file_errors(tmp_path: Path) -> None:
    with pytest.raises(RegimeInjectionError):
        load_regime_series(tmp_path / "absent.csv")


def test_date_key_normalizes_common_bar_date_types() -> None:
    import pandas as pd

    assert date_key(dt.date(2020, 3, 2)) == D1
    assert date_key(dt.datetime(2020, 3, 2, 15, 30)) == D1
    assert date_key(pd.Timestamp("2020-03-02")) == D1
    assert date_key("2020-03-02") == D1
    with pytest.raises(RegimeInjectionError):
        date_key(20200302)


def test_required_and_uncovered_dates() -> None:
    import pandas as pd

    spy = pd.DataFrame(
        {"close": [1.0, 1.0, 1.0]}, index=pd.to_datetime([D1, D2, D3]),
    )
    assert required_sim_dates(spy, D1, D2) == [D1, D2]
    assert required_sim_dates(pd.DataFrame(), D1, D3) == []
    assert required_sim_dates(None, D1, D3) == []
    series = _series({D1: "BEAR"})
    assert uncovered_dates(series, [D1, D2, D3]) == [D2, D3]


# ── injection semantics ──────────────────────────────────────────────────────


def test_injection_replaces_labels_and_flips_decision(monkeypatch) -> None:
    """The crafted fixture: computed = BULL_CALM every bar (no exits);
    injected BEAR on D1/D3 must flip the downstream decision path."""
    task_mod = _install_fake_task_regime(monkeypatch, computed="BULL_CALM")

    baseline_exit, baseline_ctxs = _bar_loop(task_mod, [D1, D2, D3])
    assert baseline_exit.fired_on == []
    assert [c.regime for c in baseline_ctxs] == ["BULL_CALM"] * 3

    installed = install_regime_injection(
        _series({D1: "BEAR", D2: "BULL_CALM", D3: "BEAR"}))
    injected_exit, injected_ctxs = _bar_loop(task_mod, [D1, D2, D3])

    assert injected_exit.fired_on == [D1, D3], \
        "injection must reach downstream decision logic, not just the label"
    assert [c.regime for c in injected_ctxs] == ["BEAR", "BULL_CALM", "BEAR"]
    assert [c.final_regime for c in injected_ctxs] == \
        ["BEAR", "BULL_CALM", "BEAR"]
    # regime_counts follow the injected series, not the computed one.
    assert injected_ctxs[0].regime_counts == {"BULL_CALM": 0, "BEAR": 1}
    # Evidence trail records both sides + the series identity.
    ev = injected_ctxs[0]._regime_evidence["regime_injection"]
    assert ev == {
        "active": True,
        "sha256": installed.series.sha256,
        "computed_regime": "BULL_CALM",
        "injected_regime": "BEAR",
    }


def test_missing_date_fails_closed_not_computed_fallback(monkeypatch) -> None:
    task_mod = _install_fake_task_regime(monkeypatch, computed="BULL_CALM")
    install_regime_injection(_series({D1: "BEAR"}))

    finalize = task_mod.RegimeFinalizeTask()
    exit_task = _PanelExitStub()

    ok_ctx = _Ctx(dt.date.fromisoformat(D1))
    finalize.run(ok_ctx)
    assert ok_ctx.regime == "BEAR"

    bad_ctx = _Ctx(dt.date.fromisoformat(D2))
    with pytest.raises(RegimeInjectionError, match=D2):
        finalize.run(bad_ctx)
        exit_task.run(bad_ctx)  # never reached — the bar dies at finalize
    assert exit_task.fired_on == [], \
        "the bar must die at finalize — no decision may run on the computed " \
        "fallback"
    assert "regime_injection" not in bad_ctx._regime_evidence, \
        "no injection record may be stamped for an uncovered date"


def test_context_without_today_fails_closed(monkeypatch) -> None:
    task_mod = _install_fake_task_regime(monkeypatch)
    install_regime_injection(_series({D1: "BEAR"}))
    ctx = _Ctx(dt.date.fromisoformat(D1))
    ctx.today = None
    with pytest.raises(RegimeInjectionError, match="today"):
        task_mod.RegimeFinalizeTask().run(ctx)


# ── install / uninstall lifecycle ────────────────────────────────────────────


def test_install_without_kernel_module_errors(monkeypatch) -> None:
    monkeypatch.delitem(
        sys.modules, "kernel.pipeline.task_regime", raising=False)
    with pytest.raises(RegimeInjectionError, match="kernel.pipeline.task_regime"):
        install_regime_injection(_series({D1: "BEAR"}))


def test_install_without_anchor_class_errors(monkeypatch) -> None:
    module = types.ModuleType("kernel.pipeline.task_regime")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(RegimeInjectionError, match="RegimeFinalizeTask"):
        install_regime_injection(_series({D1: "BEAR"}))


def test_double_install_refused_and_uninstall_restores(monkeypatch) -> None:
    task_mod = _install_fake_task_regime(monkeypatch)
    original_run = task_mod.RegimeFinalizeTask.run

    install_regime_injection(_series({D1: "BEAR"}))
    assert getattr(task_mod.RegimeFinalizeTask.run, "__regime_injection__", False)
    with pytest.raises(RegimeInjectionError, match="already installed"):
        install_regime_injection(_series({D1: "BEAR"}))

    uninstall_regime_injection()
    assert task_mod.RegimeFinalizeTask.run is original_run
    uninstall_regime_injection()  # idempotent


def test_label_outside_kernel_regimes_errors(monkeypatch) -> None:
    _install_fake_task_regime(monkeypatch)
    _install_fake_kernel_config(
        monkeypatch, ["BULL_CALM", "BULL_VOLATILE", "CHOPPY", "BEAR"])
    with pytest.raises(RegimeInjectionError, match="NOT_A_REGIME"):
        install_regime_injection(_series({D1: "NOT_A_REGIME"}))


def test_metadata_records_sha_and_label_validation_state(
    monkeypatch, tmp_path: Path,
) -> None:
    _install_fake_task_regime(monkeypatch)
    _install_fake_kernel_config(
        monkeypatch, ["BULL_CALM", "BULL_VOLATILE", "CHOPPY", "BEAR"])
    body = f"{D1},BEAR\n"
    p = tmp_path / "series.csv"
    p.write_text(body)

    installed = install_regime_injection(load_regime_series(p))

    assert installed.metadata() == {
        "active": True,
        "sha256": hashlib.sha256(body.encode()).hexdigest(),
        "source": str(p),
        "n_dates": 1,
        "label_set_validated": True,
    }


def test_metadata_says_when_label_set_was_not_validatable(monkeypatch) -> None:
    _install_fake_task_regime(monkeypatch)
    monkeypatch.delitem(sys.modules, "kernel.config", raising=False)
    installed = install_regime_injection(_series({D1: "BEAR"}))
    assert installed.metadata()["label_set_validated"] is False
