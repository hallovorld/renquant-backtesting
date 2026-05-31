"""Generate the latest-run simulation dashboard."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SEARCH_ROOTS = (
    Path("artifacts"),
    Path("reports"),
    Path("runs"),
    Path("../RenQuant/backtesting/renquant_104/artifacts"),
)


@dataclass
class LatestRun:
    source: Path
    mtime: float
    kind: str
    quality_score: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    cuts: list[dict[str, Any]] = field(default_factory=list)
    regimes: list[dict[str, Any]] = field(default_factory=list)
    trade_counts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def generate_latest_run_docs(
    *,
    search_roots: list[Path] | None = None,
    docs_dir: Path = Path("docs"),
    now: datetime | None = None,
) -> Path:
    """Write ``docs/latest-run.md`` and SVG assets for the newest run metrics."""
    now = now or datetime.now(timezone.utc)
    docs_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = docs_dir / "latest-run-assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    latest = find_latest_run(search_roots or list(DEFAULT_SEARCH_ROOTS))
    if latest is None:
        out = docs_dir / "latest-run.md"
        out.write_text(_empty_dashboard(now), encoding="utf-8")
        return out

    _write_svg_assets(latest, assets_dir)
    out = docs_dir / "latest-run.md"
    out.write_text(_dashboard_markdown(latest, now), encoding="utf-8")
    return out


def find_latest_run(search_roots: list[Path]) -> LatestRun | None:
    candidates: list[LatestRun] = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            run = _latest_run_from_path(path)
            if run is not None:
                candidates.append(run)
    if not candidates:
        return None
    selected, ignored_newest = _select_latest_run(candidates)
    if ignored_newest is not None:
        selected.warnings.append(
            "Newer lower-information artifact ignored: "
            f"`{_display_path(ignored_newest.source)}` "
            f"({_run_quality_reason(ignored_newest)})."
        )
    return selected


def _latest_run_from_path(path: Path) -> LatestRun | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(payload, dict):
        return None

    wf = _wf_gate_metadata(payload)
    if wf is not None:
        metrics = _wf_metrics(wf)
        if metrics:
            run = LatestRun(
                source=path,
                mtime=path.stat().st_mtime,
                kind="wf_gate",
                metrics=metrics,
                cuts=_wf_cuts(wf),
                regimes=_wf_regimes(wf),
                trade_counts=_trade_counts(wf),
            )
            return _finalize_run(run)

    equity = _equity_metrics(payload)
    if equity:
        trace = _trade_summary_for_equity_path(path)
        for key in ("n_buys", "n_sells", "n_trades"):
            if key not in equity and _is_number(trace.get(key)):
                equity[key] = trace[key]
        enriched_payload = dict(payload)
        enriched_payload.update(equity)
        run = LatestRun(
            source=path,
            mtime=path.stat().st_mtime,
            kind="equity_curve",
            metrics=equity,
            cuts=[_equity_cut(enriched_payload)],
            trade_counts=_trade_counts_from_trace(trace),
        )
        return _finalize_run(run)
    return None


def _wf_gate_metadata(payload: dict[str, Any]) -> dict[str, Any] | None:
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    wf = meta.get("wf_gate_metadata") if isinstance(meta.get("wf_gate_metadata"), dict) else None
    if wf is None and isinstance(payload.get("wf_gate_metadata"), dict):
        wf = payload["wf_gate_metadata"]
    if wf is None and any(k.startswith("wf_3cut_") for k in payload):
        wf = payload
    return wf


def _wf_metrics(wf: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "passed",
        "diagnostic_only",
        "wf_3cut_sharpe_mean",
        "wf_3cut_sharpe_std",
        "spy_sharpe_mean",
        "strategy_minus_spy_sharpe_mean",
        "wf_3cut_apy_mean",
        "spy_apy_mean",
        "strategy_minus_spy_apy_mean",
        "n_positive_cuts",
        "n_cuts_beat_spy_sharpe",
        "n_cuts_beat_spy_apy",
        "real_ic",
        "sanity_shuffled_ic",
        "sanity_placebo_ic",
    )
    out = {key: wf.get(key) for key in keys if key in wf}
    for nested_key in ("trade_contract", "trade_monotonicity", "alpha_economics", "sanity_regime_ic"):
        value = wf.get(nested_key)
        if isinstance(value, dict):
            out[f"{nested_key}_passed"] = value.get("passed")
            out[f"{nested_key}_reason"] = value.get("reason")
    if wf.get("wf_reason"):
        out["wf_reason"] = wf.get("wf_reason")
    if wf.get("run_at"):
        out["run_at"] = wf.get("run_at")
    return out


def _equity_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "apy",
        "sharpe",
        "event_level_apy",
        "event_level_sharpe",
        "annual_net_apy",
        "annual_net_sharpe",
        "max_drawdown",
        "max_dd",
        "annual_net_max_dd",
        "turnover_ratio",
        "total_return",
        "annual_net_total_return",
        "final_value",
        "annual_net_final_value",
        "n_trades",
        "n_buys",
        "n_sells",
    )
    return {key: payload.get(key) for key in keys if _is_number(payload.get(key))}


def _equity_cut(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": _cut_label(payload),
        "strategy_sharpe": _number(payload.get("annual_net_sharpe"), payload.get("sharpe")),
        "spy_sharpe": None,
        "strategy_apy": _number(payload.get("annual_net_apy"), payload.get("apy")),
        "spy_apy": None,
        "buys": _number(payload.get("n_buys"), payload.get("n_trades")),
        "sells": _number(payload.get("n_sells")),
        "regime": None,
    }


def _wf_cuts(wf: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cut in wf.get("cuts") or []:
        if not isinstance(cut, dict):
            continue
        market = cut.get("market_context") if isinstance(cut.get("market_context"), dict) else {}
        trades = cut.get("trade_trace_summary") if isinstance(cut.get("trade_trace_summary"), dict) else {}
        rows.append({
            "label": _cut_label(cut),
            "strategy_sharpe": _number(cut.get("annual_net_sharpe"), cut.get("sharpe")),
            "spy_sharpe": _number(market.get("spy_sharpe"), cut.get("spy_sharpe")),
            "strategy_apy": _number(cut.get("annual_net_apy"), cut.get("apy")),
            "spy_apy": _number(market.get("spy_apy"), cut.get("spy_apy")),
            "buys": _number(trades.get("n_buys")),
            "sells": _number(trades.get("n_sells")),
            "regime": cut.get("dominant_hmm_regime") or cut.get("dominant_spy_grid_regime"),
        })
    return rows


def _wf_regimes(wf: dict[str, Any]) -> list[dict[str, Any]]:
    regimes = wf.get("benchmark_by_dominant_regime")
    if not isinstance(regimes, dict):
        return []
    rows = []
    for name, stats in sorted(regimes.items()):
        if not isinstance(stats, dict):
            continue
        rows.append({
            "label": str(name),
            "strategy_sharpe": _number(stats.get("mean_sharpe")),
            "spy_sharpe": _number(stats.get("mean_spy_sharpe")),
            "strategy_apy": _number(stats.get("mean_apy")),
            "spy_apy": _number(stats.get("mean_spy_apy")),
        })
    return rows


def _trade_counts(wf: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "trade_buy_source_counts_total",
        "trade_sell_exit_reason_counts_total",
        "trade_buy_regime_counts_total",
        "trade_sell_regime_counts_total",
    ):
        if isinstance(wf.get(key), dict):
            out[key] = wf[key]
    return out


def _trade_summary_for_equity_path(path: Path) -> dict[str, Any]:
    """Load trade counts from sibling trade trace sidecars, when present."""
    base = _trace_base(path)
    if base is None:
        return {}
    out: dict[str, Any] = {}
    trades_path = base.with_name(base.name + ".trades.json")
    if trades_path.exists():
        try:
            rows = json.loads(trades_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            rows = None
        if isinstance(rows, list):
            buys = [
                row for row in rows
                if isinstance(row, dict) and row.get("action") == "buy"
            ]
            sells = [
                row for row in rows
                if isinstance(row, dict) and row.get("action") == "sell"
            ]
            out["n_buys"] = len(buys)
            out["n_sells"] = len(sells)
            out["n_trades"] = len(buys) + len(sells)
            out["trade_buy_source_counts_total"] = _count_rows(buys, "source_job")
            out["trade_sell_exit_reason_counts_total"] = _count_rows(sells, "exit_reason")
            out["trade_buy_regime_counts_total"] = _count_rows(buys, "regime")
            out["trade_sell_regime_counts_total"] = _count_rows(sells, "regime")

    report_path = base.with_name(base.name + ".report.md")
    if report_path.exists():
        try:
            report = report_path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            report = ""
        for key in ("n_buys", "n_sells"):
            if key not in out:
                value = _report_number(report, key)
                if value is not None:
                    out[key] = value
        if "n_trades" not in out and "n_buys" in out and "n_sells" in out:
            out["n_trades"] = int(out["n_buys"]) + int(out["n_sells"])
        if "No trade events recorded." in report:
            out.setdefault("n_buys", 0)
            out.setdefault("n_sells", 0)
            out.setdefault("n_trades", 0)
    return out


def _trace_base(path: Path) -> Path | None:
    suffix = ".equity.json"
    name = path.name
    if not name.endswith(suffix):
        return None
    return path.with_name(name[: -len(suffix)])


def _count_rows(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(str(row.get(key)) for row in rows if row.get(key) not in (None, ""))
    return dict(sorted(counts.items(), key=lambda item: str(item[0])))


def _report_number(report: str, key: str) -> int | None:
    match = re.search(rf"^- {re.escape(key)}:\s*([+-]?\d+)", report, flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def _trade_counts_from_trace(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        key: trace[key]
        for key in (
            "trade_buy_source_counts_total",
            "trade_sell_exit_reason_counts_total",
            "trade_buy_regime_counts_total",
            "trade_sell_regime_counts_total",
        )
        if isinstance(trace.get(key), dict) and trace[key]
    }


def _finalize_run(run: LatestRun) -> LatestRun:
    run.quality_score = _quality_score(run)
    if _is_no_trade_run(run):
        run.warnings.append(
            "Selected artifact has no trade events; Sharpe and trade diagnostics "
            "are expected to be sparse."
        )
    return run


def _selection_key(run: LatestRun) -> tuple[int, float, str]:
    return (run.quality_score, run.mtime, str(run.source))


def _select_latest_run(candidates: list[LatestRun]) -> tuple[LatestRun, LatestRun | None]:
    newest = max(candidates, key=lambda run: (run.mtime, str(run.source)))
    if not _is_lower_information_run(newest):
        return newest, None

    alternatives = [
        run for run in candidates
        if run.source != newest.source and run.quality_score > newest.quality_score
    ]
    if not alternatives:
        return newest, None
    return max(alternatives, key=_selection_key), newest


def _quality_score(run: LatestRun) -> int:
    score = 400 if run.kind == "wf_gate" else 100
    if _number(
        run.metrics.get("wf_3cut_sharpe_mean"),
        run.metrics.get("annual_net_sharpe"),
        run.metrics.get("sharpe"),
    ) is not None:
        score += 120
    if _number(
        run.metrics.get("spy_sharpe_mean"),
        run.metrics.get("spy_apy_mean"),
    ) is not None:
        score += 30
    if run.regimes:
        score += 30
    if run.trade_counts:
        score += 30
    total_trades = _total_trades(run)
    if total_trades is not None:
        score += min(int(total_trades), 100)
        if int(total_trades) == 0:
            score -= 175
    if _is_no_trade_run(run):
        score -= 75
    return score


def _total_trades(run: LatestRun) -> int | None:
    buys = run.metrics.get("n_buys")
    sells = run.metrics.get("n_sells")
    if _is_number(buys) and _is_number(sells):
        return int(buys) + int(sells)
    trades = run.metrics.get("n_trades")
    if _is_number(trades):
        return int(trades)
    return None


def _is_lower_information_run(run: LatestRun) -> bool:
    if _is_no_trade_run(run):
        return True
    has_performance_metric = _number(
        run.metrics.get("wf_3cut_sharpe_mean"),
        run.metrics.get("annual_net_sharpe"),
        run.metrics.get("sharpe"),
        run.metrics.get("wf_3cut_apy_mean"),
        run.metrics.get("annual_net_apy"),
        run.metrics.get("apy"),
        run.metrics.get("annual_net_total_return"),
        run.metrics.get("total_return"),
    ) is not None
    return not has_performance_metric and not run.regimes and not run.trade_counts


def _is_no_trade_run(run: LatestRun) -> bool:
    total_trades = _total_trades(run)
    if total_trades == 0:
        return True
    if total_trades is not None:
        return False
    sharpe = _number(
        run.metrics.get("wf_3cut_sharpe_mean"),
        run.metrics.get("annual_net_sharpe"),
        run.metrics.get("sharpe"),
    )
    total_return = _number(
        run.metrics.get("annual_net_total_return"),
        run.metrics.get("total_return"),
    )
    apy = _number(run.metrics.get("annual_net_apy"), run.metrics.get("apy"))
    return sharpe is None and total_return == 0.0 and apy == 0.0


def _run_quality_reason(run: LatestRun) -> str:
    reasons: list[str] = []
    if _is_no_trade_run(run):
        reasons.append("no trades")
    if _number(
        run.metrics.get("wf_3cut_sharpe_mean"),
        run.metrics.get("annual_net_sharpe"),
        run.metrics.get("sharpe"),
    ) is None:
        reasons.append("no finite Sharpe")
    if not run.trade_counts:
        reasons.append("no trade sidecar counts")
    return ", ".join(reasons) or f"quality_score={run.quality_score}"


def _write_svg_assets(run: LatestRun, assets_dir: Path) -> None:
    (assets_dir / "summary.svg").write_text(_summary_svg(run), encoding="utf-8")
    (assets_dir / "cuts.svg").write_text(_grouped_bar_svg(run.cuts, "Cut Performance"), encoding="utf-8")
    (assets_dir / "regimes.svg").write_text(_grouped_bar_svg(run.regimes, "Regime Performance"), encoding="utf-8")
    (assets_dir / "trades.svg").write_text(_trade_svg(run.trade_counts), encoding="utf-8")


def _dashboard_markdown(run: LatestRun, now: datetime) -> str:
    cards = _metric_table(run.metrics)
    cuts = _cut_table(run.cuts)
    regimes = _regime_table(run.regimes)
    trades = _trade_table(run.trade_counts)
    warnings = _warning_block(run.warnings)
    source = _display_path(run.source)
    generated = now.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        "# Latest Simulation Run",
        "",
        f"Generated: `{generated}`",
        f"Source: `{source}`",
        f"Detected format: `{run.kind}`",
        f"Selection quality score: `{run.quality_score}`",
        "",
    ]
    if warnings:
        lines.extend([
            "## Selection Notes",
            "",
            warnings,
            "",
        ])
    lines.extend([
        "## Scoreboard",
        "",
        "![Summary](latest-run-assets/summary.svg)",
        "",
        cards,
        "",
        "## Walk-Forward Cuts",
        "",
        "![Cut performance](latest-run-assets/cuts.svg)",
        "",
        cuts,
        "",
        "## Regime View",
        "",
        "![Regime performance](latest-run-assets/regimes.svg)",
        "",
        regimes,
        "",
        "## Trade Diagnostics",
        "",
        "![Trade diagnostics](latest-run-assets/trades.svg)",
        "",
        trades,
        "",
        "_Refresh with `make latest-report`._",
        "",
    ])
    return "\n".join(lines)


def _empty_dashboard(now: datetime) -> str:
    generated = now.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    return "\n".join([
        "# Latest Simulation Run",
        "",
        f"Generated: `{generated}`",
        "",
        "No simulation or walk-forward metric JSON was found under the configured search roots.",
        "",
        "Refresh after a run with `make latest-report`, or pass explicit roots:",
        "",
        "```bash",
        "python -m renquant_backtesting.reporting.latest_run_docs --root artifacts/diagnostics",
        "```",
        "",
    ])


def _metric_table(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "_No scalar metrics found._"
    rows = ["| Metric | Value |", "|---|---:|"]
    preferred = (
        "passed",
        "diagnostic_only",
        "wf_3cut_sharpe_mean",
        "spy_sharpe_mean",
        "strategy_minus_spy_sharpe_mean",
        "wf_3cut_apy_mean",
        "spy_apy_mean",
        "strategy_minus_spy_apy_mean",
        "n_positive_cuts",
        "n_cuts_beat_spy_sharpe",
        "n_cuts_beat_spy_apy",
        "real_ic",
        "sanity_placebo_ic",
        "trade_contract_passed",
        "trade_monotonicity_passed",
        "alpha_economics_passed",
        "annual_net_sharpe",
        "sharpe",
        "annual_net_apy",
        "apy",
        "annual_net_total_return",
        "total_return",
        "annual_net_max_dd",
        "max_dd",
        "turnover_ratio",
        "final_value",
        "n_buys",
        "n_sells",
        "n_trades",
    )
    rendered: set[str] = set()
    for key in preferred:
        if key in metrics:
            rows.append(f"| `{key}` | {_fmt(metrics[key], percent=_is_percent_metric(key))} |")
            rendered.add(key)
    for key in sorted(k for k in metrics if k not in rendered and not str(k).endswith("_reason")):
        rows.append(f"| `{key}` | {_fmt(metrics[key], percent=_is_percent_metric(str(key)))} |")
    if metrics.get("wf_reason"):
        rows.append(f"| `wf_reason` | {str(metrics['wf_reason'])} |")
    return "\n".join(rows)


def _cut_table(cuts: list[dict[str, Any]]) -> str:
    if not cuts:
        return "_No cut-level metrics found._"
    rows = ["| Cut | Sharpe | SPY Sharpe | APY | SPY APY | Buys | Sells |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in cuts:
        rows.append(
            f"| {row['label']} | {_fmt(row.get('strategy_sharpe'))} | {_fmt(row.get('spy_sharpe'))} | "
            f"{_fmt(row.get('strategy_apy'), percent=True)} | {_fmt(row.get('spy_apy'), percent=True)} | "
            f"{_fmt(row.get('buys'))} | {_fmt(row.get('sells'))} |"
        )
    return "\n".join(rows)


def _regime_table(regimes: list[dict[str, Any]]) -> str:
    if not regimes:
        return "_No regime metrics found._"
    rows = ["| Regime | Sharpe | SPY Sharpe | APY | SPY APY |", "|---|---:|---:|---:|---:|"]
    for row in regimes:
        rows.append(
            f"| {row['label']} | {_fmt(row.get('strategy_sharpe'))} | {_fmt(row.get('spy_sharpe'))} | "
            f"{_fmt(row.get('strategy_apy'), percent=True)} | {_fmt(row.get('spy_apy'), percent=True)} |"
        )
    return "\n".join(rows)


def _trade_table(counts: dict[str, Any]) -> str:
    if not counts:
        return "_No aggregate trade-count diagnostics found._"
    rows = ["| Group | Count |", "|---|---:|"]
    for group, values in counts.items():
        if not isinstance(values, dict):
            continue
        for key, count in sorted(values.items(), key=lambda item: str(item[0])):
            rows.append(f"| `{group}.{key}` | {_fmt(count)} |")
    return "\n".join(rows)


def _warning_block(warnings: list[str]) -> str:
    if not warnings:
        return ""
    return "\n".join(f"- {warning}" for warning in warnings)


def _summary_svg(run: LatestRun) -> str:
    metrics = run.metrics
    sharpe = _number(metrics.get("wf_3cut_sharpe_mean"), metrics.get("annual_net_sharpe"), metrics.get("sharpe"))
    spy_sharpe = _number(metrics.get("spy_sharpe_mean"))
    apy = _number(metrics.get("wf_3cut_apy_mean"), metrics.get("annual_net_apy"), metrics.get("apy"))
    spy_apy = _number(metrics.get("spy_apy_mean"))
    passed = metrics.get("passed")
    status = (
        "NO TRADES" if _is_no_trade_run(run)
        else "PASS" if passed is True
        else "FAIL" if passed is False
        else "LATEST"
    )
    return _svg_wrap(760, 260, "\n".join([
        '<rect x="0" y="0" width="760" height="260" fill="#f8fafc"/>',
        '<rect x="24" y="24" width="712" height="212" rx="8" fill="#ffffff" stroke="#d7dde8"/>',
        f'<text x="48" y="64" font-size="24" font-weight="700" fill="#132238">Latest Run {status}</text>',
        f'<text x="48" y="91" font-size="13" fill="#617085">{_esc(run.source.name)}</text>',
        _metric_card(48, 122, "Strategy Sharpe", _fmt(sharpe), "#2563eb"),
        _metric_card(224, 122, "SPY Sharpe", _fmt(spy_sharpe), "#64748b"),
        _metric_card(400, 122, "Strategy APY", _fmt(apy, percent=True), "#059669"),
        _metric_card(576, 122, "SPY APY", _fmt(spy_apy, percent=True), "#64748b"),
    ]))


def _metric_card(x: int, y: int, label: str, value: str, color: str) -> str:
    return "\n".join([
        f'<rect x="{x}" y="{y}" width="136" height="78" rx="8" fill="#f8fafc" stroke="#e2e8f0"/>',
        f'<text x="{x + 14}" y="{y + 26}" font-size="12" fill="#64748b">{_esc(label)}</text>',
        f'<text x="{x + 14}" y="{y + 58}" font-size="24" font-weight="700" fill="{color}">{_esc(value)}</text>',
    ])


def _grouped_bar_svg(rows: list[dict[str, Any]], title: str) -> str:
    if not rows:
        return _empty_svg(title)
    width, height = 860, 360
    plot_x, plot_y, plot_w, plot_h = 70, 64, 740, 220
    values = []
    for row in rows:
        for key in ("strategy_sharpe", "spy_sharpe"):
            if _is_number(row.get(key)):
                values.append(float(row[key]))
    lo = min([0.0, *values])
    hi = max([0.0, *values])
    if hi == lo:
        hi = lo + 1.0
    baseline = plot_y + plot_h - ((0.0 - lo) / (hi - lo)) * plot_h
    group_w = plot_w / max(len(rows), 1)
    parts = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text x="30" y="38" font-size="22" font-weight="700" fill="#132238">{_esc(title)}</text>',
        f'<line x1="{plot_x}" y1="{baseline:.1f}" x2="{plot_x + plot_w}" y2="{baseline:.1f}" stroke="#94a3b8"/>',
    ]
    for i, row in enumerate(rows):
        cx = plot_x + i * group_w + group_w / 2
        for offset, key, color in ((-13, "strategy_sharpe", "#2563eb"), (13, "spy_sharpe", "#94a3b8")):
            value = row.get(key)
            if not _is_number(value):
                continue
            y = plot_y + plot_h - ((float(value) - lo) / (hi - lo)) * plot_h
            bar_h = abs(baseline - y)
            parts.append(
                f'<rect x="{cx + offset - 10:.1f}" y="{min(y, baseline):.1f}" '
                f'width="20" height="{max(bar_h, 1):.1f}" rx="4" fill="{color}"/>'
            )
        parts.append(
            f'<text x="{cx:.1f}" y="{plot_y + plot_h + 34}" font-size="11" '
            f'text-anchor="middle" fill="#475569">{_esc(str(row.get("label", ""))[:18])}</text>'
        )
    parts.append('<circle cx="650" cy="34" r="5" fill="#2563eb"/><text x="662" y="39" font-size="12" fill="#475569">Strategy</text>')
    parts.append('<circle cx="730" cy="34" r="5" fill="#94a3b8"/><text x="742" y="39" font-size="12" fill="#475569">SPY</text>')
    return _svg_wrap(width, height, "\n".join(parts))


def _trade_svg(counts: dict[str, Any]) -> str:
    items: list[tuple[str, float]] = []
    for group, values in counts.items():
        if isinstance(values, dict):
            for key, value in values.items():
                if _is_number(value):
                    items.append((f"{group}.{key}", float(value)))
    if not items:
        return _empty_svg("Trade Diagnostics")
    items = sorted(items, key=lambda item: item[1], reverse=True)[:8]
    width, height = 860, 340
    max_v = max(v for _, v in items) or 1.0
    parts = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc"/>',
        '<text x="30" y="38" font-size="22" font-weight="700" fill="#132238">Trade Diagnostics</text>',
    ]
    for i, (label, value) in enumerate(items):
        y = 68 + i * 30
        w = 560 * (value / max_v)
        parts.append(f'<text x="30" y="{y + 16}" font-size="12" fill="#475569">{_esc(label[-56:])}</text>')
        parts.append(f'<rect x="260" y="{y}" width="{w:.1f}" height="18" rx="5" fill="#0f766e"/>')
        parts.append(f'<text x="{270 + w:.1f}" y="{y + 14}" font-size="12" fill="#0f172a">{_fmt(value)}</text>')
    return _svg_wrap(width, height, "\n".join(parts))


def _empty_svg(title: str) -> str:
    return _svg_wrap(860, 220, f"""
  <rect x="0" y="0" width="860" height="220" fill="#f8fafc"/>
  <text x="30" y="42" font-size="22" font-weight="700" fill="#132238">{_esc(title)}</text>
  <text x="30" y="94" font-size="15" fill="#64748b">No chartable metrics found in the latest run artifact.</text>
""")


def _svg_wrap(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        '<style>text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}</style>\n'
        f"{body}\n</svg>\n"
    )


def _cut_label(cut: dict[str, Any]) -> str:
    start = str(cut.get("start") or "")[:10]
    end = str(cut.get("end") or "")[:10]
    if start and end:
        return f"{start} to {end}"
    return str(cut.get("cut") or cut.get("label") or "cut")


def _number(*values: Any) -> float | None:
    for value in values:
        if _is_number(value):
            return float(value)
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value == value and value not in (float("inf"), float("-inf"))


def _fmt(value: Any, *, percent: bool = False) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return value.replace("|", "\\|")
    if not _is_number(value):
        return "n/a"
    v = float(value)
    if percent:
        return f"{v * 100:.2f}%"
    if abs(v) >= 100:
        return f"{v:,.0f}"
    return f"{v:.3f}"


def _is_percent_metric(key: str) -> bool:
    return key in {
        "apy",
        "annual_net_apy",
        "event_level_apy",
        "wf_3cut_apy_mean",
        "spy_apy_mean",
        "strategy_minus_spy_apy_mean",
    }


def _esc(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        pass
    try:
        return str(Path("..") / resolved.relative_to(Path.cwd().resolve().parent))
    except ValueError:
        return str(resolved)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", default=[], help="Search root for JSON metrics")
    parser.add_argument("--docs-dir", default="docs", help="Output docs directory")
    args = parser.parse_args(argv)
    roots = [Path(p) for p in args.root] if args.root else list(DEFAULT_SEARCH_ROOTS)
    out = generate_latest_run_docs(search_roots=roots, docs_dir=Path(args.docs_dir))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
