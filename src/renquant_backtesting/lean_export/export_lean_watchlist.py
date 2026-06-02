#!/usr/bin/env python
"""Export LEAN-compatible daily data for ALL symbols in a strategy watchlist.

Reads cached parquet from data/ohlcv/{SYMBOL}/1d.parquet and writes:
  - backtesting/data/equity/usa/daily/{symbol}.zip   (price CSV)
  - backtesting/data/equity/usa/map_files/{symbol}.csv
  - backtesting/data/equity/usa/factor_files/{symbol}.csv

The script compares the zip's modification time against the parquet source.
If the parquet is newer the zip is considered stale and is re-exported
automatically — no need to pass --force for routine daily updates.

Usage::

    python scripts/export_lean_watchlist.py --strategy renquant_102
    python scripts/export_lean_watchlist.py --strategy renquant_102 --symbols CRM UNH SHOP
    python scripts/export_lean_watchlist.py --strategy renquant_102 --force   # re-export all

Requires: pandas, pyarrow (available in the renquant conda environment).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from renquant_backtesting.repo_root import resolve_repo_root

try:
    from .export_lean_data import export_symbol
except ImportError:  # pragma: no cover - legacy direct-script execution
    from export_lean_data import export_symbol  # type: ignore[no-redef]


def _lean_daily(repo_root: Path) -> Path:
    return repo_root / "backtesting" / "data" / "equity" / "usa" / "daily"


def _ohlcv_roots(repo_root: Path) -> list[Path]:
    return [
        repo_root / "data" / "ohlcv",
        repo_root / "Notebooks" / "data" / "ohlcv",
    ]


def get_watchlist(strategy_name: str, repo_root: Path) -> list[str]:
    """Read watchlist + benchmark from a strategy config."""
    config_path = repo_root / "backtesting" / strategy_name / "strategy_config.json"
    if not config_path.exists():
        print(f"ERROR: strategy config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    config = json.loads(config_path.read_text())
    symbols = list(config.get("watchlist", []))
    benchmark = config.get("benchmark", "SPY")
    if benchmark and benchmark not in symbols:
        symbols.append(benchmark)
    stock_symbol = config.get("stock_symbol")
    if stock_symbol and stock_symbol not in symbols:
        symbols.append(stock_symbol)
    return symbols


def _parquet_path(symbol: str, repo_root: Path) -> Path | None:
    """Return the parquet path for *symbol*, checking both cache locations."""
    for root in _ohlcv_roots(repo_root):
        p = root / symbol.upper() / "1d.parquet"
        if p.exists():
            return p
    return None


def _zip_path(symbol: str, repo_root: Path) -> Path:
    return _lean_daily(repo_root) / f"{symbol.lower()}.zip"


def _export_status(symbol: str, repo_root: Path) -> tuple[str, str | None]:
    """Return (status, reason) for the symbol.

    status is one of: 'missing', 'stale', 'ok'
    reason is a human-readable explanation shown in the log.
    """
    import datetime

    zip_p = _zip_path(symbol, repo_root)
    pq_p  = _parquet_path(symbol, repo_root)

    if not zip_p.exists():
        return "missing", "no LEAN zip yet"

    if pq_p is None:
        return "ok", "no parquet to compare (zip kept)"

    zip_mt = os.path.getmtime(zip_p)
    pq_mt  = os.path.getmtime(pq_p)

    if pq_mt > zip_mt:
        delta_min = (pq_mt - zip_mt) / 60
        zip_date  = datetime.datetime.fromtimestamp(zip_mt).strftime("%Y-%m-%d %H:%M")
        pq_date   = datetime.datetime.fromtimestamp(pq_mt).strftime("%Y-%m-%d %H:%M")
        reason = f"parquet updated {pq_date}, zip is from {zip_date} (+{delta_min:.0f}m behind)"
        return "stale", reason

    zip_date = datetime.datetime.fromtimestamp(zip_mt).strftime("%Y-%m-%d %H:%M")
    return "ok", f"zip up-to-date ({zip_date})"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export LEAN data for all symbols in a strategy watchlist"
    )
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--force", action="store_true", help="Re-export all, ignoring freshness check")
    parser.add_argument("--repo-root", default=None,
                        help="Umbrella RenQuant repo root. Defaults to RENQUANT_REPO_ROOT or cwd.")
    args = parser.parse_args()

    repo_root = resolve_repo_root(args.repo_root)
    all_symbols = get_watchlist(args.strategy, repo_root)
    symbols = args.symbols if args.symbols else all_symbols
    unknown = set(symbols) - set(all_symbols)
    if unknown:
        print(f"WARNING: {unknown} not in strategy watchlist, exporting anyway")

    exported = 0
    skipped  = 0
    failed   = 0

    for symbol in sorted(symbols):
        status, reason = _export_status(symbol, repo_root)

        if not args.force and status == "ok":
            print(f"  {symbol:6s} — up-to-date, skipping  [{reason}]")
            skipped += 1
            continue

        pq_p = _parquet_path(symbol, repo_root)
        if pq_p is None:
            print(f"  {symbol:6s} — NO parquet cache found")
            print(f"           Run: python -c \"import common; common.fetch_ohlcv('{symbol}')\"")
            failed += 1
            continue

        action = "force-export" if args.force else status  # 'missing' or 'stale'
        try:
            daily_zip, map_file, factor_file = export_symbol(symbol, repo_root=repo_root)
            print(f"  {symbol:6s} — {action} → {daily_zip.name}  [{reason}]")
            exported += 1
        except Exception as e:
            print(f"  {symbol:6s} — FAILED ({action}): {e}")
            failed += 1

    print(f"\nDone: {exported} exported, {skipped} up-to-date, {failed} failed")
    total_zips = len(list(_lean_daily(repo_root).glob("*.zip")))
    print(f"Total LEAN data files: {total_zips}")

    if failed > 0:
        print("\nTo fetch missing parquet caches:")
        print("  conda activate renquant")
        print("  python -c \"import common; common.fetch_ohlcv('SYMBOL')\"")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
