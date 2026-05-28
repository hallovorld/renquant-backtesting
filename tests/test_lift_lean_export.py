"""Parity test for the LEAN format-core lift (export_lean_data.build_daily_lines)."""
from __future__ import annotations

import importlib

import pandas as pd

lean = importlib.import_module("renquant_backtesting.lean")


def test_build_daily_lines_lean_decicent_format() -> None:
    df = pd.DataFrame(
        {"open": [100.0], "high": [101.5], "low": [99.25], "close": [100.75],
         "volume": [12345.0]},
        index=[pd.Timestamp("2024-03-18")],
    )
    lines = lean.build_daily_lines(df)
    assert lines == ["20240318 00:00,1000000,1015000,992500,1007500,12345"]


def test_build_daily_lines_rounds_and_orders_by_index() -> None:
    df = pd.DataFrame(
        {"open": [10.12345, 10.0], "high": [10.2, 10.1], "low": [10.0, 9.9],
         "close": [10.15, 10.05], "volume": [100.4, 200.6]},
        index=[pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")],
    )
    lines = lean.build_daily_lines(df)
    assert len(lines) == 2
    # deci-cent rounding: 10.12345*10000 = 101234.5 -> 101234 (banker's/round)
    assert lines[0].startswith("20240103 00:00,")
    assert lines[1].startswith("20240104 00:00,")
    # volume rounds to int
    assert lines[0].endswith(",100") and lines[1].endswith(",201")
