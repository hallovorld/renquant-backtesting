"""LEAN daily-equity format core (pure; df -> LEAN CSV lines).

Lifted verbatim (behavior-identical) from the umbrella
`scripts/export_lean_data.py::build_daily_lines`. LEAN stores equity prices as
deci-cents (price * 10000, integer) with a `yyyymmdd 00:00` timestamp.
"""
from __future__ import annotations

import pandas as pd


def build_daily_lines(df: pd.DataFrame) -> list[str]:
    """Format an OHLCV frame into LEAN daily-equity CSV lines.

    LEAN equity convention: prices are integer deci-cents (price * 10000);
    timestamp is `yyyymmdd 00:00`; columns open,high,low,close,volume.
    """
    lines = []
    for timestamp, row in df.iterrows():
        date_text = pd.Timestamp(timestamp).strftime("%Y%m%d 00:00")
        open_price = int(round(float(row["open"]) * 10000))
        high_price = int(round(float(row["high"]) * 10000))
        low_price = int(round(float(row["low"]) * 10000))
        close_price = int(round(float(row["close"]) * 10000))
        volume = int(round(float(row["volume"])))
        lines.append(
            f"{date_text},{open_price},{high_price},{low_price},{close_price},{volume}"
        )
    return lines
