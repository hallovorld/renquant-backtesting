"""LEAN assembly — format cached OHLCV into LEAN daily-equity artifacts.

Functional-lift (copy-not-move) of the umbrella's LEAN-export format core.
The pure format function ``build_daily_lines`` lives here; the umbrella-path-
coupled ``export_symbol`` orchestration (reads repo data dirs) stays in the
umbrella until it is rewired to base-data manifests.
"""
from __future__ import annotations

from .export import build_daily_lines

__all__ = ["build_daily_lines"]
