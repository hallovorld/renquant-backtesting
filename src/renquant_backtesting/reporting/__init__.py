"""Reporting helpers for backtest and simulation artifacts."""

from typing import Any


def generate_latest_run_docs(*args: Any, **kwargs: Any):
    from .latest_run_docs import generate_latest_run_docs as _impl

    return _impl(*args, **kwargs)

__all__ = ["generate_latest_run_docs"]
