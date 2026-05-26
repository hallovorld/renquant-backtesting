from __future__ import annotations

import importlib
import sys


def test_backtesting_import_does_not_pull_live_brokers_or_training() -> None:
    importlib.import_module("renquant_backtesting")

    forbidden_prefixes = (
        "alpaca",
        "ib_insync",
        "renquant_execution",
        "renquant_model_gbdt",
        "renquant_model_patchtst",
    )
    offenders = sorted(
        name for name in sys.modules
        if name in forbidden_prefixes or name.startswith(forbidden_prefixes)
    )
    assert offenders == []
