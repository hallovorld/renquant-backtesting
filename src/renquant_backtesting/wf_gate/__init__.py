"""WF gate + sim ledger owned by renquant-backtesting.

The three modules in this package were lifted byte-for-byte from the umbrella's
``scripts/`` directory on 2026-05-30 (see ``README.md`` for status). The umbrella
copies remain rollback targets while wrappers move to package entry points.

Modules
-------
``runner``     — ``scripts/run_wf_gate.py`` (2525 lines)
``sim_driver`` — ``scripts/run_sim_104.py`` (304 lines)
``sim_ledger`` — ``scripts/sim_trade_ledger.py`` (1205 lines)

``python -m renquant_backtesting.wf_gate`` is the multirepo wrapper target for
the WF gate when ``RENQUANT_REPO_ROOT`` points at the umbrella checkout. The
larger Task/Job/Pipeline refactor still proceeds incrementally.
"""
