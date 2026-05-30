"""WF gate + sim ledger — parallel copy in renquant-backtesting.

The three modules in this package were lifted byte-for-byte from the umbrella's
``scripts/`` directory on 2026-05-30 (see ``README.md`` for status). The umbrella
copies remain authoritative and live; this package is a staging area for the
Task/Job/Pipeline refactor (§1c) that will land in a follow-up session.

Modules
-------
``runner``     — ``scripts/run_wf_gate.py`` (2525 lines)
``sim_driver`` — ``scripts/run_sim_104.py`` (304 lines)
``sim_ledger`` — ``scripts/sim_trade_ledger.py`` (1205 lines)

Do **not** point production cron / preflight / orchestrator at this package
yet — the umbrella scripts are still the live path. Once the refactor lands
and an integration smoke proves byte-equivalence, callers can be flipped one
at a time.
"""
