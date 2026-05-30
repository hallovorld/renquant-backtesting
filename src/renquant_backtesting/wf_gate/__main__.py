"""B13 Phase 5 entry point — ``python -m renquant_backtesting.wf_gate``.

Delegates to ``runner.main()`` which is the byte-equivalent copy of umbrella
``scripts/run_wf_gate.py``. This module exists so production wrappers (the
weekly_wf_promote.sh chain, daily_104.sh, orchestrator) can flip from
``python scripts/run_wf_gate.py`` to ``python -m renquant_backtesting.wf_gate``
without changing any decision semantics — the lifted runner IS the same code.

Rollback: any wrapper can revert to ``scripts/run_wf_gate.py`` directly; the
umbrella baseline is preserved verbatim (copy-not-move).
"""
from __future__ import annotations

import sys


def main() -> int:
    # The runner module IS the production gate logic; it parses its own argv
    # and writes wf_gate_metadata into the artifact's sidecar/payload.
    from . import runner  # noqa: PLC0415
    rc = runner.main()
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
