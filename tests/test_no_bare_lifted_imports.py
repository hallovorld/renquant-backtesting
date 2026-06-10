from __future__ import annotations

import re
from pathlib import Path

# Umbrella-script module names whose lifted package equivalent has a
# DIFFERENT module name, so a bare import can never resolve inside this
# package — it only works when the umbrella scripts/ dir happens to be on
# sys.path. This killed weekly_wf_promote twice (2026-06-08 qp_contracts,
# 2026-06-09 sim_trade_ledger): the gate ran from .subrepo_runtime and all
# sim cuts died on ModuleNotFoundError.
BANNED_IMPORT_SOURCES = ("sim_trade_ledger", "qp_contracts")

SRC = Path(__file__).resolve().parents[1] / "src"


def test_no_bare_imports_of_renamed_lifted_modules() -> None:
    pattern = re.compile(
        r"^\s*(?:from\s+(%s)\s+import|import\s+(%s)\b)"
        % ("|".join(BANNED_IMPORT_SOURCES), "|".join(BANNED_IMPORT_SOURCES)),
        re.MULTILINE,
    )
    offenders: list[str] = []
    for py in SRC.rglob("*.py"):
        for m in pattern.finditer(py.read_text()):
            line_no = py.read_text()[: m.start()].count("\n") + 1
            offenders.append(f"{py.relative_to(SRC)}:{line_no}: {m.group(0).strip()}")
    assert not offenders, (
        "bare imports of umbrella-script modules that were renamed when "
        "lifted into the package (use renquant_backtesting.wf_gate.* "
        "instead):\n" + "\n".join(offenders)
    )
