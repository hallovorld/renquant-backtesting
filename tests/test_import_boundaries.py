"""Import-boundary tests for renquant-backtesting.

Per RFC §"Cross-Repo Contracts → Boundary test matrix" and §"Forbidden
dependencies", backtesting consumes scorers through ``renquant_common.
load_scorer`` only; it must NOT import any model-family or execution
package directly.

Both a runtime check (sys.modules after import) and an AST scan over the
source tree are enforced; the AST scan catches lazy/guarded imports
that the runtime check misses.
"""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

BACKTESTING_SRC = (
    Path(__file__).parent.parent / "src" / "renquant_backtesting"
)

FORBIDDEN_ROOT_IMPORTS = (
    "alpaca",
    "ib_insync",
    "renquant_execution",
    "renquant_model_gbdt",
    "renquant_model_patchtst",
    "renquant_model",  # post-P3 merged repo
    "torch",
    "transformers",
    "xgboost",
    "lightgbm",
    "catboost",
)


def test_backtesting_import_does_not_pull_live_brokers_or_training() -> None:
    """Runtime check — eagerly imported modules do not include forbidden roots."""
    before = set(sys.modules)
    importlib.import_module("renquant_backtesting")
    imported = set(sys.modules) - before
    offenders = sorted(
        name for name in imported
        if name in FORBIDDEN_ROOT_IMPORTS or name.startswith(FORBIDDEN_ROOT_IMPORTS)
    )
    assert offenders == [], (
        "renquant-backtesting must not import model-family or execution "
        "packages at runtime. Scorers consume via renquant_common.load_scorer; "
        "broker code lives in renquant-execution."
    )


def _root(module_name: str) -> str:
    return module_name.split(".", 1)[0]


def _collect_imports(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(_root(alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and node.level == 0:
                roots.add(_root(node.module))
    return roots


def test_backtesting_source_does_not_reference_forbidden_modules() -> None:
    """Static AST scan — no .py file references a forbidden import,
    even inside function bodies."""
    offenders: list[tuple[Path, str]] = []
    for py in BACKTESTING_SRC.rglob("*.py"):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        roots = _collect_imports(tree)
        bad = roots & set(FORBIDDEN_ROOT_IMPORTS)
        for root in sorted(bad):
            offenders.append((py.relative_to(BACKTESTING_SRC), root))
    assert offenders == [], (
        f"renquant-backtesting source references forbidden imports: "
        f"{offenders}. Backend-specific code belongs in the corresponding "
        f"renquant-model subdir; broker code in renquant-execution."
    )
