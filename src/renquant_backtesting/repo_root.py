"""Repository-root resolution helpers for package CLIs.

Subrepo modules often operate on the umbrella RenQuant checkout's data,
strategy configs, logs, and artifacts. Resolve that root explicitly instead
of assuming it is relative to the installed package location.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def resolve_repo_root(value: str | Path | None = None) -> Path:
    """Return the umbrella RenQuant root for a package CLI.

    Precedence:
      1. explicit CLI value
      2. ``RENQUANT_REPO_ROOT``
      3. current working directory
    """
    candidate = value or os.environ.get("RENQUANT_REPO_ROOT") or Path.cwd()
    return Path(candidate).expanduser().resolve()


def strategy_dir(repo_root: Path, strategy: str) -> Path:
    """Return the umbrella strategy directory for artifacts and state."""
    return repo_root / "backtesting" / strategy


def resolve_strategy_config_path(
    repo_root: Path,
    strategy: str,
    value: str | Path | None = None,
) -> Path:
    """Return the strategy config path for package CLIs.

    Precedence:
      1. explicit CLI value
      2. ``RENQUANT_STRATEGY_CONFIG`` from the umbrella delegate
      3. umbrella ``backtesting/<strategy>/strategy_config.json``
    """
    candidate = value or os.environ.get("RENQUANT_STRATEGY_CONFIG")
    if candidate:
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        return path.resolve()
    return (strategy_dir(repo_root, strategy) / "strategy_config.json").resolve()


def load_strategy_config(
    repo_root: Path,
    strategy: str,
    value: str | Path | None = None,
) -> tuple[dict, Path]:
    """Load a strategy config and return ``(config, path)``."""
    path = resolve_strategy_config_path(repo_root, strategy, value)
    return json.loads(path.read_text()), path


def resolve_strategy_artifact_path(
    repo_root: Path,
    strategy: str,
    raw_path: str | Path,
) -> Path:
    """Resolve a config artifact path against the umbrella strategy dir."""
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return strategy_dir(repo_root, strategy) / path
