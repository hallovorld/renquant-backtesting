"""Repository-root resolution helpers for package CLIs.

Subrepo modules often operate on the umbrella RenQuant checkout's data,
strategy configs, logs, and artifacts. Resolve that root explicitly instead
of assuming it is relative to the installed package location.
"""
from __future__ import annotations

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

