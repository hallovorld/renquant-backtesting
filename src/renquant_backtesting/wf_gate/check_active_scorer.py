"""Active-config scorer gate check (RFC #259 P0b) — canonical, multi-repo home.

Asserts the scorer artifact referenced by each ACTIVE strategy config's
``ranking.panel_scoring.artifact_path`` carries a passing ``wf_gate_metadata``,
delegating to :func:`assert_artifact_gated`. Catches the 2026-06-05-style
config-edit promotion bypass at CI / preflight instead of silently at runtime
(where live ``P-WF-GATE`` then blocks all buys → the ~2-week no-buy).

renquant-backtesting owns the gate, so the logic lives here; the umbrella only
delegates via ``python -m renquant_backtesting.wf_gate.check_active_scorer``
(RenQuant CLAUDE.md §3.5 multi-repo placement). Run with ``RENQUANT_REPO_ROOT``
pointing at the umbrella checkout (the daily/weekly wrappers already export it).

Exit codes:
  0 — every checked scorer is gated.
  1 — at least one ACTIVE production scorer is NOT gated (governance violation).
  2 — config / artifact-resolution error (cannot determine state — fail-closed).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from renquant_backtesting.forensics.model_acceptance import assert_artifact_gated
from renquant_backtesting.repo_root import (
    resolve_repo_root,
    resolve_strategy_artifact_path,
    strategy_dir,
)

# The configs the daily-full path actually loads (RenQuant CLAUDE.md §4.2): the
# primary live config (real paper orders) and the shadow leg (readonly). Both
# feed the daily signal, so both must carry a gated scorer.
DEFAULT_CONFIGS = ("strategy_config.json", "strategy_config.shadow.json")


def check_config(repo_root: Path, strategy: str, config_name: str) -> tuple[str, str]:
    """Check one strategy config's scorer artifact.

    Returns ``(status, detail)`` where status is ``"ok"`` / ``"violation"`` /
    ``"error"``. Pure (no printing / no exit) so it is unit-testable.
    """
    sdir = strategy_dir(repo_root, strategy)
    cand = Path(config_name)
    config_path = cand if cand.is_absolute() else sdir / config_name
    if not config_path.exists():
        return "error", f"config not found at {config_path}"
    try:
        cfg = json.loads(config_path.read_text())
        ps = cfg.get("ranking", {}).get("panel_scoring", {})
        if not isinstance(ps, dict) or not ps.get("artifact_path"):
            return "error", "no ranking.panel_scoring.artifact_path"
        artifact = resolve_strategy_artifact_path(
            repo_root, strategy, ps["artifact_path"],
        ).resolve()
        kind = ps.get("kind", "?")
    except (json.JSONDecodeError, OSError) as exc:
        return "error", f"cannot read/resolve config: {exc}"
    try:
        wf = assert_artifact_gated(artifact)
    except ValueError as exc:
        return "violation", f"(kind={kind}) {exc}"
    run_at = (wf or {}).get("run_at", "?")
    return "ok", f"(kind={kind} artifact={artifact.name} passed={wf.get('passed')} run_at={run_at})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RFC #259 P0b active-scorer gate check")
    ap.add_argument("--repo-root", default=None,
                    help="umbrella RenQuant root (default: $RENQUANT_REPO_ROOT or cwd)")
    ap.add_argument("--strategy", default="renquant_104")
    ap.add_argument("--config", action="append", dest="configs",
                    help="strategy config filename (under the strategy dir) or path; "
                         "repeatable. Default: the active primary + shadow configs.")
    args = ap.parse_args(argv)

    repo_root = resolve_repo_root(args.repo_root)
    configs = args.configs or list(DEFAULT_CONFIGS)

    violations = 0
    errors = 0
    for name in configs:
        status, detail = check_config(repo_root, args.strategy, name)
        label = Path(name).name
        if status == "ok":
            print(f"OK  {label}: scorer gated {detail}")
        elif status == "violation":
            print(f"VIOLATION  {label}: production scorer NOT gated {detail}", file=sys.stderr)
            violations += 1
        else:
            print(f"ERROR  {label}: {detail}", file=sys.stderr)
            errors += 1

    if errors:
        return 2
    if violations:
        print(
            f"\n{violations} active production config(s) carry an UNGATED scorer "
            f"— RFC #259 P0 promotion-bypass governance violation.",
            file=sys.stderr,
        )
        return 1
    print(f"\nAll {len(configs)} active production scorer(s) gated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
