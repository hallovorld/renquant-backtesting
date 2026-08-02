"""Stage-1 lineage lane — #94 implementation slice 3 (dual-read stamps).

ONE entry point, ``attempt_lineage_stamp``, called by the gate runner after
``inspect_artifact_usage`` when the eval scope is ``walkforward_manifest``. It
ALWAYS returns a stamp block and NEVER raises: any internal failure yields
``{"lineage_lane": "unavailable", "reason": …}`` so the recipe path's behavior
and stamps are byte-unchanged no matter what happens in here (the merged #94
design's Stage-1 contract; admission is NOT touched anywhere in this module).

For the gbdt prod recipe the lineage is built IN MEMORY from the WF manifest's
own retrains: the 43 per-window artifacts already on disk self-carry the
admissibility fields, so no model-repo dependency exists on the live gate path.
The identity rule is the merged #94 one: ``lineage_root_sha = sha256(recipe_id
+ LF + LF-joined ordered per-window artifact shas + LF)`` with ``recipe_id`` =
the candidate recipe fingerprint the recipe path already validated — the same
identity the stamp's recipe block carries, so the two blocks are cross-checkable.

The caller grid (first-OOS-date per window) comes from the MANIFEST's own
cutoff ladder — window w's OOS starts at the first trading day after its
cutoff and the grid is derived by the (cut, next_cut] rule the corpus builders
use. The artifacts' own declared windows are never consulted (the
no-self-attestation rule).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from renquant_backtesting.wf_gate.lineage_admissibility import (
    check_window,
    lineage_root_sha,
)

#: Stage-1 minimum, mirrors lineage_admissibility.DEFAULT_MIN_ADMISSIBLE_WINDOWS.
MIN_WINDOWS = 8


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_gbdt_lineage_view(manifest_path: Path, strategy_dir: Path,
                            recipe_id: str) -> dict:
    """An in-memory lineage view over the WF manifest's retrains.

    Raises on structural problems — the caller (``attempt_lineage_stamp``)
    converts every raise into a stamped ``unavailable``.
    """
    man = json.loads(Path(manifest_path).read_text())
    retrains = man.get("retrains") or []
    if not retrains:
        raise ValueError("WF manifest has no retrains")
    # STRUCTURAL integrity (review round 1): the ladder must be unique and
    # chronologically ordered, and every artifact's SELF-CARRIED cutoff must
    # equal its manifest retrain cutoff — a reordered ladder or a wrong
    # artifact_uri must not acquire an admissible root.
    cutoffs = [str(r.get("cutoff_date", ""))[:10] for r in retrains]
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("WF manifest cutoff ladder has duplicates")
    if cutoffs != sorted(cutoffs):
        raise ValueError("WF manifest cutoff ladder is not chronologically ordered")
    folds = []
    shas = []
    for r in retrains:
        rel = r.get("artifact_uri")
        if not rel:
            raise ValueError(f"retrain {r.get('cutoff_date')} lacks artifact_uri")
        p = Path(strategy_dir) / str(rel)
        if not p.is_file():
            raise FileNotFoundError(f"window artifact missing: {p}")
        manifest_cut = str(r.get("cutoff_date", ""))[:10]
        art_cut = str(json.loads(p.read_text()).get("cutoff_date", ""))[:10]
        if art_cut != manifest_cut:
            raise ValueError(
                f"artifact cutoff {art_cut!r} != manifest retrain cutoff "
                f"{manifest_cut!r} at {rel} — wrong artifact behind this window")
        sha = _sha256_file(p)
        folds.append({"cutoff_date": manifest_cut,
                      "artifact_path": str(p), "artifact_sha256": sha})
        shas.append(sha)
    return {
        "recipe_id": recipe_id,
        "lineage_root_sha": lineage_root_sha(recipe_id, shas),
        "folds": folds,
    }


def _grid_from_cutoff_ladder(folds: list[dict]) -> dict[str, pd.Timestamp]:
    """First OOS date per window = the first business day AFTER its cutoff —
    the (cut, next_cut] rule's opening edge, derived from the MANIFEST ladder."""
    out: dict[str, pd.Timestamp] = {}
    for f in folds:
        cut = pd.Timestamp(f["cutoff_date"])
        out[f["cutoff_date"]] = cut + pd.offsets.BDay(1)
    return out


def attempt_lineage_stamp(*, artifact_usage: dict, strategy_dir: Path,
                          label_horizon_bdays: int,
                          recipe_id: str | None = None) -> dict:
    """The Stage-1 stamp block. NEVER raises; admission is never touched."""
    try:
        if (artifact_usage or {}).get("eval_scope") != "walkforward_manifest":
            return {"lineage_lane": "unavailable",
                    "reason": "eval scope is not walkforward_manifest"}
        manifest_raw = (artifact_usage or {}).get("manifest_path")
        if not manifest_raw:
            return {"lineage_lane": "unavailable",
                    "reason": "artifact_usage carries no manifest_path"}
        rid = recipe_id or str(
            (artifact_usage or {}).get("candidate_recipe_fingerprint") or "")
        if not rid:
            return {"lineage_lane": "unavailable",
                    "reason": "no recipe identity available for the lineage root"}
        view = build_gbdt_lineage_view(Path(manifest_raw), Path(strategy_dir), rid)
        grid = _grid_from_cutoff_ladder(view["folds"])
        windows = []
        n_adm = 0
        for f in view["folds"]:
            art = json.loads(Path(f["artifact_path"]).read_text())
            v = check_window(art, grid[f["cutoff_date"]], label_horizon_bdays,
                             artifact_sha=f["artifact_sha256"])
            windows.append(v.as_stamp())
            n_adm += int(v.admissible)
        verdict = "admissible" if n_adm >= MIN_WINDOWS else "refused"
        return {
            "lineage_lane": "stage1",
            "candidate_lineage_used": True,
            "candidate_artifact_used": False,   # documented meaning per #94:
            # this lane never direct-scores the served booster
            "lineage_root_sha": view["lineage_root_sha"],
            "recipe_id": rid,
            "n_windows": len(windows),
            "n_admissible": n_adm,
            "lineage_admissibility": verdict,
            "windows": windows,
        }
    except Exception as exc:  # noqa: BLE001 — the lane must never break the gate
        return {"lineage_lane": "unavailable", "reason": str(exc)[:300]}
