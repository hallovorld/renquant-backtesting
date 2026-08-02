"""Causal admissibility for candidate-LINEAGE gate scoring (design: PR #94).

Implements the fail-closed admissibility contract of
``doc/design/2026-08-01-wf-gate-candidate-scoring-lane.md`` as a pure module —
no runner surgery in this slice, no admission behavior anywhere:

* a LINEAGE is an ordered list of per-window snapshot artifacts, identity-bound
  by ``lineage_root_sha = sha256(recipe_id + LF + LF-joined ordered artifact
  shas + LF)``;
* a window is ADMISSIBLE for scoring its OOS range only when its snapshot's
  causal provenance verifies on the artifact's own self-carried fields:
  ``effective_train_cutoff + label_horizon (BDays) < first OOS score date``,
  with the realized embargo margin RECORDED;
* absence of any required field, a digest mismatch against the manifest, or a
  violated cutoff is a REFUSAL — a stamped outcome, never a silent skip;
* fewer admissible windows than ``min_admissible_windows`` refuses the whole
  lineage verdict.

Everything returned here is a plain dict ready to be embedded verbatim in a
gate stamp. The module never loads boosters and never scores — that is the
next slice; this one decides only WHO MAY BE SCORED.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

#: Fewer admissible windows than this refuses the lineage verdict outright.
#: Matches the gap-block minimum any downstream block statistic needs (>= 2
#: blocks) with headroom; a lineage that cannot clear it has no business
#: producing gate evidence. Overridable per call, never silently.
DEFAULT_MIN_ADMISSIBLE_WINDOWS = 8

#: The fields a per-window snapshot must SELF-CARRY (the #94 contract; the
#: measured gbdt and clf window artifacts both already carry all of them).
REQUIRED_SNAPSHOT_FIELDS = (
    "cutoff_date",
    "cutoff_embargo_days",
    "effective_train_cutoff_date",
)


def lineage_root_sha(recipe_id: str, ordered_artifact_shas: list[str]) -> str:
    """The #94 identity rule, exactly: sha256 over ``recipe_id + LF + joined + LF``."""
    payload = recipe_id + "\n" + "\n".join(ordered_artifact_shas) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class WindowVerdict:
    cutoff_date: str
    admissible: bool
    reason: str
    embargo_margin_bdays: int | None = None
    artifact_sha256: str | None = None
    detail: dict = field(default_factory=dict)

    def as_stamp(self) -> dict:
        out = {
            "cutoff_date": self.cutoff_date,
            "admissibility": "admissible" if self.admissible else "refused",
            "reason": self.reason,
            "embargo_margin_bdays": self.embargo_margin_bdays,
            "artifact_sha256": self.artifact_sha256,
        }
        out.update(self.detail)
        return out


def check_window(artifact: dict, first_oos_date: pd.Timestamp,
                 label_horizon_bdays: int,
                 artifact_sha: str | None = None) -> WindowVerdict:
    """One window's causal admissibility, from the snapshot's OWN fields.

    ``label_horizon_bdays`` comes from the GATE's label contract, deliberately
    not from the artifact — an artifact must not get to shrink its own horizon.
    """
    cutoff = str(artifact.get("cutoff_date", ""))[:10] or "?"
    missing = [f for f in REQUIRED_SNAPSHOT_FIELDS if not artifact.get(f)]
    if missing:
        return WindowVerdict(cutoff, False,
                             f"missing self-carried provenance fields: {missing}",
                             artifact_sha256=artifact_sha)
    etc = pd.Timestamp(str(artifact["effective_train_cutoff_date"]))
    first_oos = pd.Timestamp(first_oos_date)
    safe_after = etc + pd.offsets.BDay(int(label_horizon_bdays))
    if not safe_after < first_oos:
        return WindowVerdict(
            cutoff, False,
            f"causal violation: effective_train_cutoff {etc.date()} + "
            f"{label_horizon_bdays} BDays = {safe_after.date()} >= first OOS "
            f"date {first_oos.date()}",
            artifact_sha256=artifact_sha,
            detail={"effective_train_cutoff_date": str(etc.date()),
                    "first_oos_date": str(first_oos.date())})
    margin = len(pd.bdate_range(safe_after, first_oos)) - 1
    return WindowVerdict(
        cutoff, True, "causally admissible",
        embargo_margin_bdays=int(margin), artifact_sha256=artifact_sha,
        detail={"effective_train_cutoff_date": str(etc.date()),
                "first_oos_date": str(first_oos.date())})


def evaluate_lineage(manifest_path: Path, *, recipe_id_key: str,
                     label_horizon_bdays: int,
                     first_oos_dates: dict[str, "pd.Timestamp"],
                     min_admissible_windows: int = DEFAULT_MIN_ADMISSIBLE_WINDOWS,
                     ) -> dict:
    """Verify a lineage manifest end-to-end and grade every window.

    ``first_oos_dates`` maps ``cutoff_date`` (YYYY-MM-DD) to that window's first
    OOS score date — supplied by the CALLER's corpus/grid, deliberately not
    trusted from the artifacts (an artifact must not get to place its own OOS
    window). Returns a stamp-ready dict; never raises on bad data — bad data is
    a REFUSAL with a reason.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return {"lineage_verdict": "refused",
                "reason": f"lineage manifest missing: {manifest_path}"}
    man = json.loads(manifest_path.read_text())
    folds = man.get("folds") or []
    recipe_id = str(man.get(recipe_id_key, ""))
    if not recipe_id or not folds:
        return {"lineage_verdict": "refused",
                "reason": f"manifest lacks {recipe_id_key!r} or folds"}

    root_claimed = str(man.get("lineage_root_sha", ""))
    base = manifest_path.parent
    window_verdicts: list[WindowVerdict] = []
    ordered_shas: list[str] = []
    for f in folds:
        cutoff = str(f.get("cutoff_date", "?"))
        rel = f.get("artifact_path")
        p = base / str(rel) if rel else None
        if p is None or not p.is_file():
            window_verdicts.append(WindowVerdict(
                cutoff, False, f"snapshot file missing: {rel}"))
            ordered_shas.append(str(f.get("artifact_sha256", "")))
            continue
        actual = _sha256_file(p)
        ordered_shas.append(actual)
        if actual != f.get("artifact_sha256"):
            window_verdicts.append(WindowVerdict(
                cutoff, False,
                "snapshot digest mismatch vs manifest "
                f"({actual[:12]}… != {str(f.get('artifact_sha256'))[:12]}…)",
                artifact_sha256=actual))
            continue
        first_oos = first_oos_dates.get(cutoff)
        if first_oos is None:
            window_verdicts.append(WindowVerdict(
                cutoff, False, "caller supplied no first OOS date for this window",
                artifact_sha256=actual))
            continue
        art = json.loads(p.read_text())
        window_verdicts.append(
            check_window(art, first_oos, label_horizon_bdays, artifact_sha=actual))

    root_recomputed = lineage_root_sha(recipe_id, ordered_shas)
    root_ok = bool(root_claimed) and root_recomputed == root_claimed
    n_adm = sum(1 for w in window_verdicts if w.admissible)
    verdict = "admissible" if (root_ok and n_adm >= min_admissible_windows) else "refused"
    reason = ("ok" if verdict == "admissible" else
              ("lineage_root_sha mismatch (recomputed "
               f"{root_recomputed[:12]}… != claimed {root_claimed[:12]}…)"
               if not root_ok else
               f"only {n_adm} admissible windows < minimum {min_admissible_windows}"))
    return {
        "lineage_verdict": verdict,
        "reason": reason,
        "lineage_root_sha_claimed": root_claimed,
        "lineage_root_sha_recomputed": root_recomputed,
        "recipe_id": recipe_id,
        "n_windows": len(window_verdicts),
        "n_admissible": n_adm,
        "min_admissible_windows": min_admissible_windows,
        "windows": [w.as_stamp() for w in window_verdicts],
    }
