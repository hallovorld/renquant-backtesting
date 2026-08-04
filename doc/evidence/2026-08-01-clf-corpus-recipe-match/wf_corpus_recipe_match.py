#!/usr/bin/env python3
"""Does this artifact's recipe have ANY walk-forward corpus? — for clf, zero of 85.

The GOAL-6 anchor says the certified clf recipe has no out-of-sample corpus. On 2026-07-31
I "corrected" that anchor using the GBDT corpus's 43 folds and had to withdraw it: I had
measured one lane's corpus and attached it to another, by DIRECTORY NAME. This settles it
by the gate's own criterion instead.

MEASURED 2026-08-01. Recipe match is `run_wf_gate._recipe_projection` — kind, feature_cols,
feature_norm_kind, label_col, lookahead_days, params:

    clf   recipe fingerprint  a4141c076b6b9591
    prod  recipe fingerprint  7d6845222c15626d
    corpus folds total                     85
    folds matching clf's recipe           **0**

All 83 usable folds carry `objective: rank:pairwise`; clf carries `binary:logistic`. **The
anchor is correct**, and it is now established mechanically rather than by folder name.

THE NEAR-MISS THAT MAKES THIS WORTH A TOOL. My first pass omitted `params` from the
projection — it is the one field of the six that is a nested dict — and got **82 of 85
matching**, i.e. the opposite conclusion. clf and prod agree on `kind`, on all 172
`feature_cols`, on `feature_norm_kind`, on `label_col` (`fwd_60d_excess`) and on
`lookahead_days` (60). The ENTIRE discriminating power here sits in one key: `objective`.

That is the complement of orch#713, and both are true: the projection is blind on the axes
separating 12 same-recipe boosters, and it is the only thing separating clf from prod. A
tool that reports WHICH field broke the match is the difference between those two readings.

Read-only. Opens artifacts, writes nothing.

Exit codes: ``0`` at least one fold matches, ``1`` none does, ``2`` usage/IO error,
``3`` SKIPPED — no corpus fold readable, so nothing was established.
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import sys
from pathlib import Path

#: `run_wf_gate._recipe_projection`, read from that function on 2026-08-01 — cited, not
#: asserted. `feature_source_contract_keys` is derived inside the gate rather than being an
#: artifact field, so it is declared unchecked rather than silently dropped.
PROJECTION_FIELDS = ("kind", "feature_cols", "feature_norm_kind", "label_col",
                     "lookahead_days", "params")
NOT_CHECKED = ("feature_source_contract_keys",)


def projection(a: dict) -> dict:
    return {
        "kind": a.get("kind"),
        "feature_cols": list(a.get("feature_cols") or []),
        "feature_norm_kind": list(a.get("feature_norm_kind") or []),
        "label_col": a.get("label_col"),
        "lookahead_days": int(a.get("lookahead_days") or 0),
        "params": a.get("params") or {},
    }


def fingerprint(a: dict) -> str:
    return hashlib.sha256(
        json.dumps(projection(a), sort_keys=True,
                   separators=(",", ":")).encode()).hexdigest()[:16]


def first_difference(a: dict, b: dict) -> list[str]:
    """WHICH projection fields differ. Reported because 'no match' without a reason is
    how a one-field difference gets mistaken for a whole-recipe difference — or missed
    entirely, which is what happened when `params` was dropped."""
    pa, pb = projection(a), projection(b)
    return [f for f in PROJECTION_FIELDS if pa[f] != pb[f]]


def survey(candidate: Path, corpus_glob: str) -> dict:
    try:
        cand = json.loads(candidate.read_text())
    except (OSError, ValueError) as exc:
        return {"status": "candidate_unreadable", "why": f"{type(exc).__name__}: {exc}"}
    cfp = fingerprint(cand)
    folds, matches = 0, 0
    diff_counts: collections.Counter = collections.Counter()
    unreadable = 0
    for p in sorted(glob.glob(corpus_glob, recursive=True)):
        try:
            d = json.loads(Path(p).read_text())
        except (OSError, ValueError):
            unreadable += 1
            continue
        # A fold with no feature_cols is not a scorer artifact (calibration companions
        # live in these directories too) and is not counted as a non-match.
        if not isinstance(d, dict) or not d.get("feature_cols"):
            continue
        folds += 1
        if fingerprint(d) == cfp:
            matches += 1
        else:
            diff_counts[",".join(first_difference(cand, d)) or "<identical projection>"] += 1
    return {
        "status": "checked" if folds else "no_folds",
        "candidate": str(candidate),
        "candidate_recipe_fingerprint": cfp,
        "n_folds": folds,
        "n_unreadable": unreadable,
        "n_matching": matches,
        "differing_field_sets": dict(diff_counts.most_common()),
        "projection_fields": list(PROJECTION_FIELDS),
        "not_checked": list(NOT_CHECKED),
        "note": ("Match is the gate's own recipe projection. Directory name is NOT used: "
                 "attaching one lane's corpus to another by folder name is an error this "
                 "programme has already made and retracted."),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--corpus-glob", required=True)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    rep = survey(a.candidate, a.corpus_glob)
    if rep["status"] == "candidate_unreadable":
        print(f"candidate unreadable: {rep['why']}", file=sys.stderr)
        return 2
    if rep["status"] == "no_folds":
        print(f"SKIPPED: no readable corpus fold matched {a.corpus_glob!r} — nothing was "
              f"established.", file=sys.stderr)
        return 3

    if a.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        print(f"  candidate {Path(rep['candidate']).name}")
        print(f"    recipe fingerprint {rep['candidate_recipe_fingerprint']}")
        print(f"    corpus folds {rep['n_folds']}   MATCHING {rep['n_matching']}")
        if rep["differing_field_sets"]:
            print("\n  fields that broke the match:")
            for fields, c in rep["differing_field_sets"].items():
                print(f"    {c:>4} fold(s)   {fields}")
        print(f"\n  projection fields: {', '.join(rep['projection_fields'])}")
        print(f"  NOT checked here : {', '.join(rep['not_checked'])}")
        print(f"\n  {rep['note']}")
    return 0 if rep["n_matching"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
