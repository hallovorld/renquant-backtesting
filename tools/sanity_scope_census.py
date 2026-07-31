#!/usr/bin/env python3
"""Census: which sanity-eval scope did each stamped artifact actually run under?

GOAL-6. `sanity_eval_scope` is the branch marker inside `wf_gate_metadata`: the
candidate-scoring branch records `static_artifact`, the manifest branch records
`walkforward_manifest`. Which one *ran* is a fact about the artifacts, so this reads
them.

**Why this exists as a tool rather than a committed CSV.** Reviewed
`[codex on backtesting#89]`: *"it only parses the newly committed CSV; it neither loads
any listed artifact nor verifies a listed SHA-256, and the manifest does not identify an
immutable source snapshot."* Exactly right — a CSV that nobody can regenerate is a
transcription, and a digest column nobody verifies is decoration. So:

* ``--emit`` walks the collection root, reads each artifact, and writes the CSV plus a
  manifest naming the root, the inclusion query, the inclusion rule and the count;
* ``--verify`` re-reads every path in a committed CSV — resolved against ``--root``,
  which the manifest names, so the census is not pinned to one machine — recomputes its
  sha256, and
  reports any digest that no longer matches, any path that has vanished, and any
  artifact now present that the census does not list.

`--verify` is the one that carries the weight: it turns the committed CSV from a claim
into something falsifiable against the bytes on disk.

**Scope, stated.** These are files on one machine, not an immutable snapshot: an
artifact store is not content-addressed and a retrain can replace a file in place. The
digest is what makes that *visible* — a mismatch says "these bytes changed", which is
information. It does not make the store immutable, and this tool does not claim to.

Usage:
    python3 tools/sanity_scope_census.py --emit   --root <dir> --out <evidence-dir>
    python3 tools/sanity_scope_census.py --verify --root <dir> --out <evidence-dir>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import sys

SCHEMA = "sanity_scope_census.v1"
INCLUSION_QUERY = "panel-ltr.alpha158_fund*.json"
INCLUSION_RULE = "files whose JSON carries wf_gate_metadata"
CSV_NAME = "sanity_scope_census.csv"
MANIFEST_NAME = "census_manifest.json"
FIELDS = ["artifact", "artifact_path", "content_sha256", "sanity_eval_scope",
          "wf_eval_scope"]


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _scopes(payload: dict) -> tuple[str, str] | None:
    """(`sanity_eval_scope`, `wf_eval_scope`) if this artifact was stamped by the gate.

    An artifact with no `wf_gate_metadata` was never run through the gate, so it is not
    evidence about which branch executed and is EXCLUDED rather than counted as a
    missing value. That exclusion is the whole reason this is a tool: the hand-built
    census it replaces listed 29 artifacts and reported a scope for all of them, while
    only 14 carry the metadata a scope could have been read from. 15 rows asserted an
    observation nobody made.
    """
    md = payload.get("wf_gate_metadata")
    if not isinstance(md, dict) or not md:
        return None
    return (str(md.get("sanity_eval_scope", "")), str(md.get("wf_eval_scope", "")))


def collect(root: pathlib.Path, query: str = INCLUSION_QUERY) -> list[dict]:
    rows = []
    for path in sorted(root.glob(query)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        scopes = _scopes(payload)
        if scopes is None:
            continue
        rows.append({"artifact": path.name,
                     "artifact_path": str(path.relative_to(root)),
                     "content_sha256": _sha256(path),
                     "sanity_eval_scope": scopes[0], "wf_eval_scope": scopes[1]})
    return rows


def verify(root: pathlib.Path, csv_path: pathlib.Path,
           query: str = INCLUSION_QUERY) -> dict:
    """Re-read what the CSV names and report every way it disagrees with disk."""
    with csv_path.open() as fh:
        committed = list(csv.DictReader(fh))
    missing, changed, scope_drift = [], [], []
    for row in committed:
        # Paths are recorded RELATIVE to the collection root, which the manifest names.
        # An absolute path would pin the census to one machine's filesystem and make
        # --verify read as "everything is missing" anywhere else.
        p = root / row["artifact_path"]
        if not p.exists():
            missing.append(row["artifact"])
            continue
        if _sha256(p) != row["content_sha256"]:
            changed.append(row["artifact"])
            continue
        payload = json.loads(p.read_text(encoding="utf-8"))
        scopes = _scopes(payload) or ("", "")
        if scopes[0] != row["sanity_eval_scope"]:
            scope_drift.append(row["artifact"])
    on_disk = {r["artifact"] for r in collect(root, query)} if root.exists() else set()
    return {
        "n_committed": len(committed),
        "missing": missing,
        "digest_changed": changed,
        "scope_drift": scope_drift,
        "uncensused_on_disk": sorted(on_disk - {r["artifact"] for r in committed}),
        "ok": not (missing or changed or scope_drift),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true")
    mode.add_argument("--verify", action="store_true")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--query", default=INCLUSION_QUERY)
    args = ap.parse_args(argv)

    root, out = pathlib.Path(args.root), pathlib.Path(args.out)
    if args.emit:
        rows = collect(root, args.query)
        out.mkdir(parents=True, exist_ok=True)
        with (out / CSV_NAME).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        (out / MANIFEST_NAME).write_text(json.dumps({
            "schema": SCHEMA, "collection_root": str(root),
            "inclusion_query": args.query, "inclusion_rule": INCLUSION_RULE,
            "n_included": len(rows),
            "scope_note": ("an artifact store is not content-addressed; a digest makes a "
                           "change VISIBLE, it does not make the store immutable"),
        }, indent=2, sort_keys=True) + "\n")
        print(f"emitted {len(rows)} row(s) to {out / CSV_NAME}")
        return 0

    result = verify(root, out / CSV_NAME, args.query)
    print(f"verified {result['n_committed']} committed row(s) against {root}")
    for key in ("missing", "digest_changed", "scope_drift", "uncensused_on_disk"):
        if result[key]:
            print(f"  {key}: {len(result[key])} — {', '.join(result[key][:6])}")
    print("  OK" if result["ok"] else "  MISMATCH")
    return 0 if result["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
