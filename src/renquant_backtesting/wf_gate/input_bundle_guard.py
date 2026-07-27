"""Frozen-input-bundle guard for WF sim runs (model#79 round-3).

Generalized port of the model#79 amendment-1 checker
(``verify_g4_input_bundle`` v1). codex's round-3 ruling on
renquant-model#79 requires this run-integrity plumbing to live in the
repo that OWNS the sim driver and to be enforced THROUGH the launched
command (``wf_gate.sim_driver --input-bundle/--input-bundle-root``),
not via an external wrapper-script convention.

Bundle contract
---------------
A *bundle* is a directory containing ``MANIFEST.sha256`` (and,
conventionally, ``ROOT_DIGEST``). Manifest line format::

    <sha256>  <size>  <relpath>

``relpath`` is relative to the *target root* (the worktree the sim
reads). Lines naming the meta files ``MANIFEST.sha256`` / ``ROOT_DIGEST``
are ignored. Blank lines are ignored. A leading ``./`` on relpaths is
stripped.

Checks (ANY failure => nonzero mismatch list; one "VOID ..." line each):

1. ``sha256(<bundle>/MANIFEST.sha256) == frozen_root_digest``. On
   failure this SHORT-CIRCUITS: an untrusted manifest makes per-file
   results meaningless, so the root-digest line is the only one
   returned.
2. Every manifest-listed file exists under the target root with a
   matching sha256.
3. Bidirectional file-set membership inside the covered groups: any
   file present under a covered directory of the target but absent
   from the manifest is a mismatch.

Covered-groups derivation rule (BUNDLE-DERIVED, not hardcoded)
--------------------------------------------------------------
The covered groups are derived from the manifest itself: for every
manifest relpath, take its parent-directory path truncated to the top
2 path levels. Examples::

    data/ohlcv/AAPL.parquet                 -> data/ohlcv
    models/model.bin                        -> models
    backtesting/renquant_104/models/m.json  -> backtesting/renquant_104

Root-level manifest entries (no parent directory) contribute NO
covered group — they are digest-checked individually but do not sweep
the target root into the membership check. Consequence of the rule: a
manifest that lists ANY file under a two-level directory prefix claims
that ENTIRE prefix — every file found under it at verify time must be
listed, or it is reported as ``VOID extra file not in manifest``.
Bundles must therefore be built over complete covered directories.

The model#79 v1 checker's separate "derived config present" check is
subsumed by the general checks: the derived config is a listed file,
so a missing/mutated copy already fails check 2 and its group is
membership-covered by check 3.

Deterministic (sorted output, sorted walks), stdlib-only, read-only,
no network.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

#: Bundle meta files: never treated as target-content entries.
META_FILES = frozenset({"MANIFEST.sha256", "ROOT_DIGEST"})

MANIFEST_NAME = "MANIFEST.sha256"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(bundle_dir: str | os.PathLike[str]) -> dict[str, str]:
    """Parse ``<bundle_dir>/MANIFEST.sha256`` into ``{relpath: sha256}``.

    Meta-file entries (``MANIFEST.sha256`` / ``ROOT_DIGEST``) and blank
    lines are skipped; a leading ``./`` is stripped from relpaths.
    """
    manifest_path = os.path.join(os.fspath(bundle_dir), MANIFEST_NAME)
    listed: dict[str, str] = {}
    with open(manifest_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            digest, _size, rel = line.split(None, 2)
            rel = rel.strip()
            if rel.startswith("./"):
                rel = rel[2:]
            if rel in META_FILES or os.path.basename(rel) in META_FILES:
                continue
            listed[rel] = digest.lower()
    return listed


def derive_covered_groups(relpaths) -> list[str]:
    """Covered groups = manifest parent dirs truncated to top 2 levels.

    See the module docstring for the full rule. Returns a sorted list;
    nested duplicates are harmless (extras are deduplicated later).
    """
    groups: set[str] = set()
    for rel in relpaths:
        parts = rel.replace(os.sep, "/").split("/")
        dir_parts = parts[:-1]
        if not dir_parts:
            continue  # root-level entry: no covered group
        groups.add("/".join(dir_parts[:2]))
    return sorted(groups)


def verify_input_bundle(
    bundle_dir: str | os.PathLike[str],
    target_root: str | os.PathLike[str],
    frozen_root_digest: str,
) -> list[str]:
    """Verify ``target_root`` against the frozen bundle.

    Returns the (deterministically ordered) list of ``VOID ...``
    mismatch lines; an empty list means the target matches the bundle
    exactly. Read-only; never raises on content mismatches (only on
    e.g. an unreadable listed-and-present file, which IS an integrity
    failure worth a loud traceback).
    """
    bundle_dir = os.fspath(bundle_dir)
    target_root = os.fspath(target_root)

    manifest_path = os.path.join(bundle_dir, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        return [f"VOID manifest missing: {manifest_path}"]

    got_root = sha256_file(manifest_path)
    if got_root != frozen_root_digest.lower():
        # Untrusted manifest => short-circuit (see module docstring).
        return [f"VOID root digest: {got_root} != frozen {frozen_root_digest}"]

    listed = load_manifest(bundle_dir)

    mismatches: list[str] = []
    for rel, digest in sorted(listed.items()):
        p = os.path.join(target_root, rel)
        if not os.path.isfile(p):
            mismatches.append(f"VOID missing: {rel}")
            continue
        if sha256_file(p) != digest:
            mismatches.append(f"VOID digest mismatch: {rel}")

    extras: set[str] = set()
    for group in derive_covered_groups(listed):
        root = os.path.join(target_root, group)
        for dirpath, dirs, files in os.walk(root):
            dirs.sort()
            for f in sorted(files):
                rel = os.path.relpath(os.path.join(dirpath, f), target_root)
                rel = rel.replace(os.sep, "/")
                if rel not in listed:
                    extras.add(rel)
    mismatches.extend(f"VOID extra file not in manifest: {rel}"
                      for rel in sorted(extras))
    return mismatches


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m renquant_backtesting.wf_gate.input_bundle_guard",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("bundle_dir",
                    help="Bundle directory containing MANIFEST.sha256")
    ap.add_argument("target_root",
                    help="Target root (worktree) the manifest relpaths "
                         "resolve against")
    ap.add_argument("--frozen-root", required=True,
                    help="Frozen 64-hex sha256 of the bundle's "
                         "MANIFEST.sha256")
    args = ap.parse_args(argv)

    mismatches = verify_input_bundle(
        args.bundle_dir, args.target_root, args.frozen_root)
    for line in mismatches:
        print(line)
    if mismatches:
        print(f"PREFLIGHT FAILED: {len(mismatches)} mismatch(es)")
        return 4
    listed = load_manifest(args.bundle_dir)
    print(f"VERIFY OK: {len(listed)} files verified, membership clean, "
          f"root={args.frozen_root.lower()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
