"""Frozen-input-bundle guard unit tests (model#79 round-3).

Tiny synthetic bundle + target fixture. Contract under test
(``wf_gate.input_bundle_guard.verify_input_bundle``):

* clean target -> empty mismatch list;
* missing listed file / mutated listed file -> one VOID line each;
* extra file inside a covered group -> VOID extra (bidirectional
  membership), while files outside every covered group are ignored;
* wrong frozen root digest -> single short-circuit VOID line;
* manifest rows naming META files (MANIFEST.sha256 / ROOT_DIGEST) are
  excluded from verification;
* covered groups are BUNDLE-DERIVED: manifest parent dirs truncated to
  the top 2 path levels; root-level entries contribute no group.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from renquant_backtesting.wf_gate.input_bundle_guard import (
    derive_covered_groups,
    main,
    verify_input_bundle,
)

TARGET_FILES = {
    "data/ohlcv/AAPL.csv": b"aapl-prices",
    "data/ohlcv/MSFT.csv": b"msft-prices",
    "models/model.bin": b"weights",
    "artifacts/wf/calibrators/cal.json": b"{}",
    "run_meta.txt": b"root-level listed file",
}


def _write_target(tmp_path: Path) -> Path:
    target = tmp_path / "target"
    for rel, blob in TARGET_FILES.items():
        p = target / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(blob)
    return target


def _build_bundle(tmp_path: Path, extra_manifest_lines: list[str] | None = None,
                  ) -> tuple[Path, str]:
    """Write MANIFEST.sha256 + ROOT_DIGEST; return (bundle_dir, root_digest)."""
    bundle = tmp_path / "bundle"
    bundle.mkdir(exist_ok=True)
    lines = [
        f"{hashlib.sha256(blob).hexdigest()}  {len(blob)}  {rel}"
        for rel, blob in sorted(TARGET_FILES.items())
    ]
    lines.extend(extra_manifest_lines or [])
    manifest = "\n".join(lines) + "\n"
    (bundle / "MANIFEST.sha256").write_text(manifest)
    root = hashlib.sha256(manifest.encode()).hexdigest()
    (bundle / "ROOT_DIGEST").write_text(root + "\n")
    return bundle, root


@pytest.fixture()
def bundle_and_target(tmp_path: Path) -> tuple[Path, Path, str]:
    target = _write_target(tmp_path)
    bundle, root = _build_bundle(tmp_path)
    return bundle, target, root


def test_clean_target_verifies_ok(bundle_and_target) -> None:
    bundle, target, root = bundle_and_target
    assert verify_input_bundle(bundle, target, root) == []


def test_missing_listed_file_is_void(bundle_and_target) -> None:
    bundle, target, root = bundle_and_target
    (target / "data/ohlcv/MSFT.csv").unlink()
    assert verify_input_bundle(bundle, target, root) == [
        "VOID missing: data/ohlcv/MSFT.csv",
    ]


def test_mutated_listed_file_is_void(bundle_and_target) -> None:
    bundle, target, root = bundle_and_target
    (target / "data/ohlcv/AAPL.csv").write_bytes(b"tampered")
    assert verify_input_bundle(bundle, target, root) == [
        "VOID digest mismatch: data/ohlcv/AAPL.csv",
    ]


def test_extra_file_in_covered_group_is_void(bundle_and_target) -> None:
    bundle, target, root = bundle_and_target
    (target / "data/ohlcv/TSLA.csv").write_bytes(b"not frozen")
    assert verify_input_bundle(bundle, target, root) == [
        "VOID extra file not in manifest: data/ohlcv/TSLA.csv",
    ]


def test_extra_file_outside_covered_groups_is_ignored(bundle_and_target) -> None:
    """Membership only sweeps covered groups.

    ``data/other/`` shares only ONE level with the covered ``data/ohlcv``
    group, and root-level entries (``run_meta.txt``) contribute no group,
    so neither stray is flagged.
    """
    bundle, target, root = bundle_and_target
    (target / "data" / "other").mkdir(parents=True)
    (target / "data/other/stray.csv").write_bytes(b"uncovered")
    (target / "stray_root.txt").write_bytes(b"uncovered")
    assert verify_input_bundle(bundle, target, root) == []


def test_extra_in_sibling_subdir_of_covered_group_is_void(
        bundle_and_target) -> None:
    """A group claims its WHOLE top-2 prefix, including sibling subdirs.

    ``artifacts/wf/calibrators/cal.json`` derives group ``artifacts/wf``,
    so an unlisted file under ``artifacts/wf/other/`` is an extra.
    """
    bundle, target, root = bundle_and_target
    (target / "artifacts" / "wf" / "other").mkdir(parents=True)
    (target / "artifacts/wf/other/x.bin").write_bytes(b"uncovered? no")
    assert verify_input_bundle(bundle, target, root) == [
        "VOID extra file not in manifest: artifacts/wf/other/x.bin",
    ]


def test_bad_root_digest_short_circuits(bundle_and_target) -> None:
    """Wrong frozen root => single VOID line even with other damage."""
    bundle, target, _root = bundle_and_target
    (target / "data/ohlcv/MSFT.csv").unlink()  # would also be VOID
    mismatches = verify_input_bundle(bundle, target, "0" * 64)
    assert len(mismatches) == 1
    assert mismatches[0].startswith("VOID root digest: ")
    assert mismatches[0].endswith(f"!= frozen {'0' * 64}")


def test_meta_file_rows_are_excluded(tmp_path: Path) -> None:
    """Manifest rows for MANIFEST.sha256 / ROOT_DIGEST are skipped.

    They are bundle meta files, absent from the target by design; the
    guard must not report them missing.
    """
    target = _write_target(tmp_path)
    fake = hashlib.sha256(b"meta").hexdigest()
    bundle, root = _build_bundle(tmp_path, extra_manifest_lines=[
        f"{fake}  4  MANIFEST.sha256",
        f"{fake}  4  ROOT_DIGEST",
    ])
    assert verify_input_bundle(bundle, target, root) == []


def test_missing_manifest_is_a_mismatch(tmp_path: Path) -> None:
    target = _write_target(tmp_path)
    empty_bundle = tmp_path / "empty_bundle"
    empty_bundle.mkdir()
    mismatches = verify_input_bundle(empty_bundle, target, "0" * 64)
    assert len(mismatches) == 1
    assert mismatches[0].startswith("VOID manifest missing: ")


def test_derive_covered_groups_rule() -> None:
    """Top-2-level truncation of manifest parent dirs; root entries none."""
    assert derive_covered_groups([
        "data/ohlcv/AAPL.csv",
        "models/model.bin",
        "artifacts/wf/calibrators/cal.json",
        "run_meta.txt",
    ]) == ["artifacts/wf", "data/ohlcv", "models"]


def test_cli_ok_exit_0(bundle_and_target, capsys) -> None:
    bundle, target, root = bundle_and_target
    rc = main([str(bundle), str(target), "--frozen-root", root])
    out = capsys.readouterr().out
    assert rc == 0
    assert f"VERIFY OK: {len(TARGET_FILES)} files verified" in out
    assert f"root={root}" in out


def test_cli_mismatch_exit_4_prints_all_void_lines(
        bundle_and_target, capsys) -> None:
    bundle, target, root = bundle_and_target
    (target / "data/ohlcv/AAPL.csv").write_bytes(b"tampered")
    (target / "data/ohlcv/MSFT.csv").unlink()
    rc = main([str(bundle), str(target), "--frozen-root", root])
    out = capsys.readouterr().out
    assert rc == 4
    assert "VOID digest mismatch: data/ohlcv/AAPL.csv" in out
    assert "VOID missing: data/ohlcv/MSFT.csv" in out
    assert "PREFLIGHT FAILED: 2 mismatch(es)" in out
