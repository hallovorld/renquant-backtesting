"""Frozen-input-bundle guard unit tests (model#79 round-3).

Tiny synthetic bundle + target fixture. Contract under test
(``wf_gate.input_bundle_guard.verify_input_bundle``):

* clean target -> empty mismatch list;
* missing listed file / mutated listed file -> one VOID line each;
* extra file inside an EXPLICIT covered root -> VOID extra
  (bidirectional membership), while files outside every covered root
  are ignored by the sweep;
* manifest entries OUTSIDE all covered roots (singletons) are still
  digest-verified individually;
* wrong frozen root digest -> single short-circuit VOID line;
* manifest rows naming META files (MANIFEST.sha256 / ROOT_DIGEST) are
  excluded from verification;
* CLI requires at least one --covered-root.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from renquant_backtesting.wf_gate.input_bundle_guard import (
    main,
    verify_input_bundle,
)

TARGET_FILES = {
    "data/ohlcv/AAPL.csv": b"aapl-prices",
    "data/ohlcv/MSFT.csv": b"msft-prices",
    "models/model.bin": b"weights",
    "artifacts/wf/calibrators/cal.json": b"{}",
    "run_meta.txt": b"root-level listed singleton",
}

#: Explicit covered roots for the fixture (caller-frozen, NOT derived).
#: ``run_meta.txt`` is deliberately outside all of them: a listed
#: singleton that must be digest-checked without sweeping the target root.
COVERED_ROOTS = ["data/ohlcv", "models", "artifacts/wf/calibrators"]


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
    assert verify_input_bundle(bundle, target, root, COVERED_ROOTS) == []


def test_missing_listed_file_is_void(bundle_and_target) -> None:
    bundle, target, root = bundle_and_target
    (target / "data/ohlcv/MSFT.csv").unlink()
    assert verify_input_bundle(bundle, target, root, COVERED_ROOTS) == [
        "VOID missing: data/ohlcv/MSFT.csv",
    ]


def test_mutated_listed_file_is_void(bundle_and_target) -> None:
    bundle, target, root = bundle_and_target
    (target / "data/ohlcv/AAPL.csv").write_bytes(b"tampered")
    assert verify_input_bundle(bundle, target, root, COVERED_ROOTS) == [
        "VOID digest mismatch: data/ohlcv/AAPL.csv",
    ]


def test_extra_file_in_covered_root_is_void(bundle_and_target) -> None:
    bundle, target, root = bundle_and_target
    (target / "data/ohlcv/TSLA.csv").write_bytes(b"not frozen")
    assert verify_input_bundle(bundle, target, root, COVERED_ROOTS) == [
        "VOID extra file not in manifest: data/ohlcv/TSLA.csv",
    ]


def test_extra_files_outside_covered_roots_are_ignored(
        bundle_and_target) -> None:
    """The sweep runs ONLY over the explicit covered roots.

    Strays in ``data/other/`` (sibling of a covered root), in
    ``artifacts/wf/other/`` (sibling subdir under a covered root's
    parent), and at the target root are all outside every covered root
    -> ignored. This is exactly the class of false VOID the derived
    top-2-level rule produced on the real G4 tree (sim outputs under
    ``data/``, code under ``backtesting/renquant_104/``).
    """
    bundle, target, root = bundle_and_target
    (target / "data" / "other").mkdir(parents=True)
    (target / "data/other/sim_runs_101.db").write_bytes(b"sim output")
    (target / "artifacts" / "wf" / "other").mkdir(parents=True)
    (target / "artifacts/wf/other/x.bin").write_bytes(b"uncovered")
    (target / "stray_root.txt").write_bytes(b"uncovered")
    assert verify_input_bundle(bundle, target, root, COVERED_ROOTS) == []


def test_singleton_entry_outside_roots_still_digest_verified(
        bundle_and_target) -> None:
    """Manifest entries outside all covered roots keep their digest check.

    ``run_meta.txt`` is not under any covered root; mutating it must
    still VOID (check 2 is unconditional), without any sweep of its
    parent directory.
    """
    bundle, target, root = bundle_and_target
    (target / "run_meta.txt").write_bytes(b"tampered singleton")
    assert verify_input_bundle(bundle, target, root, COVERED_ROOTS) == [
        "VOID digest mismatch: run_meta.txt",
    ]


def test_bad_root_digest_short_circuits(bundle_and_target) -> None:
    """Wrong frozen root => single VOID line even with other damage."""
    bundle, target, _root = bundle_and_target
    (target / "data/ohlcv/MSFT.csv").unlink()  # would also be VOID
    mismatches = verify_input_bundle(bundle, target, "0" * 64, COVERED_ROOTS)
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
    assert verify_input_bundle(bundle, target, root, COVERED_ROOTS) == []


def test_missing_manifest_is_a_mismatch(tmp_path: Path) -> None:
    target = _write_target(tmp_path)
    empty_bundle = tmp_path / "empty_bundle"
    empty_bundle.mkdir()
    mismatches = verify_input_bundle(
        empty_bundle, target, "0" * 64, COVERED_ROOTS)
    assert len(mismatches) == 1
    assert mismatches[0].startswith("VOID manifest missing: ")


def _cli_covered_root_args() -> list[str]:
    args: list[str] = []
    for r in COVERED_ROOTS:
        args += ["--covered-root", r]
    return args


def test_cli_ok_exit_0(bundle_and_target, capsys) -> None:
    bundle, target, root = bundle_and_target
    rc = main([str(bundle), str(target), "--frozen-root", root,
               *_cli_covered_root_args()])
    out = capsys.readouterr().out
    assert rc == 0
    assert f"VERIFY OK: {len(TARGET_FILES)} files verified" in out
    assert f"root={root}" in out


def test_cli_mismatch_exit_4_prints_all_void_lines(
        bundle_and_target, capsys) -> None:
    bundle, target, root = bundle_and_target
    (target / "data/ohlcv/AAPL.csv").write_bytes(b"tampered")
    (target / "data/ohlcv/MSFT.csv").unlink()
    rc = main([str(bundle), str(target), "--frozen-root", root,
               *_cli_covered_root_args()])
    out = capsys.readouterr().out
    assert rc == 4
    assert "VOID digest mismatch: data/ohlcv/AAPL.csv" in out
    assert "VOID missing: data/ohlcv/MSFT.csv" in out
    assert "PREFLIGHT FAILED: 2 mismatch(es)" in out


def test_cli_requires_at_least_one_covered_root(bundle_and_target) -> None:
    bundle, target, root = bundle_and_target
    with pytest.raises(SystemExit) as exc:
        main([str(bundle), str(target), "--frozen-root", root])
    assert exc.value.code == 2  # argparse error
