"""Durable OOS pick-table export contract — the owning implementation.

RenQuant PR #430 (``scripts/regen_oos_pick_table.py``) established the durable
pick-table evidence contract for Track A: a fixed ``(date, name)`` schema, a
deterministic per-date decile algorithm, and a sidecar reproducibility manifest
anchored by a canonical, order-independent CONTENT hash of the exported table
(plus a secondary parquet transport hash). Per the #59 review, that contract
must not fork: the per-(date, name) scoring lives in THIS repo, so this module
is the contract's owning implementation, and the umbrella generator is meant to
become a thin wrapper over it.

Contract invariants (all hash-compatible with #430 by construction — see the
golden-hash test in ``tests/analysis/test_build_pick_table.py``):

* Schema: one row per ``(date, name)`` with columns
  ``[date, name, score, decile_rank, <label>, regime]`` (#430 uses
  ``fwd_60d_excess`` as the label column; the label is parameterized here and
  stamped in the sidecar).
* ``decile_rank``: cross-sectional decile of ``score`` within the date,
  0 (worst) .. 9 (best), via ``rank(method="first")`` + ``qcut`` — the same
  deterministic tie-break/fallback algorithm as #430.
* Canonical content hash: order-independent (canonical ``(date, name)``
  re-sort) with platform-stable fixed 10-decimal float formatting — identical
  bytes, and therefore identical digests, to #430's
  ``canonical_table_content_hash`` for ``label="fwd_60d_excess"``.
* Temporal semantics: the ``<label>`` column is a REALIZED forward return only
  observable after the pick date + label horizon. The export is a
  research-only artifact, never a live inference input, and canonical
  production output paths are refused (:func:`refuse_production_output_path`).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

N_DECILES = 10

#: The only ``data/`` subtree the research-artifact contract permits writing
#: into (same contract RenQuant#430 established: ``data/exp/`` is the explicit
#: experiment area; every other ``data/`` path is canonical production data).
RESEARCH_DATA_SEGMENT = "exp"


def pick_table_columns(label: str) -> list[str]:
    """The contract column order — matches RenQuant#430's schema exactly
    (``fwd_60d_excess`` generalized to ``<label>``)."""
    return ["date", "name", "score", "decile_rank", str(label), "regime"]


def decile_rank(scores: pd.Series) -> pd.Series:
    """Cross-sectional decile of ``scores`` within one date, 0 (worst) .. 9
    (best/top) — decile 9 is the model's top-decile long-side candidates.

    Same deterministic algorithm as RenQuant#430's ``_decile_rank``: ties are
    broken by ``rank(method="first")`` before ``qcut`` so ``qcut`` bins
    strictly-ordered integer ranks (never raw floats), which makes
    ``duplicates="drop"`` a pure safety net rather than something that silently
    reshuffles bin membership. Falls back to fewer than :data:`N_DECILES`
    buckets on a date with too few distinct names for 10 clean bins (documented,
    not a crash); fewer than 2 distinct values collapses to a single bucket 0.

    This intentionally REPLACES the ``rank(pct=True) * 10`` construction the
    first revision of backtesting#59 used, which mis-bucketed common small
    cross-sections (with exactly 10 distinct scores it produced deciles 1..9
    with a doubled 9 and no 0).
    """
    n_unique = int(scores.nunique())
    if n_unique < 2:
        return pd.Series(0, index=scores.index, dtype=int)
    n_bins = min(N_DECILES, n_unique)
    ranks = pd.qcut(
        scores.rank(method="first"), n_bins, labels=False, duplicates="drop"
    )
    return ranks.astype(int)


def build_pick_table(
    val: pd.DataFrame,
    mu: pd.Series,
    label: str,
    regimes: pd.DataFrame | None,
) -> pd.DataFrame:
    """Per-(date, name) OOS pick table (#430 schema) from scored validation rows.

    ``val`` must carry ``date``/``ticker``/``label`` columns; ``mu`` is the
    manifest-scored series. Alignment is strictly BY INDEX LABEL — never by
    positional ``to_numpy()`` — and the function fails closed instead of
    silently mis-attaching evidence:

    * duplicate index labels in ``val`` or ``mu``  -> ``ValueError``
    * ``mu`` missing any ``val`` index label       -> ``ValueError``
    * non-finite score after alignment             -> ``ValueError``
      (a durable evidence table must not silently drop scored rows)
    * duplicate ``(date, name)`` rows              -> ``ValueError``
    * regime join is validated one-to-one per date (``merge(validate="m:1")``);
      duplicate regime dates -> ``ValueError``

    Output is canonically sorted by ``(date, name)`` with columns
    :func:`pick_table_columns` (``ticker`` exported as ``name`` per the #430
    schema).
    """
    if val.index.has_duplicates:
        raise ValueError(
            "build_pick_table: val has duplicate index labels; "
            "refusing ambiguous score alignment"
        )
    mu = pd.Series(mu)
    if mu.index.has_duplicates:
        raise ValueError(
            "build_pick_table: mu has duplicate index labels; "
            "refusing ambiguous score alignment"
        )
    missing = val.index.difference(mu.index)
    if len(missing) > 0:
        raise ValueError(
            f"build_pick_table: mu is missing scores for {len(missing)} val rows "
            f"(e.g. {list(missing[:5])}); refusing silent row loss"
        )
    scores = pd.to_numeric(mu.loc[val.index], errors="coerce").astype(float)
    n_bad = int((~np.isfinite(scores.to_numpy())).sum())
    if n_bad:
        raise ValueError(
            f"build_pick_table: {n_bad} non-finite score(s) after alignment; "
            "a durable evidence table must not silently drop scored rows"
        )
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(val["date"]).dt.normalize().to_numpy(),
            "name": val["ticker"].astype(str).to_numpy(),
            "score": scores.to_numpy(),
            str(label): val[label].astype(float).to_numpy(),
        },
        index=val.index,
    )
    if out.duplicated(subset=["date", "name"]).any():
        raise ValueError(
            "build_pick_table: duplicate (date, name) rows in val; the contract "
            "is one row per (date, name)"
        )
    if regimes is None or len(regimes) == 0:
        out["regime"] = None
    else:
        r = regimes[["date", "regime"]].copy()
        r["date"] = pd.to_datetime(r["date"]).dt.normalize()
        if r["date"].duplicated().any():
            raise ValueError(
                "build_pick_table: regimes has duplicate dates; the regime join "
                "must be one-to-one per date"
            )
        out = out.merge(r, on="date", how="left", validate="m:1")
    out["decile_rank"] = out.groupby("date")["score"].transform(decile_rank)
    out = out[pick_table_columns(label)]
    return out.sort_values(["date", "name"]).reset_index(drop=True)


def canonical_table_content_hash(
    table: pd.DataFrame, *, label: str = "fwd_60d_excess"
) -> str:
    """SHA256 of the table's CONTENT — scores/labels/regimes, not its shape.

    Same algorithm as RenQuant#430's ``canonical_table_content_hash`` (label
    column parameterized; digests are identical to #430's for
    ``label="fwd_60d_excess"`` — pinned by a golden-hash test):

    * ORDER-INDEPENDENT — canonically re-sorted by ``(date, name)`` here
      regardless of the caller's row order.
    * PLATFORM-STABLE FLOAT REPRESENTATION — ``score``/``<label>`` are
      formatted to fixed 10-decimal-place strings, not hashed as raw float64
      bytes.

    This is the field a regeneration must match to PROVE it reproduced the
    same evidence content — matching row/date/name counts alone does not.
    """
    cols = pick_table_columns(label)
    canon = table[cols].copy()
    canon["date"] = pd.to_datetime(canon["date"]).dt.strftime("%Y-%m-%d")
    canon["name"] = canon["name"].astype(str)
    canon["score"] = canon["score"].astype(float).map(lambda v: f"{v:.10f}")
    canon["decile_rank"] = canon["decile_rank"].astype(int)
    canon[str(label)] = canon[str(label)].astype(float).map(lambda v: f"{v:.10f}")
    canon["regime"] = canon["regime"].astype(str)
    canon = canon.sort_values(["date", "name"]).reset_index(drop=True)
    lines = [
        "|".join(str(v) for v in row)
        for row in canon.itertuples(index=False, name=None)
    ]
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _portable_ref(path: Path) -> str:
    """Best-effort portable reference for the sidecar: prefer the stable
    ``backtesting/...`` suffix over a machine-absolute path (never load-bearing
    — the sha256 stamped alongside is the durable identity; same rationale as
    RenQuant#430's ``_relpath``)."""
    text = str(Path(path).resolve())
    marker = "backtesting/"
    idx = text.find(marker)
    if idx != -1:
        return text[idx:]
    return text


def default_sidecar_path(parquet_path: Path) -> Path:
    """``<output stem>.manifest.json`` next to the parquet — same convention
    as RenQuant#430."""
    return Path(parquet_path).with_suffix("").with_suffix(".manifest.json")


def build_pick_table_manifest(
    table: pd.DataFrame,
    *,
    label: str,
    generator: str,
    generator_path: Path,
    manifest_input: Path,
    reference_artifact: Path,
    val_cut: str,
    val_start: str,
    val_end: str,
    label_lookahead_days: int | None,
    output_content_sha256: str,
    output_parquet_sha256: str,
) -> dict:
    """The sidecar reproducibility manifest — same structure as RenQuant#430's
    (schema / recipe / counts / output), extended with an explicit
    ``temporal_semantics`` block.

    ``generator_sha256`` / ``contract_module_sha256`` are CONTENT hashes of the
    code files that actually ran (a git commit hash stamped by the run itself
    is self-referential and goes stale — #430 review round 2), and
    ``output_content_sha256`` is the canonical logical content hash a future
    regeneration must reproduce (#430 review round 3).
    """
    contract_path = Path(__file__).resolve()
    horizon_text = (
        f"~{int(label_lookahead_days)} trading days"
        if label_lookahead_days
        else "the label horizon"
    )
    return {
        "schema": {
            "columns": pick_table_columns(label),
            "description": (
                "one row per (date, name); score = point-in-time model raw "
                "score (mu) from the same manifest contract the WF gate uses; "
                "decile_rank = cross-sectional decile within date, "
                f"0(worst)-9(best/top); {label} = realized forward label as "
                "loaded by the sanity panel; regime = live regime label at the "
                "pick date"
            ),
        },
        "recipe": {
            "generator": str(generator),
            "generator_sha256": sha256_file(Path(generator_path)),
            "contract_module": "renquant_backtesting/analysis/pick_table.py",
            "contract_module_sha256": sha256_file(contract_path),
            "manifest_input": _portable_ref(manifest_input),
            "manifest_input_sha256": sha256_file(Path(manifest_input)),
            "reference_artifact": _portable_ref(reference_artifact),
            "reference_artifact_sha256": sha256_file(Path(reference_artifact)),
            "label": str(label),
            "val_cut": str(val_cut),
            "val_start": str(val_start),
            "val_end": str(val_end),
        },
        "counts": {
            "n_rows": int(len(table)),
            "n_dates": int(table["date"].nunique()),
            "n_names": int(table["name"].nunique()),
        },
        "output": {
            "output_content_sha256": str(output_content_sha256),
            "output_content_sha256_note": (
                "the PROVENANCE ANCHOR — sha256 of the table's actual content "
                "(canonically re-sorted by (date, name), floats formatted to a "
                "fixed 10-decimal-place string), NOT its shape. A regeneration "
                "must match this value to prove it reproduced this evidence "
                "table's content; matching counts alone does not. Same "
                "algorithm as RenQuant#430 canonical_table_content_hash — see "
                "renquant_backtesting.analysis.pick_table."
            ),
            "output_parquet_sha256": str(output_parquet_sha256),
            "output_parquet_sha256_note": (
                "a SECONDARY, weaker transport hash of the literal on-disk "
                ".parquet bytes — detects local file corruption, but is NOT "
                "portable across parquet library/compression versions even for "
                "identical logical content. Verify reproducibility against "
                "output_content_sha256, not this field."
            ),
        },
        "temporal_semantics": {
            "research_only": True,
            "label_realized": (
                f"the `{label}` column is a REALIZED forward excess return: "
                f"for a row dated D it is only observable {horizon_text} AFTER "
                "D. Rows therefore embed future information relative to their "
                "own date."
            ),
            "label_lookahead_days": (
                int(label_lookahead_days) if label_lookahead_days else None
            ),
            "not_a_live_input": (
                "this artifact exists for offline research (Track A "
                "conditional pick-quality evaluation); it must NEVER be read "
                "as a live inference/scoring input. Canonical production "
                "output paths (data/ outside data/exp/, artifacts/) are "
                "mechanically refused at write time — see "
                "refuse_production_output_path()."
            ),
        },
        "note": (
            "wall-clock generation timestamp intentionally omitted for "
            "determinism/reproducibility framing (same as RenQuant#430); the "
            "generator/contract/input content hashes are the provenance "
            "record, not a run timestamp."
        ),
    }


def verify_pick_table(parquet_path: Path, sidecar_path: Path | None = None) -> dict:
    """Reload the parquet, recompute the canonical content hash, and check it
    (plus counts) against the sidecar manifest. Raises ``ValueError`` on any
    content/count mismatch; the parquet transport hash comparison is reported
    but non-fatal (it is not portable across parquet encoders by design).

    This is the check a regeneration/follow-up MUST run — writing the file is
    not evidence; matching ``output_content_sha256`` is.
    """
    parquet_path = Path(parquet_path)
    sidecar_path = (
        Path(sidecar_path) if sidecar_path is not None
        else default_sidecar_path(parquet_path)
    )
    sidecar = json.loads(Path(sidecar_path).read_text())
    label = str(sidecar["recipe"]["label"])
    table = pd.read_parquet(parquet_path)
    content = canonical_table_content_hash(table, label=label)
    expected = str(sidecar["output"]["output_content_sha256"])
    if content != expected:
        raise ValueError(
            f"pick-table content hash mismatch for {parquet_path}: "
            f"recomputed {content} != stamped {expected}"
        )
    actual_counts = {
        "n_rows": int(len(table)),
        "n_dates": int(table["date"].nunique()),
        "n_names": int(table["name"].nunique()),
    }
    stamped = sidecar.get("counts") or {}
    mismatched = {
        k: {"stamped": int(stamped[k]), "actual": v}
        for k, v in actual_counts.items()
        if k in stamped and int(stamped[k]) != v
    }
    if mismatched:
        raise ValueError(
            f"pick-table count mismatch for {parquet_path}: {mismatched}"
        )
    transport = sha256_file(parquet_path)
    return {
        "content_sha256": content,
        "content_verified": True,
        "counts_verified": True,
        "parquet_sha256": transport,
        "parquet_transport_match": (
            transport == str(sidecar["output"].get("output_parquet_sha256"))
        ),
    }


def refuse_production_output_path(path: Path) -> None:
    """Refuse canonical production data/artifact paths for the research dump.

    The pick table embeds REALIZED forward labels — a research-only artifact
    that must never sit where it could be mistaken for (or read as) a live
    inference input. The only ``data/`` location the research-artifact
    contract permits is the explicit experiment area ``data/exp/`` (the
    contract RenQuant#430 established); any other ``(^|/)data/`` path and any
    ``(^|/)artifacts/`` path (the canonical sim/prod artifact trees) is
    refused. Scratch/tmp/home research paths are unaffected.
    """
    posix = Path(path).resolve().as_posix()
    parts = posix.split("/")
    for i, seg in enumerate(parts[:-1]):
        if seg == "artifacts":
            raise ValueError(
                "refusing to write the research pick table under a canonical "
                f"artifacts/ tree: {posix} — it embeds realized forward labels "
                "(research-only); write to a data/exp/ or scratch path instead"
            )
        if seg == "data" and parts[i + 1] != RESEARCH_DATA_SEGMENT:
            raise ValueError(
                "refusing to write the research pick table under a canonical "
                f"data/ tree (only data/{RESEARCH_DATA_SEGMENT}/ is permitted "
                f"by the research-artifact contract): {posix}"
            )
