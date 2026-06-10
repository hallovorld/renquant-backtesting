"""M6 reproduction — the §5.2 time-shift placebo FAIL is consistent with an
overlapping-label confound.

Two modes, both cheap (no model retrain — reuses the gate's already-stamped
Layer-1a diagnostics and the raw label panel):

* ``--mode autocorr`` — reproduce the cross-sectional label-autocorrelation decay
  across {1x,2x,3x}×horizon for fwd_5d / fwd_20d / fwd_60d. The decisive root data:
  only the daily-sampled fwd_60d label is autocorrelated at the gate's 2×-horizon
  (=120 trading-date rows) shift point, so its time-shift placebo is a
  confounded diagnostic unless calibrated against label persistence.

* ``--mode regime`` — from a stamped WF-gate artifact
  (``metadata.wf_gate_metadata.model_placebo_profile``), decompose the model
  placebo IC by regime at the gate shift and compute
  ``corr(placebo_ic, label_autocorr_ic)`` across regimes. A large positive
  correlation supports an overlapping-label-confound diagnosis, but it is not a
  proof that every leakage path is absent.

See ``RenQuant/doc/research/2026-06-10-m6-placebo-gate-verdict.md`` for the full
verdict and the Layer-1b fix path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# (label_column, horizon_days) for the canonical fwd labels.
_LABELS: tuple[tuple[str, int], ...] = (
    ("fwd_5d_excess", 5),
    ("fwd_20d_excess", 20),
    ("fwd_60d_excess", 60),
)


def cross_sectional_autocorr(
    df: pd.DataFrame, label: str, lag: int, *, min_names: int = 20
) -> tuple[float, int]:
    """Mean over trading-date rows of cross-sectional corr(label_t, label_{t-lag}).

    Returns (mean_autocorr, n_dates_used). This mirrors the §2 measurement in the
    2026-06-08 overlapping-label RFC and the gate's ``label_autocorr_ic``.
    """
    piv = (
        df.dropna(subset=[label])
        .pivot_table(index="date", columns="ticker", values=label)
        .sort_index()
    )
    vals: list[float] = []
    for i in range(lag, len(piv)):
        a = piv.iloc[i]
        b = piv.iloc[i - lag]
        mask = a.notna() & b.notna()
        if int(mask.sum()) >= min_names:
            c = np.corrcoef(a[mask], b[mask])[0, 1]
            if np.isfinite(c):
                vals.append(float(c))
    return (float(np.mean(vals)) if vals else float("nan"), len(vals))


def run_autocorr(rawlabel_path: Path) -> dict[str, Any]:
    cols = ["ticker", "date", *[lab for lab, _ in _LABELS]]
    df = pd.read_parquet(rawlabel_path, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    out: dict[str, Any] = {}
    print(f"{'label':18s} {'h':>4s} {'AC@1h':>9s} {'AC@2h':>9s} {'AC@3h':>9s}  n(@2h)")
    for label, h in _LABELS:
        a1, _ = cross_sectional_autocorr(df, label, h)
        a2, n2 = cross_sectional_autocorr(df, label, 2 * h)
        a3, _ = cross_sectional_autocorr(df, label, 3 * h)
        out[label] = {
            "horizon_trading_date_rows": h,
            "ac_1h": a1,
            "ac_2h": a2,
            "ac_3h": a3,
            "n_dates_2h": n2,
        }
        print(f"{label:18s} {h:4d} {a1:+9.4f} {a2:+9.4f} {a3:+9.4f}  {n2}")
    prod = out["fwd_60d_excess"]
    short = out["fwd_5d_excess"]
    print(
        f"\nDIAGNOSIS: prod label fwd_60d is autocorrelated at the gate shift "
        f"(AC@2h={prod['ac_2h']:+.4f}); short-horizon fwd_5d is decorrelated "
        f"(AC@2h={short['ac_2h']:+.4f}). This supports treating the fwd_60d "
        "time-shift placebo as persistence-confounded."
    )
    return {
        "mode": "autocorr",
        "rawlabel_path": str(rawlabel_path),
        "lag_semantics": "trading_date_row_offsets",
        "labels": out,
        "diagnosis": (
            "fwd_60d time-shift placebo is consistent with an "
            "overlapping-label persistence confound"
        ),
    }


def _nested_get(obj: Any, key: str) -> Any:
    """First match for ``key`` anywhere in a nested dict/list, else None."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _nested_get(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _nested_get(v, key)
            if found is not None:
                return found
    return None


def run_regime(artifact_path: Path, *, mult: str = "2x", min_dates: int = 25) -> dict[str, Any]:
    artifact = json.loads(artifact_path.read_text())
    profile = _nested_get(artifact, "model_placebo_profile")
    if not profile:
        raise SystemExit(
            "artifact has no model_placebo_profile (Layer-1a not stamped) — "
            "run the gate first or pass a stamped staging artifact"
        )
    pooled = profile.get("pooled", {}).get(mult, {})
    print(f"=== POOLED gate point (shift={mult}) ===")
    ar = float(pooled.get("aligned_real_ic"))
    pl = float(pooled.get("placebo_ic"))
    print(f"aligned_real_ic = {ar:+.4f}")
    print(f"placebo_ic      = {pl:+.4f}   (0.5x threshold = {0.5 * ar:+.4f}; ratio = {pl / ar:.3f})")
    print(f"genuine_ic      = {float(pooled.get('genuine_ic')):+.4f}")
    print(f"label_autocorr  = {float(pooled.get('label_autocorr_ic')):+.4f}")

    rows: list[dict[str, Any]] = []
    print(f"\n=== Eligible regimes (n_dates>={min_dates}) at {mult} ===")
    print(f"{'regime':14s} {'placebo':>9s} {'lbl_auto':>9s} {'aligned':>9s} {'genuine':>9s} {'n':>5s}")
    for regime, blocks in (profile.get("per_regime") or {}).items():
        b = blocks.get(mult, {})
        n = int(b.get("n_dates") or 0)
        if n < min_dates:
            continue
        row = {
            "regime": regime,
            "placebo_ic": float(b["placebo_ic"]),
            "label_autocorr_ic": float(b["label_autocorr_ic"]),
            "aligned_real_ic": float(b["aligned_real_ic"]),
            "genuine_ic": float(b["genuine_ic"]),
            "n_dates": n,
        }
        rows.append(row)
        print(
            f"{regime:14s} {row['placebo_ic']:+9.4f} {row['label_autocorr_ic']:+9.4f} "
            f"{row['aligned_real_ic']:+9.4f} {row['genuine_ic']:+9.4f} {n:5d}"
        )

    corr = float("nan")
    if len(rows) >= 3:
        corr = float(
            np.corrcoef(
                [r["placebo_ic"] for r in rows],
                [r["label_autocorr_ic"] for r in rows],
            )[0, 1]
        )
    print(
        f"\ncorr(placebo_ic, label_autocorr_ic) across regimes = {corr:+.3f}"
        "\nDIAGNOSIS: a large positive correlation supports the hypothesis that "
        "the model placebo is tracking target persistence. It is not, by itself, "
        "proof that every leakage path is absent."
    )
    return {
        "mode": "regime",
        "artifact_path": str(artifact_path),
        "shift_multiple": mult,
        "min_dates": min_dates,
        "pooled": pooled,
        "regimes": rows,
        "corr_placebo_autocorr": corr,
        "diagnosis": (
            "large positive regime correlation is consistent with an "
            "overlapping-label persistence confound; it is not a standalone "
            "leakage-exoneration test"
        ),
    }


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("autocorr", "regime"), required=True)
    p.add_argument(
        "--rawlabel",
        type=Path,
        help="path to alpha158_*_fundamental_dataset_rawlabel.parquet (autocorr mode)",
    )
    p.add_argument(
        "--artifact",
        type=Path,
        help="path to a stamped WF-gate staging artifact JSON (regime mode)",
    )
    p.add_argument("--mult", default="2x", choices=("1x", "2x", "3x"))
    p.add_argument("--min-dates", type=int, default=25)
    p.add_argument("--out", type=Path, help="optional JSON output path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "autocorr":
        if not args.rawlabel:
            raise SystemExit("--rawlabel is required for --mode autocorr")
        payload = run_autocorr(args.rawlabel)
    else:
        if not args.artifact:
            raise SystemExit("--artifact is required for --mode regime")
        payload = run_regime(args.artifact, mult=args.mult, min_dates=args.min_dates)
    if args.out:
        write_output(args.out, payload)


if __name__ == "__main__":
    main()
