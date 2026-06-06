#!/usr/bin/env python3
"""Assemble Track B evidence into a promotion verdict.

This is intentionally a small JSON assembler.  It does not run walk-forward
simulation or model scoring; callers pass already-produced evidence JSON files
from the Track B chain and this module extracts the required fields.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MIN_BULL_CALM_IC = 0.02
DEFAULT_MIN_AA_MEAN_IC = 0.02
DEFAULT_MAX_ABS_SHUFFLE_IC = 0.005
DEFAULT_MAX_ABS_PLACEBO_IC = 0.005
DEFAULT_MAX_PLACEBO_RATIO = 0.5
REQUIRED_REGIME = "BULL_CALM"
REQUIRED_PLACEBO_SHIFT_DAYS = 120


@dataclass(frozen=True)
class ExtractedValue:
    value: Any
    source: str


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _get_path(payload: Any, path: str) -> Any:
    cur = payload
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _first_path(
    payloads: Iterable[tuple[str, dict[str, Any]]],
    paths: Iterable[str],
) -> ExtractedValue | None:
    for source, payload in payloads:
        for path in paths:
            value = _get_path(payload, path)
            if value is not None:
                return ExtractedValue(value=value, source=f"{source}:{path}")
    return None


def _first_number(
    payloads: Iterable[tuple[str, dict[str, Any]]],
    paths: Iterable[str],
) -> ExtractedValue | None:
    found = _first_path(payloads, paths)
    if found is None:
        return None
    number = _finite_float(found.value)
    if number is None:
        return None
    return ExtractedValue(value=number, source=found.source)


def _normalize_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose common evidence roots without callers needing exact wrapping."""
    out = [payload]
    plan = payload.get("plan")
    if isinstance(plan, dict):
        out.append(plan)
    wf = payload.get("wf_gate_metadata")
    if isinstance(wf, dict):
        out.append(wf)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        wf_nested = metadata.get("wf_gate_metadata")
        if isinstance(wf_nested, dict):
            out.append(wf_nested)
    return out


def _flatten_payloads(
    evidence: Iterable[tuple[str, dict[str, Any]]],
) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for source, payload in evidence:
        for idx, item in enumerate(_normalize_payload(payload)):
            suffix = "" if idx == 0 else f"#root{idx}"
            out.append((f"{source}{suffix}", item))
    return out


def _extract_bull_calm_ic(
    payloads: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    regime_paths = (
        "sanity_regime_ic.regimes.BULL_CALM",
        "by_regime.BULL_CALM",
        "regime_ic.BULL_CALM",
        "per_regime_ic.BULL_CALM",
    )
    row = _first_path(payloads, regime_paths)
    stats = row.value if row is not None and isinstance(row.value, dict) else {}
    mean = _finite_float(stats.get("mean_ic"))
    if mean is None:
        mean = _finite_float(stats.get("ic"))
    return {
        "regime": REQUIRED_REGIME,
        "mean_ic": mean,
        "n_dates": stats.get("n_dates"),
        "n_rows": stats.get("n_rows", stats.get("n_raw_rows")),
        "hit_rate": stats.get("hit_rate"),
        "eligible": stats.get("eligible"),
        "source": row.source if row is not None else None,
    }


def _extract_shuffle(payloads: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    value = _first_number(
        payloads,
        (
            "sanity_shuffled_ic",
            "shuffle_ic",
            "shuffled_ic",
            "sanity.shuffle_ic",
            "interpretation.shuffle_ic",
        ),
    )
    return {
        "ic": value.value if value is not None else None,
        "source": value.source if value is not None else None,
    }


def _extract_aa(payloads: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    mean = _first_number(
        payloads,
        (
            "aa_mean",
            "aa.mean_ic",
            "aa.ic_mean",
            "a_a.mean_ic",
            "sanity_aa_ic",
            "sanity.aa_mean",
        ),
    )
    std = _first_number(
        payloads,
        (
            "aa_std",
            "aa.std_ic",
            "aa.ic_std",
            "a_a.std_ic",
            "sanity.aa_std",
        ),
    )
    seeds = _first_path(payloads, ("aa_seeds", "aa.seeds", "a_a.seeds"))
    return {
        "mean_ic": mean.value if mean is not None else None,
        "std_ic": std.value if std is not None else None,
        "seeds": seeds.value if seeds is not None else None,
        "source": mean.source if mean is not None else None,
    }


def _iter_shift_rows(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for key in ("shift_diagnostics", "placebo_shift_diagnostics"):
        rows = payload.get(key)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    yield row
    by_regime = payload.get("by_regime_shift_diagnostics")
    if isinstance(by_regime, dict):
        for rows in by_regime.values():
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        yield row


def _extract_time_shift_120(
    payloads: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    for source, payload in payloads:
        for row in _iter_shift_rows(payload):
            if int(row.get("shift_days") or -1) != REQUIRED_PLACEBO_SHIFT_DAYS:
                continue
            ic = _finite_float(
                row.get("model_placebo_ic", row.get("ic", row.get("placebo_ic")))
            )
            return {
                "shift_days": REQUIRED_PLACEBO_SHIFT_DAYS,
                "placebo_ic": ic,
                "aligned_real_ic": _finite_float(row.get("aligned_real_ic")),
                "label_autocorr_ic": _finite_float(row.get("label_autocorr_ic")),
                "n_dates": row.get("n_dates"),
                "n_rows": row.get("n_rows"),
                "source": source,
            }
    for source, payload in payloads:
        gate_shift = _finite_float(payload.get("sanity_placebo_gate_shift_days"))
        if gate_shift is None or int(gate_shift) != REQUIRED_PLACEBO_SHIFT_DAYS:
            continue
        return {
            "shift_days": REQUIRED_PLACEBO_SHIFT_DAYS,
            "placebo_ic": _finite_float(payload.get("sanity_placebo_ic")),
            "aligned_real_ic": _finite_float(
                payload.get("sanity_placebo_aligned_real_ic")
            ),
            "label_autocorr_ic": None,
            "n_dates": None,
            "n_rows": None,
            "source": f"{source}:sanity_placebo_ic",
        }
    return {
        "shift_days": REQUIRED_PLACEBO_SHIFT_DAYS,
        "placebo_ic": None,
        "aligned_real_ic": None,
        "label_autocorr_ic": None,
        "n_dates": None,
        "n_rows": None,
        "source": None,
    }


def _placebo_threshold(aligned_real_ic: float | None, *, max_abs: float, ratio: float) -> float:
    if aligned_real_ic is None:
        return float(max_abs)
    return max(float(max_abs), float(ratio) * abs(float(aligned_real_ic)))


def assemble_track_b_verdict(
    evidence: Iterable[tuple[str, dict[str, Any]]],
    *,
    min_bull_calm_ic: float = DEFAULT_MIN_BULL_CALM_IC,
    min_aa_mean_ic: float = DEFAULT_MIN_AA_MEAN_IC,
    max_abs_shuffle_ic: float = DEFAULT_MAX_ABS_SHUFFLE_IC,
    max_abs_placebo_ic: float = DEFAULT_MAX_ABS_PLACEBO_IC,
    max_placebo_ratio: float = DEFAULT_MAX_PLACEBO_RATIO,
) -> dict[str, Any]:
    payloads = _flatten_payloads(evidence)
    bull = _extract_bull_calm_ic(payloads)
    shuffle = _extract_shuffle(payloads)
    aa = _extract_aa(payloads)
    shift120 = _extract_time_shift_120(payloads)

    blocked: list[str] = []
    bull_ic = _finite_float(bull.get("mean_ic"))
    if bull_ic is None:
        blocked.append("BULL_CALM per-regime IC missing")
    elif bull_ic < float(min_bull_calm_ic):
        blocked.append(
            f"BULL_CALM per-regime IC {bull_ic:+.4f} < {min_bull_calm_ic:+.4f}"
        )

    shuffle_ic = _finite_float(shuffle.get("ic"))
    if shuffle_ic is None:
        blocked.append("shuffle IC missing")
    elif abs(shuffle_ic) > float(max_abs_shuffle_ic):
        blocked.append(
            f"shuffle IC {shuffle_ic:+.4f} exceeds +/-{max_abs_shuffle_ic:.4f}"
        )

    aa_mean = _finite_float(aa.get("mean_ic"))
    if aa_mean is None:
        blocked.append("A/A mean IC missing")
    elif aa_mean < float(min_aa_mean_ic):
        blocked.append(f"A/A mean IC {aa_mean:+.4f} < {min_aa_mean_ic:+.4f}")

    placebo_ic = _finite_float(shift120.get("placebo_ic"))
    aligned_real_ic = _finite_float(shift120.get("aligned_real_ic"))
    placebo_limit = _placebo_threshold(
        aligned_real_ic,
        max_abs=max_abs_placebo_ic,
        ratio=max_placebo_ratio,
    )
    if placebo_ic is None:
        blocked.append("+120d time-shift placebo IC missing")
    elif abs(placebo_ic) > placebo_limit:
        blocked.append(
            f"+120d time-shift placebo IC {placebo_ic:+.4f} "
            f"exceeds +/-{placebo_limit:.4f}"
        )

    passed = not blocked
    return {
        "track": "Track B",
        "required_evidence": {
            "bull_calm_per_regime_ic": bull,
            "shuffle": shuffle,
            "aa": aa,
            "time_shift_placebo_120d": {
                **shift120,
                "threshold": placebo_limit,
            },
        },
        "promotion_verdict": {
            "passed": passed,
            "recommendation": "PROMOTE" if passed else "REJECT",
            "blocked_reasons": blocked,
            "thresholds": {
                "min_bull_calm_ic": float(min_bull_calm_ic),
                "min_aa_mean_ic": float(min_aa_mean_ic),
                "max_abs_shuffle_ic": float(max_abs_shuffle_ic),
                "max_abs_placebo_ic": float(max_abs_placebo_ic),
                "max_placebo_ratio": float(max_placebo_ratio),
                "required_regime": REQUIRED_REGIME,
                "required_time_shift_days": REQUIRED_PLACEBO_SHIFT_DAYS,
            },
        },
    }


def _expand_evidence_path(raw: str) -> list[Path]:
    path = Path(raw)
    if path.is_dir():
        return sorted(path.glob("*.json"))
    return [path]


def load_evidence(paths: Iterable[str]) -> list[tuple[str, dict[str, Any]]]:
    evidence: list[tuple[str, dict[str, Any]]] = []
    for raw in paths:
        for path in _expand_evidence_path(raw):
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                raise ValueError(f"evidence root must be a JSON object: {path}")
            evidence.append((str(path), payload))
    if not evidence:
        raise ValueError("no evidence JSON files provided")
    return evidence


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--evidence",
        action="append",
        required=True,
        help="Evidence JSON file or directory. May be passed more than once.",
    )
    ap.add_argument("--output", default="", help="Optional output JSON path.")
    ap.add_argument("--min-bull-calm-ic", type=float, default=DEFAULT_MIN_BULL_CALM_IC)
    ap.add_argument("--min-aa-mean-ic", type=float, default=DEFAULT_MIN_AA_MEAN_IC)
    ap.add_argument(
        "--max-abs-shuffle-ic",
        type=float,
        default=DEFAULT_MAX_ABS_SHUFFLE_IC,
    )
    ap.add_argument(
        "--max-abs-placebo-ic",
        type=float,
        default=DEFAULT_MAX_ABS_PLACEBO_IC,
    )
    ap.add_argument(
        "--max-placebo-ratio",
        type=float,
        default=DEFAULT_MAX_PLACEBO_RATIO,
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    verdict = assemble_track_b_verdict(
        load_evidence(args.evidence),
        min_bull_calm_ic=float(args.min_bull_calm_ic),
        min_aa_mean_ic=float(args.min_aa_mean_ic),
        max_abs_shuffle_ic=float(args.max_abs_shuffle_ic),
        max_abs_placebo_ic=float(args.max_abs_placebo_ic),
        max_placebo_ratio=float(args.max_placebo_ratio),
    )
    text = json.dumps(verdict, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
