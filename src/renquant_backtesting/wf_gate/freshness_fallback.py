"""RFC#210 freshness fallback — the governed answer to a chronic gate REJECT.

DECISION RECORD. backtesting#101 (PREREG-DRAFT 2026-08-03 + same-night
amendment) under the operator's P0 directive, verbatim: "wf gate placebo这个
问题给我彻底解决掉！这太耽误事情了！prod又不动了！这是p0！". The amendment
withdrew any change to the gate criterion itself — the v3 enforcement stays
blocked on its own shadow-eval protocol (codex 2026-07-02 review; see
runner.py GATE_VERSION) — and kept the piece that needs no frozen margin:
implement the ALREADY-DECIDED freshness governance (RFC #210: no served
model >28 days; else serve the BEST from the last 10 days).

The policy, exactly:

  FALLBACK-PROMOTE the staged candidate iff ALL of
    1. the gate REJECTED it (this module is only consulted on REJECT, and
       verifies the stamp agrees — defence in depth);
    2. the served production model is STALE: trained more than
       ``MAX_SERVED_AGE_DAYS`` (28) before as-of;
    3. the candidate is RECENT: trained within ``CANDIDATE_WINDOW_DAYS``
       (10) of as-of;
    4. the candidate's stamped ``sanity_placebo_genuine_ic`` is a finite
       float >= 0.0 — never serve a candidate measured NEGATIVE net of the
       placebo (genuine_ic is used as an ORDINAL/sign quantity here, not a
       pass threshold: no frozen margin is introduced);
    5. NO DOWNWARD RATCHET: when the served model was ITSELF
       fallback-promoted, the candidate's genuine_ic must be STRICTLY
       greater than the served model's stamped fallback genuine_ic — a
       chain of fallbacks may only walk up.

Every verdict names each check's measured value. A promoted artifact is
stamped ``promotion_basis: "freshness_fallback_rfc210"`` plus the decision
inputs, so downstream can always distinguish governance-served from
gate-passed, and the silent-refusal sentinel counts the promotion as an
ACTION (the wrapper prints the action line on this path).

Field-type discipline (the green-over-malformed class, orch#770 /
pipeline#259): every identity/number read from an artifact must be the
expected non-empty type BEFORE comparison; absence or the wrong type is a
REFUSE with its own named reason, never a coerced pass.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

PROMOTION_BASIS = "freshness_fallback_rfc210"
MAX_SERVED_AGE_DAYS = 28      # RFC #210 SLA on the served model
CANDIDATE_WINDOW_DAYS = 10    # RFC #210 "best from the last 10 days"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, "rb") as fh:
            obj = json.loads(fh.read())
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _parse_date(value: Any) -> dt.date | None:
    if not (isinstance(value, str) and value):
        return None
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError:
        return None


def _finite_float(value: Any) -> float | None:
    """A real number or None — bool is NOT a number here, NaN is refused."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def decide(prod_path: Path, staging_path: Path, as_of: dt.date) -> dict[str, Any]:
    """The pure decision. Returns a verdict dict; never raises on bad input."""
    checks: list[dict[str, Any]] = []
    verdict: dict[str, Any] = {
        "policy": PROMOTION_BASIS,
        "as_of": as_of.isoformat(),
        "prod_artifact": str(prod_path),
        "staging_artifact": str(staging_path),
        "checks": checks,
    }

    def refuse(name: str, why: str, **extra: Any) -> dict[str, Any]:
        checks.append({"check": name, "ok": False, "why": why, **extra})
        verdict["decision"] = "REFUSE"
        verdict["refused_on"] = name
        return verdict

    def ok(name: str, **extra: Any) -> None:
        checks.append({"check": name, "ok": True, **extra})

    prod = _read_json(prod_path)
    if prod is None:
        return refuse("prod_readable", f"production artifact unreadable: {prod_path}")
    staging = _read_json(staging_path)
    if staging is None:
        return refuse("staging_readable", f"staging artifact unreadable: {staging_path}")

    # 1. the gate actually rejected the candidate (stamped verdict).
    wf = ((staging.get("metadata") or {}).get("wf_gate_metadata")
          if isinstance(staging.get("metadata"), dict) else None)
    if not isinstance(wf, dict) or not wf:
        return refuse("staging_gate_stamp",
                      "staging artifact carries no metadata.wf_gate_metadata — "
                      "the gate never evaluated it; nothing to fall back from")
    # The runner's OWN stamped verdict is ``passed`` (an explicit bool at the
    # head of wf_gate_metadata). [codex on backtesting#102]: an absent,
    # corrupted, or never-recorded decision must never become permission —
    # only an explicit False proceeds. (The first draft read
    # ``gate_verdict_before_override``, an ORCHESTRATOR promotion-time
    # provenance field absent from staging stamps — the read-the-contract
    # class; the producer contract needed no repair, the consumer did.)
    gate_verdict = wf.get("passed")
    if not isinstance(gate_verdict, bool):
        return refuse("gate_rejected",
                      f"stamped gate verdict `passed` is not an explicit "
                      f"boolean (got {gate_verdict!r}) — an absent or "
                      f"corrupted decision is not permission; re-run the gate",
                      stamped_verdict=gate_verdict)
    if gate_verdict is True:
        return refuse("gate_rejected",
                      "the gate PASSED this candidate — use the normal promote "
                      "path, not the fallback")
    ok("gate_rejected", stamped_verdict=gate_verdict)

    # 2. served model stale beyond the SLA.
    prod_trained = _parse_date(prod.get("trained_date"))
    if prod_trained is None:
        return refuse("prod_trained_date",
                      f"production trained_date is not a parseable ISO date "
                      f"(got {prod.get('trained_date')!r})")
    staleness = (as_of - prod_trained).days
    if staleness <= MAX_SERVED_AGE_DAYS:
        return refuse("prod_stale",
                      f"served model is {staleness}d old (<= {MAX_SERVED_AGE_DAYS}d "
                      f"SLA) — the fallback does not apply to a fresh book",
                      prod_trained=prod_trained.isoformat(),
                      staleness_days=staleness)
    ok("prod_stale", prod_trained=prod_trained.isoformat(), staleness_days=staleness)

    # 3. candidate recent enough.
    cand_trained = _parse_date(staging.get("trained_date"))
    if cand_trained is None:
        return refuse("candidate_trained_date",
                      f"staging trained_date is not a parseable ISO date "
                      f"(got {staging.get('trained_date')!r})")
    cand_age = (as_of - cand_trained).days
    if cand_age < 0 or cand_age > CANDIDATE_WINDOW_DAYS:
        return refuse("candidate_recent",
                      f"candidate trained {cand_trained.isoformat()} is outside "
                      f"the {CANDIDATE_WINDOW_DAYS}d window (age {cand_age}d)",
                      candidate_age_days=cand_age)
    ok("candidate_recent", candidate_trained=cand_trained.isoformat(),
       candidate_age_days=cand_age)

    # 4. genuine_ic present, finite, non-negative.
    genuine = _finite_float(wf.get("sanity_placebo_genuine_ic"))
    if genuine is None:
        return refuse("genuine_ic_present",
                      f"stamped sanity_placebo_genuine_ic is not a finite number "
                      f"(got {wf.get('sanity_placebo_genuine_ic')!r})")
    if genuine < 0.0:
        return refuse("genuine_ic_nonnegative",
                      f"candidate measures NEGATIVE net of placebo "
                      f"(genuine_ic={genuine:+.6f}) — never served",
                      genuine_ic=genuine)
    ok("genuine_ic_nonnegative", genuine_ic=genuine)

    # 5. no downward ratchet across chained fallbacks.
    prod_meta = prod.get("metadata") if isinstance(prod.get("metadata"), dict) else {}
    prod_basis = prod_meta.get("promotion_basis")
    if prod_basis == PROMOTION_BASIS:
        prior = _finite_float(prod_meta.get("fallback_genuine_ic"))
        if prior is None:
            return refuse("ratchet_prior_readable",
                          "served model is fallback-promoted but carries no "
                          "finite fallback_genuine_ic — cannot prove the chain "
                          "walks up; refusing rather than guessing")
        if genuine <= prior:
            return refuse("ratchet_up_only",
                          f"candidate genuine_ic {genuine:+.6f} does not exceed "
                          f"the served fallback's {prior:+.6f} — a chain of "
                          f"fallbacks may only walk up",
                          genuine_ic=genuine, prior_genuine_ic=prior)
        ok("ratchet_up_only", genuine_ic=genuine, prior_genuine_ic=prior)
    else:
        ok("ratchet_up_only", note="served model is gate-passed; no ratchet bar",
           prod_promotion_basis=prod_basis)

    verdict["decision"] = "FALLBACK_PROMOTE"
    verdict["genuine_ic"] = genuine
    verdict["prod_staleness_days"] = staleness
    return verdict


def stamp(staging_path: Path, verdict: dict[str, Any]) -> None:
    """Write the promotion-basis stamp into the staging artifact, atomically.

    Refuses (ValueError) unless the verdict IS a FALLBACK_PROMOTE — a stamp
    without a decision would be an unexplained governance marker.
    """
    if verdict.get("decision") != "FALLBACK_PROMOTE":
        raise ValueError("stamp() requires a FALLBACK_PROMOTE verdict")
    obj = _read_json(staging_path)
    if obj is None:
        raise ValueError(f"staging artifact unreadable for stamping: {staging_path}")
    meta = obj.setdefault("metadata", {})
    if not isinstance(meta, dict):
        raise ValueError("staging metadata is not an object; refusing to replace it")
    meta["promotion_basis"] = PROMOTION_BASIS
    meta["fallback_genuine_ic"] = verdict["genuine_ic"]
    meta["fallback_as_of"] = verdict["as_of"]
    meta["fallback_prod_staleness_days"] = verdict["prod_staleness_days"]
    fd, tmp = tempfile.mkstemp(dir=str(staging_path.parent),
                               prefix=staging_path.name + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh)
        os.replace(tmp, staging_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prod", required=True, type=Path)
    ap.add_argument("--staging", required=True, type=Path)
    ap.add_argument("--as-of", default=None,
                    help="YYYY-MM-DD (default: today)")
    ap.add_argument("--stamp", action="store_true",
                    help="on FALLBACK_PROMOTE, stamp the staging artifact")
    args = ap.parse_args(argv)
    as_of = (dt.date.fromisoformat(args.as_of) if args.as_of
             else dt.date.today())
    verdict = decide(args.prod, args.staging, as_of)
    if args.stamp and verdict.get("decision") == "FALLBACK_PROMOTE":
        stamp(args.staging, verdict)
        verdict["stamped"] = True
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict.get("decision") == "FALLBACK_PROMOTE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
