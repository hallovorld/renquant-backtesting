"""RFC#210 freshness fallback — the governed answer to a chronic gate REJECT.

DECISION RECORD. backtesting#101 (PREREG-DRAFT 2026-08-03 + same-night
amendment) under the operator's P0 directive, verbatim: "wf gate placebo这个
问题给我彻底解决掉！这太耽误事情了！prod又不动了！这是p0！". The amendment
withdrew any change to the gate criterion itself — the v3 enforcement stays
blocked on its own shadow-eval protocol (codex 2026-07-02 review; see
runner.py GATE_VERSION) — and kept the piece that needs no frozen margin:
implement the ALREADY-DECIDED freshness governance (RFC #210: no served
model >28 days; else serve the BEST from the last 10 days).

AMENDMENT A4 (RFC#210, orchestrator design doc
``doc/design/2026-06-30-model-freshness-governance.md``, 2026-08-30). The
first implementation admitted any candidate with a non-negative genuine_ic
and then RATCHETED it against the served fallback's stamped genuine_ic. That
rule promoted (2026-08-04) an artifact whose signal is indistinguishable from
a 120-day-shifted placebo (genuine_ic +0.0029 against the 0.02 bar), and then
could never promote again (later candidates +0.0006..+0.0000). A comparison
against a model that itself failed the floor is not a quality predicate.
Under A4 a Pillar-3 promotion requires the QUALITY FLOOR and INFRA-ONLY
failure classes; the ratchet is removed.

The policy, exactly:

  FALLBACK-PROMOTE the staged candidate iff ALL of
    1. the gate REJECTED it (this module is only consulted on REJECT, and
       verifies the stamp agrees — defence in depth);
    2. the served production model is STALE: trained more than
       ``MAX_SERVED_AGE_DAYS`` (28) before as-of;
    3. the candidate is RECENT: trained within ``CANDIDATE_WINDOW_DAYS``
       (10) of as-of;
    4. QUALITY FLOOR: the candidate's stamped ``sanity_placebo_genuine_ic``
       (= aligned_real_ic − placebo_ic, the Fix-3 difference test) is a
       finite float >= ``FALLBACK_GENUINE_IC_FLOOR`` — the SAME 0.02 the §5.2
       sanity bar uses (``runner.PLACEBO_GENUINE_IC_MARGIN``; one constant,
       one place). The served artifact's own genuine_ic is REPORTED in the
       verdict for context and never compared;
    5. INFRA-ONLY FAILURE CLASSES: every failing sub-verdict in the
       candidate's stamp must be in ``INFRA_ONLY_FAILURE_CLASSES``
       (§4.3.1). Any QUALITY/SUBSTANCE class — sub-SPY / negative ΔSharpe
       (Fix-4), shuffled-label leakage, regime sanity IC, trade
       monotonicity, recipe mismatch, a diagnostic-only stamp — or any
       failure the classifier cannot attribute is fail-closed, regardless
       of genuine_ic.

AMENDMENT A4-T1 (TEMPORARY, 2026-08-31 to 2026-09-07, fail-closed). The
served model lapsed on day 29 and the current recipe produces zero trades in
all WF cuts. A4-T1 adds a NARROW, TIME-LIMITED, COUPLED bypass with TWO
eligibility paths:

  Path 1 (structural zero-trade): ``_is_zero_trade_structural(failures, wf)``
  — wf_reason "zero trades across all", substance classes EXACTLY the four
  zero-trade classes, trade-dependent details "no round-trip".

  Path 2 (candidate exception): ``_is_a4t1_candidate(staging_path, wf,
  staging)`` — run-ID parsed exactly from the filename (regex, not
  substring), wf_gate_metadata content verified by SHA-256 digest against a
  hardcoded constant, single-consumption (refuses after promotion_basis is
  stamped). The digest and run-ID are recorded in the verdict and stamp for
  downstream auditability.

Either path, combined with ``_a4t1_active(as_of)``, gates BOTH the floor
relaxation (0.001) and the failure-class exception. The standing A4 constants
are unchanged; the bypass is a separate code path.
Operator authorization: orchestrator session 428feb92, 2026-08-31.

Why the enumerated infra list is {placebo_ceiling} today: §4.3.1 admits the
structural placebo floor (Fix-3) as bypassable ONLY with the difference test
as its predicate — check 4 IS that predicate. The other infra classes named
there (phase/timeout, path-not-found) abort the runner before any stamp is
written, so they never reach this module as a ``passed=False`` stamp; a
partially-executed sim ("N/3 sim cuts failed execution") is stamped but its
WF economics were never measured, and §4.3.3's independent OOS floor for
that case has no implemented predicate — fail-closed.

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
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from renquant_backtesting.wf_gate.runner import PLACEBO_GENUINE_IC_MARGIN, SHUF_IC_MAX

PROMOTION_BASIS = "freshness_fallback_rfc210"
MAX_SERVED_AGE_DAYS = 28      # RFC #210 SLA on the served model
CANDIDATE_WINDOW_DAYS = 10    # RFC #210 "best from the last 10 days"

# A4: the Fix-3 difference floor. NOT a second copy of the number — it IS the
# §5.2 sanity bar (0.02, FROZEN 2026-07-02). The sanity battery's shadow
# verdict is strictly-greater; the fallback floor is >= per A4's wording
# ("genuine_ic >= 0.02"), which differs only at exact equality.
FALLBACK_GENUINE_IC_FLOOR = PLACEBO_GENUINE_IC_MARGIN

# §4.3.1 enumerated, CLOSED list of failure classes the fallback may act on.
# See the module docstring for why it is this short. Everything else is
# QUALITY/SUBSTANCE (fail-closed), including anything unclassified.
INFRA_ONLY_FAILURE_CLASSES = frozenset({"placebo_ceiling"})

# ── A4-T1 TEMPORARY BYPASS (2026-08-31 to 2026-09-07, fail-closed) ──────
# Operator authorization: orchestrator session 428feb92, 2026-08-31.
# Operator chose option 3 ("临时降低 A4 quality floor") from three options
# presented after the manual promote showed: served model day-29 lapse
# (trained 2026-08-02), WF gate REJECT (zero trades across all 3 cuts),
# fallback REFUSE (genuine_ic=0.0016 < 0.02 floor + substance failures
# from zero-trade outcomes).
# RESTORE: remove this block after expiry or when a recipe produces
# non-zero WF trades AND meets genuine_ic >= 0.02.
_A4T1_START = dt.date(2026, 8, 31)
_A4T1_EXPIRY = dt.date(2026, 9, 7)
_A4T1_FLOOR = 0.001
_A4T1_ZERO_TRADE_CLASSES = frozenset({
    "wf_benchmark_economics", "trade_contract",
    "trade_monotonicity", "alpha_economics",
})
_A4T1_CANDIDATE_RUN_ID = "20260831T141820Z"
_A4T1_CANDIDATE_WF_DIGEST = (
    "7cff5b7c27a8ced62f70b25fdd146479cb11e5f0d47aee14e22b22249f0fbfbc"
)
_RUN_ID_RE = re.compile(r"weekly_(\d{8}T\d{6}Z)\.staging\.json$")


def _wf_digest(wf: dict) -> str:
    return hashlib.sha256(
        json.dumps(wf, sort_keys=True).encode("utf-8"),
    ).hexdigest()


def _a4t1_active(as_of: dt.date) -> bool:
    return _A4T1_START <= as_of <= _A4T1_EXPIRY


def _is_zero_trade_structural(
    failures: list[dict[str, Any]], wf: dict[str, Any],
) -> bool:
    """True iff the candidate is a complete zero-trade WF outcome.

    Requires ALL of:
    - wf_reason contains "zero trades across all" (anti-vacuity: the
      runner measured all cuts and none produced trades; a partial
      "zero trades in one cut" does NOT match);
    - the substance failure classes are EXACTLY _A4T1_ZERO_TRADE_CLASSES
      (not a subset — all four must be present, and no additional
      substance classes);
    - trade_contract/monotonicity/alpha_economics detail mentions
      "no round-trip" (anti-vacuity: absence-of-data, not a failed
      quality check on actual trades).
    """
    wf_reason = wf.get("wf_reason")
    if not (isinstance(wf_reason, str)
            and "zero trades across all" in wf_reason):
        return False
    substance = {f["class"] for f in failures if f["kind"] != "infra"}
    if substance != _A4T1_ZERO_TRADE_CLASSES:
        return False
    for f in failures:
        if f["kind"] == "infra":
            continue
        cls = f["class"]
        if cls in ("trade_contract", "trade_monotonicity", "alpha_economics"):
            if "no round-trip" not in f.get("detail", "").lower():
                return False
    return True


def _is_a4t1_candidate(
    staging_path: Path, wf: dict, staging: dict,
) -> bool:
    m = _RUN_ID_RE.search(staging_path.name)
    if not m or m.group(1) != _A4T1_CANDIDATE_RUN_ID:
        return False
    if _wf_digest(wf) != _A4T1_CANDIDATE_WF_DIGEST:
        return False
    meta = staging.get("metadata")
    if isinstance(meta, dict) and meta.get("promotion_basis") is not None:
        return False
    return True


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


def _sub_verdict_failed(value: Any) -> tuple[bool, str]:
    """(failed?, reason) for a stamped {passed, reason} sub-verdict dict.

    The runner composes ``overall_pass`` from ``bool(result["passed"])`` —
    only an explicit True is a pass here; anything else is a failure with
    whatever reason the stamp carries (or its own absence named).
    """
    if not isinstance(value, dict):
        return True, f"sub-verdict absent or not an object (got {value!r})"
    if value.get("passed") is True:
        return False, ""
    reason = value.get("reason")
    return True, (reason if isinstance(reason, str) and reason
                  else f"passed={value.get('passed')!r}")


def classify_gate_failures(wf: dict[str, Any]) -> list[dict[str, Any]]:
    """Name every failing sub-verdict in a stamped ``wf_gate_metadata``.

    The runner does not stamp a failure class; it stamps each sub-verdict
    (``wf_reason``, the sanity fields, ``trade_contract``, ...) and composes
    ``passed`` from them (``runner._compute_overall_pass``). This reads
    those same fields back — the runner's own contract, not parallel gate
    logic — and tags each failure with a §4.3.1 class. Anything that cannot
    be attributed is ``unclassified`` (substance, fail-closed).

    Each entry: {"class": <name>, "kind": "infra"|"substance", "detail": str}.
    """
    failures: list[dict[str, Any]] = []

    def fail(cls: str, detail: str) -> None:
        failures.append({
            "class": cls,
            "kind": "infra" if cls in INFRA_ONLY_FAILURE_CLASSES else "substance",
            "detail": detail[:300],
        })

    # A diagnostic-only stamp (a required gate skipped) is not evidence.
    skipped = wf.get("skipped_required_gates")
    if wf.get("diagnostic_only") is True or (isinstance(skipped, list) and skipped):
        fail("diagnostic_only", f"required gates skipped: {skipped!r}")

    # Validation scope: the candidate must have been evaluated directly or
    # via a validated matching recipe — otherwise it is not the model the
    # recipe claims (recipe-mismatch, §4.3.1 substance).
    if not (wf.get("candidate_artifact_used") is True
            or wf.get("recipe_validated") is True):
        fail("recipe_mismatch",
             f"candidate_artifact_used={wf.get('candidate_artifact_used')!r} "
             f"recipe_validated={wf.get('recipe_validated')!r} "
             f"scope={wf.get('wf_eval_scope')!r}")

    # Walk-forward economics (Fix-4). The runner stamps only its reason
    # string; its own contract prefixes PASS:/FAIL:.
    wf_reason = wf.get("wf_reason")
    if isinstance(wf_reason, str) and wf_reason.startswith("PASS"):
        pass
    elif isinstance(wf_reason, str) and "sim cuts failed" in wf_reason:
        # A phase failure — WF economics never measured; §4.3.3 has no
        # implemented OOS-floor predicate for it, so it is NOT in the
        # bypassable list (fail-closed).
        fail("wf_sim_execution", wf_reason)
    elif isinstance(wf_reason, str) and wf_reason.startswith("FAIL"):
        fail("wf_benchmark_economics", wf_reason)
    else:
        fail("wf_unclassified", f"wf_reason={wf_reason!r}")

    # Sanity battery: shuffled-label leak guard, placebo ceiling (Fix-3),
    # regime sanity IC. Same contract: the reason prefixes PASS:/FAIL:.
    sanity_reason = wf.get("sanity_reason")
    if not (isinstance(sanity_reason, str) and sanity_reason.startswith("PASS")):
        named = False
        shuf = _finite_float(wf.get("sanity_shuffled_ic"))
        if shuf is not None and abs(shuf) >= SHUF_IC_MAX:
            named = True
            fail("leakage_shuffled_label",
                 f"|shuffled_ic|={abs(shuf):.4f} >= {SHUF_IC_MAX}")
        if wf.get("sanity_placebo_absolute_rule_pass") is False:
            named = True
            fail("placebo_ceiling",
                 f"placebo_ic={wf.get('sanity_placebo_ic')!r} >= threshold "
                 f"{wf.get('sanity_placebo_absolute_rule_threshold')!r}; "
                 f"bypassable only via the difference-test floor (check 4)")
        regime = wf.get("sanity_regime_ic")
        if isinstance(regime, dict) and regime.get("passed") is False:
            named = True
            fail("regime_sanity_ic", str(regime.get("reason")))
        if not named:
            fail("sanity_unclassified", f"sanity_reason={sanity_reason!r}")

    for key, cls in (("trade_contract", "trade_contract"),
                     ("trade_monotonicity", "trade_monotonicity"),
                     ("alpha_economics", "alpha_economics")):
        failed, reason = _sub_verdict_failed(wf.get(key))
        if failed:
            fail(cls, reason)

    # config_parity: the runner defaults a MISSING ``passed`` to True
    # (``parity_result.get("passed", True)``); an explicit non-True fails.
    parity = wf.get("config_parity")
    if isinstance(parity, dict) and "passed" in parity and parity.get("passed") is not True:
        fail("config_parity", str(parity.get("reason") or parity.get("passed")))
    elif parity is not None and not isinstance(parity, dict):
        fail("config_parity", f"not an object (got {parity!r})")

    # qp_contract: None is legitimate (no qp lane); a dict must say True.
    qp = wf.get("qp_contract")
    if qp is not None:
        failed, reason = _sub_verdict_failed(qp)
        if failed:
            fail("qp_contract", reason)

    if not failures and wf.get("passed") is False:
        fail("unclassified",
             "stamp says passed=False but no sub-verdict names the failure")
    return failures


def decide(prod_path: Path, staging_path: Path, as_of: dt.date) -> dict[str, Any]:
    """The pure decision. Returns a verdict dict; never raises on bad input."""
    checks: list[dict[str, Any]] = []
    verdict: dict[str, Any] = {
        "policy": PROMOTION_BASIS,
        "as_of": as_of.isoformat(),
        "prod_artifact": str(prod_path),
        "staging_artifact": str(staging_path),
        "quality_floor": FALLBACK_GENUINE_IC_FLOOR,
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

    # Served-model context — REPORTED, never compared (A4: a comparison
    # against a model that itself failed the floor is not a quality
    # predicate).
    prod_meta = prod.get("metadata") if isinstance(prod.get("metadata"), dict) else {}
    served_basis = prod_meta.get("promotion_basis")
    served_genuine = (_finite_float(prod_meta.get("fallback_genuine_ic"))
                      if served_basis == PROMOTION_BASIS else None)
    verdict["served_promotion_basis"] = served_basis
    verdict["served_fallback_genuine_ic"] = served_genuine

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

    # 4. quality floor: genuine_ic present, finite, >= the §5.2 bar (A4).
    genuine = _finite_float(wf.get("sanity_placebo_genuine_ic"))
    if genuine is None:
        return refuse("genuine_ic_present",
                      f"stamped sanity_placebo_genuine_ic is not a finite number "
                      f"(got {wf.get('sanity_placebo_genuine_ic')!r})")

    # Pre-compute failure classes and A4-T1 eligibility BEFORE the floor
    # check so both checks 4 and 5 use ONE coupled predicate.
    failures = classify_gate_failures(wf)
    verdict["failure_classes"] = failures
    a4t1_structural = _is_zero_trade_structural(failures, wf)
    a4t1_candidate = _is_a4t1_candidate(staging_path, wf, staging)
    a4t1 = _a4t1_active(as_of) and (a4t1_structural or a4t1_candidate)

    floor_ctx = {
        "genuine_ic": genuine,
        "floor": FALLBACK_GENUINE_IC_FLOOR,
        "served_fallback_genuine_ic": served_genuine,
        "served_promotion_basis": served_basis,
    }
    if genuine < FALLBACK_GENUINE_IC_FLOOR:
        if a4t1 and genuine >= _A4T1_FLOOR:
            verdict["quality_floor"] = _A4T1_FLOOR
            verdict["quality_floor_standing"] = FALLBACK_GENUINE_IC_FLOOR
            verdict["a4t1_override"] = True
            verdict["a4t1_expiry"] = _A4T1_EXPIRY.isoformat()
            verdict["a4t1_authorization"] = "orch-session-428feb92-2026-08-31"
            floor_ctx["a4t1_override"] = True
            floor_ctx["a4t1_floor"] = _A4T1_FLOOR
            floor_ctx["a4t1_expiry"] = _A4T1_EXPIRY.isoformat()
            ok("quality_floor",
               why=f"a4t1_override(genuine_ic={genuine:+.4f} >= "
                   f"{_A4T1_FLOOR:g}, expires {_A4T1_EXPIRY})",
               **floor_ctx)
        else:
            return refuse("quality_floor",
                          f"quality_floor_not_met(genuine_ic={genuine:+.4f} < "
                          f"{FALLBACK_GENUINE_IC_FLOOR:g})",
                          **floor_ctx)
    else:
        ok("quality_floor", why="quality_floor_met", **floor_ctx)

    # 5. failure classes all infra-only (§4.3.1); any substance class or
    #    anything unclassified is fail-closed regardless of genuine_ic.
    substance = [f["class"] for f in failures if f["kind"] != "infra"]
    if substance:
        if a4t1:
            ok("failure_classes",
               why=f"a4t1_zero_trade_bypass(expires {_A4T1_EXPIRY})",
               failure_classes=failures,
               a4t1_override=True,
               a4t1_expiry=_A4T1_EXPIRY.isoformat())
        else:
            return refuse("failure_classes",
                          f"substance_failure_fail_closed("
                          f"{','.join(substance)})",
                          failure_classes=failures)
    else:
        ok("failure_classes", why="infra_only_ok", failure_classes=failures)

    verdict["decision"] = "FALLBACK_PROMOTE"
    verdict["genuine_ic"] = genuine
    verdict["prod_staleness_days"] = staleness
    if verdict.get("a4t1_override") and a4t1_candidate and not a4t1_structural:
        verdict["a4t1_candidate_run_id"] = _A4T1_CANDIDATE_RUN_ID
        verdict["a4t1_candidate_wf_digest"] = _wf_digest(wf)
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
    meta["fallback_quality_floor"] = verdict["quality_floor"]
    if verdict.get("a4t1_override"):
        meta["fallback_a4t1_override"] = True
        meta["fallback_a4t1_expiry"] = verdict["a4t1_expiry"]
        meta["fallback_a4t1_authorization"] = verdict["a4t1_authorization"]
        meta["fallback_standing_quality_floor"] = verdict["quality_floor_standing"]
        if verdict.get("a4t1_candidate_run_id"):
            meta["fallback_a4t1_candidate_run_id"] = verdict["a4t1_candidate_run_id"]
            meta["fallback_a4t1_candidate_wf_digest"] = verdict["a4t1_candidate_wf_digest"]
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
