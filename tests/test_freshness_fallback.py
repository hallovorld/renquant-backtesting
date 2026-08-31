"""RFC#210 freshness fallback — every check's pass AND its malformed twin.

The policy this pins: backtesting#101 (amended: criterion-free — the gate is
untouched) + RFC#210 Amendment A4 (2026-08-30) + A4-T1 TEMPORARY OVERRIDE
(2026-08-31): floor lowered to 0.001 and infra-only set expanded to cover
zero-trade WF outcomes. See freshness_fallback.py module docstring for
the restore condition.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from renquant_backtesting.wf_gate import freshness_fallback as F
from renquant_backtesting.wf_gate import runner as R

AS_OF = dt.date(2026, 8, 9)


def _write(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def _prod(tmp_path, trained="2026-06-21", basis=None, prior_g=None):
    meta = {}
    if basis is not None:
        meta["promotion_basis"] = basis
    if prior_g is not None:
        meta["fallback_genuine_ic"] = prior_g
    return _write(tmp_path / "prod.json",
                  {"trained_date": trained, "metadata": meta})


def _wf(genuine=0.02, verdict=False, **over):
    """A stamp shaped like the runner's: a PLACEBO-ONLY reject by default —
    every other sub-verdict passes, the enforced placebo ceiling fails, and
    ``genuine_ic`` is the caller's. Overrides let a test flip one class."""
    wf = {
        "passed": verdict,
        "diagnostic_only": False,
        "skipped_required_gates": [],
        "candidate_artifact_used": False,
        "recipe_validated": True,
        "wf_eval_scope": "walkforward_manifest",
        "wf_reason": "PASS: absolute Sharpe floor met and SPY benchmark met",
        "sanity_reason": "FAIL: shuf_ic=+0.0010 (need |·| < 0.005), genuine_ic=...",
        "sanity_shuffled_ic": 0.001,
        "sanity_placebo_ic": 0.043,
        "sanity_placebo_aligned_real_ic": 0.046,
        "sanity_placebo_absolute_rule_pass": False,
        "sanity_placebo_absolute_rule_threshold": 0.023,
        "sanity_placebo_genuine_ic": genuine,
        "sanity_regime_ic": {"passed": True, "reason": "ok"},
        "trade_contract": {"passed": True, "reason": "trade ledger contract OK"},
        "trade_monotonicity": {"passed": True, "reason": "ok"},
        "alpha_economics": {"passed": True, "reason": "ok"},
        "config_parity": {"passed": True},
        "qp_contract": None,
    }
    wf.update(over)
    return wf


def _staging(tmp_path, trained="2026-08-02", genuine=0.02, verdict=False,
             stamp=True, **over):
    wf = _wf(genuine=genuine, verdict=verdict, **over) if stamp else {}
    return _write(tmp_path / "staging.json",
                  {"trained_date": trained,
                   "metadata": {"wf_gate_metadata": wf} if stamp else {}})


# The measured 2026-08-02 candidate (the one promoted 2026-08-04 under the
# old ratchet): sub-SPY (ΔSharpe −0.479), regime sanity IC failed, trade
# monotonicity failed, placebo ceiling failed, genuine_ic +0.0029.
REAL_20260802_SUBSTANCE = dict(
    wf_reason=("FAIL: absolute_ok=True, benchmark_ok=False, regime_ok=False; "
               "mean Sharpe +0.602, 3/3 cuts > 0; SPY mean Sharpe +1.081, "
               "ΔSharpe -0.479, beat SPY Sharpe 1/3, beat SPY APY 0/3"),
    sanity_regime_ic={"passed": False,
                      "reason": "regime sanity IC failed: BULL_CALM,BULL_VOLATILE,CHOPPY"},
    trade_monotonicity={"passed": False,
                        "reason": "score monotonicity failed for entry_rank_score"},
)


# ── the floor ─────────────────────────────────────────────────────────────

def test_floor_is_the_a4t1_temporary_override():
    assert F.FALLBACK_GENUINE_IC_FLOOR == pytest.approx(0.001)


def test_placebo_only_reject_at_the_floor_promotes(tmp_path):
    v = F.decide(_prod(tmp_path), _staging(tmp_path, genuine=0.001), AS_OF)
    assert v["decision"] == "FALLBACK_PROMOTE", v
    assert v["genuine_ic"] == pytest.approx(0.001)
    assert v["prod_staleness_days"] == 49
    assert v["quality_floor"] == pytest.approx(0.001)


def test_just_below_the_floor_refuses_with_the_floor_reason(tmp_path):
    v = F.decide(_prod(tmp_path), _staging(tmp_path, genuine=0.0009), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "quality_floor"
    c = v["checks"][-1]
    assert c["why"] == "quality_floor_not_met(genuine_ic=+0.0009 < 0.001)"
    assert c["genuine_ic"] == pytest.approx(0.0009)
    assert c["floor"] == pytest.approx(0.001)


def test_the_REAL_20260804_promotion_is_now_refused(tmp_path):
    """genuine_ic +0.0029 (the artifact the ratchet promoted) passes the
    A4-T1 floor (0.001) but its substance class (regime_sanity_ic) still
    refuses.  Even at +0.03 the substance class blocks."""
    v = F.decide(_prod(tmp_path),
                 _staging(tmp_path, genuine=0.0029, **REAL_20260802_SUBSTANCE), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "failure_classes"
    assert v["checks"][-1]["why"] == (
        "substance_failure_fail_closed(regime_sanity_ic)")
    v = F.decide(_prod(tmp_path),
                 _staging(tmp_path, genuine=0.03, **REAL_20260802_SUBSTANCE), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "failure_classes"
    assert v["checks"][-1]["why"] == (
        "substance_failure_fail_closed(regime_sanity_ic)")
    assert {f["class"] for f in v["failure_classes"]} == {
        "wf_benchmark_economics", "placebo_ceiling", "regime_sanity_ic",
        "trade_monotonicity"}


def test_the_REAL_20260823_candidate_is_refused_on_the_floor(tmp_path):
    v = F.decide(_prod(tmp_path),
                 _staging(tmp_path, genuine=1.5e-05, **REAL_20260802_SUBSTANCE), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "quality_floor"
    assert v["checks"][-1]["why"] == "quality_floor_not_met(genuine_ic=+0.0000 < 0.001)"


def test_negative_genuine_ic_is_never_served(tmp_path):
    v = F.decide(_prod(tmp_path), _staging(tmp_path, genuine=-0.0001), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "quality_floor"


def test_missing_genuine_ic_refuses(tmp_path):
    v = F.decide(_prod(tmp_path), _staging(tmp_path, genuine=None), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "genuine_ic_present"


@pytest.mark.parametrize("bad", ["0.03", True, float("nan"), float("inf")])
def test_non_number_genuine_ic_refuses_not_coerces(tmp_path, bad):
    """The orch#770/pipeline#259 class: a string or bool must never pass a
    numeric gate by coercion."""
    v = F.decide(_prod(tmp_path), _staging(tmp_path, genuine=bad), AS_OF)
    assert v["decision"] == "REFUSE"
    assert v["refused_on"] in ("genuine_ic_present", "quality_floor")


# ── the served model is context, never a comparand ────────────────────────

@pytest.mark.parametrize("basis,prior", [
    (F.PROMOTION_BASIS, 0.5),      # served fallback HIGHER than the candidate
    (F.PROMOTION_BASIS, 0.0001),   # served fallback LOWER
    (F.PROMOTION_BASIS, 0.02),     # served fallback EQUAL
    (F.PROMOTION_BASIS, None),     # served fallback with no stamped value
    (None, None),                  # served model gate-passed
])
def test_served_fallback_value_is_reported_never_compared(tmp_path, basis, prior):
    prod = _prod(tmp_path, basis=basis, prior_g=prior)
    v = F.decide(prod, _staging(tmp_path, genuine=0.001), AS_OF)
    assert v["decision"] == "FALLBACK_PROMOTE", v
    v = F.decide(prod, _staging(tmp_path, genuine=0.0009), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "quality_floor"
    expect = prior if basis == F.PROMOTION_BASIS else None
    assert v["served_fallback_genuine_ic"] == expect
    assert v["served_promotion_basis"] == basis
    assert v["checks"][-1]["served_fallback_genuine_ic"] == expect


def test_the_old_ratchet_no_longer_exists_in_the_module():
    src = Path(F.__file__).read_text(encoding="utf-8")
    for token in ("ratchet_up_only", "ratchet_prior_readable",
                  "prior_genuine_ic", "may only walk up"):
        assert token not in src, token
    assert not any(c["check"].startswith("ratchet")
                   for c in F.decide(Path("/nonexistent"), Path("/nonexistent"),
                                     AS_OF)["checks"])


# ── failure classes (§4.3.1) ──────────────────────────────────────────────

def test_infra_only_list_is_the_a4t1_expanded_set():
    assert F.INFRA_ONLY_FAILURE_CLASSES == frozenset({
        "placebo_ceiling", "wf_benchmark_economics",
        "trade_contract", "trade_monotonicity", "alpha_economics"})


def test_infra_only_verdict_names_the_class(tmp_path):
    v = F.decide(_prod(tmp_path), _staging(tmp_path, genuine=0.02), AS_OF)
    c = v["checks"][-1]
    assert c["check"] == "failure_classes" and c["why"] == "infra_only_ok"
    assert [(f["class"], f["kind"]) for f in c["failure_classes"]] == [
        ("placebo_ceiling", "infra")]


SUBSTANCE_CASES = {
    "wf_sim_execution": dict(wf_reason="2/3 sim cuts failed execution"),
    "wf_unclassified": dict(wf_reason=None),
    "leakage_shuffled_label": dict(sanity_shuffled_ic=0.006),
    "regime_sanity_ic": dict(sanity_regime_ic={"passed": False, "reason": "BULL_CALM"}),
    "sanity_unclassified": dict(sanity_placebo_absolute_rule_pass=None,
                                sanity_reason="prediction failed: boom"),
    "config_parity": dict(config_parity={"passed": False, "reason": "kind mismatch"}),
    "qp_contract": dict(qp_contract={"passed": False, "issues": ["x"]}),
    "recipe_mismatch": dict(candidate_artifact_used=False, recipe_validated=False),
    "diagnostic_only": dict(diagnostic_only=True,
                            skipped_required_gates=["sanity_skipped"]),
}

A4T1_INFRA_CASES = {
    "wf_benchmark_economics": dict(wf_reason="FAIL: absolute_ok=True, benchmark_ok=False"),
    "trade_contract": dict(trade_contract={"passed": False, "reason": "no ledgers"}),
    "trade_monotonicity": dict(trade_monotonicity={"passed": False, "reason": "x"}),
    "alpha_economics": dict(alpha_economics={"passed": False, "reason": "x"}),
}


@pytest.mark.parametrize("cls,over", sorted(SUBSTANCE_CASES.items()))
def test_substance_class_refuses_regardless_of_genuine_ic(tmp_path, cls, over):
    """A candidate far above the floor (0.5) with ONE substance class is
    refused, and the verdict names that class."""
    v = F.decide(_prod(tmp_path), _staging(tmp_path, genuine=0.5, **over), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "failure_classes", v
    assert v["checks"][-1]["why"] == f"substance_failure_fail_closed({cls})"
    kinds = {f["class"]: f["kind"] for f in v["failure_classes"]}
    assert kinds[cls] == "substance"
    assert kinds.get("placebo_ceiling", "infra") == "infra"


@pytest.mark.parametrize("cls,over", sorted(A4T1_INFRA_CASES.items()))
def test_a4t1_infra_class_promotes_above_the_floor(tmp_path, cls, over):
    """A4-T1 reclassified these as infra — they no longer block fallback."""
    v = F.decide(_prod(tmp_path), _staging(tmp_path, genuine=0.5, **over), AS_OF)
    assert v["decision"] == "FALLBACK_PROMOTE", v
    kinds = {f["class"]: f["kind"] for f in v["failure_classes"]}
    assert kinds[cls] == "infra"


def test_sub_verdict_missing_passed_is_a_failure_not_a_pass(tmp_path):
    """The runner requires bool(result['passed']); an absent or non-bool
    value is not a pass here either.  Uses qp_contract (substance) so the
    failure actually blocks the fallback."""
    for bad in ({"passed": "True"}, {"passed": 1}, {"passed": False, "issues": ["x"]}):
        v = F.decide(_prod(tmp_path),
                     _staging(tmp_path, genuine=0.5, qp_contract=bad), AS_OF)
        assert v["refused_on"] == "failure_classes", bad
        assert "qp_contract" in v["checks"][-1]["why"]


def test_rejected_stamp_with_no_failing_sub_verdict_is_unclassified(tmp_path):
    """passed=False while every sub-verdict reads as a pass: the classifier
    cannot attribute the reject — fail-closed, never 'infra_only_ok'."""
    v = F.decide(_prod(tmp_path),
                 _staging(tmp_path, genuine=0.5,
                          sanity_placebo_absolute_rule_pass=True,
                          sanity_reason="PASS: shuf_ic=... placebo_ic=..."),
                 AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "failure_classes"
    assert v["checks"][-1]["why"] == "substance_failure_fail_closed(unclassified)"


def test_classifier_on_a_clean_pass_stamp_names_nothing():
    wf = _wf(genuine=0.05, verdict=True,
             sanity_placebo_absolute_rule_pass=True,
             sanity_reason="PASS: ...")
    assert F.classify_gate_failures(wf) == []


# ── the untouched checks ──────────────────────────────────────────────────

def test_fresh_prod_refuses(tmp_path):
    v = F.decide(_prod(tmp_path, trained="2026-07-20"), _staging(tmp_path), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "prod_stale"


def test_staleness_boundary_is_strictly_greater_than_28(tmp_path):
    exactly_28 = (AS_OF - dt.timedelta(days=28)).isoformat()
    v = F.decide(_prod(tmp_path, trained=exactly_28), _staging(tmp_path), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "prod_stale"
    day_29 = (AS_OF - dt.timedelta(days=29)).isoformat()
    v = F.decide(_prod(tmp_path, trained=day_29), _staging(tmp_path), AS_OF)
    assert v["decision"] == "FALLBACK_PROMOTE"


def test_old_candidate_refuses(tmp_path):
    v = F.decide(_prod(tmp_path), _staging(tmp_path, trained="2026-07-25"), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "candidate_recent"


def test_future_dated_candidate_refuses(tmp_path):
    v = F.decide(_prod(tmp_path), _staging(tmp_path, trained="2026-08-10"), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "candidate_recent"


def test_gate_passed_candidate_uses_the_normal_path(tmp_path):
    v = F.decide(_prod(tmp_path), _staging(tmp_path, verdict=True), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "gate_rejected"


def test_unstamped_staging_refuses(tmp_path):
    v = F.decide(_prod(tmp_path), _staging(tmp_path, stamp=False), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "staging_gate_stamp"


def test_unreadable_inputs_refuse(tmp_path):
    v = F.decide(tmp_path / "nope.json", _staging(tmp_path), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "prod_readable"
    v = F.decide(_prod(tmp_path), tmp_path / "nope.json", AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "staging_readable"


@pytest.mark.parametrize("bad", [None, "False", 0, 1, "rejected"])
def test_non_boolean_stamped_verdict_refuses_not_permits(tmp_path, bad):
    """[codex on #102] an absent/corrupted/never-recorded gate decision must
    never become permission for a production promotion — only an explicit
    boolean False proceeds. (None here means the producer stamped the key as
    null; a MISSING key is the unstamped case covered above.)"""
    v = F.decide(_prod(tmp_path), _staging(tmp_path, verdict=bad), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "gate_rejected"


def test_missing_passed_key_refuses(tmp_path):
    staging = _write(tmp_path / "staging.json",
                     {"trained_date": "2026-08-02",
                      "metadata": {"wf_gate_metadata": {
                          "sanity_placebo_genuine_ic": 0.02}}})
    v = F.decide(_prod(tmp_path), staging, AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "gate_rejected"


# ── stamp + CLI (the umbrella Step 4b contract) ───────────────────────────

def test_stamp_writes_atomically_and_only_on_promote(tmp_path):
    staging = _staging(tmp_path)
    v = F.decide(_prod(tmp_path), staging, AS_OF)
    F.stamp(staging, v)
    obj = json.loads(staging.read_text())
    assert obj["metadata"]["promotion_basis"] == F.PROMOTION_BASIS
    assert obj["metadata"]["fallback_genuine_ic"] == pytest.approx(0.02)
    assert obj["metadata"]["fallback_as_of"] == "2026-08-09"
    assert obj["metadata"]["fallback_quality_floor"] == pytest.approx(0.001)
    # a REFUSE verdict must never stamp
    refuse = F.decide(_prod(tmp_path, trained="2026-08-01"), staging, AS_OF)
    with pytest.raises(ValueError):
        F.stamp(staging, refuse)


def test_cli_exit_codes(tmp_path, capsys):
    prod, staging = _prod(tmp_path), _staging(tmp_path)
    rc = F.main(["--prod", str(prod), "--staging", str(staging),
                 "--as-of", "2026-08-09"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["decision"] == "FALLBACK_PROMOTE"
    rc = F.main(["--prod", str(prod), "--staging", str(staging),
                 "--as-of", "2026-07-01"])
    assert rc == 1
    out = json.loads(capsys.readouterr().out)
    assert out["decision"] == "REFUSE" and out["refused_on"] == "prod_stale"


def test_cli_refuse_on_the_floor_exits_1_and_prints_the_reason(tmp_path, capsys):
    prod, staging = _prod(tmp_path), _staging(tmp_path, genuine=0.0006)
    rc = F.main(["--prod", str(prod), "--staging", str(staging),
                 "--as-of", "2026-08-09", "--stamp"])
    assert rc == 1
    out = json.loads(capsys.readouterr().out)
    assert out["refused_on"] == "quality_floor"
    assert out["checks"][-1]["why"] == "quality_floor_not_met(genuine_ic=+0.0006 < 0.001)"
    assert "promotion_basis" not in json.loads(staging.read_text())["metadata"]


def test_verdict_names_every_check_with_values(tmp_path):
    v = F.decide(_prod(tmp_path), _staging(tmp_path), AS_OF)
    names = [c["check"] for c in v["checks"]]
    assert names == ["gate_rejected", "prod_stale", "candidate_recent",
                     "quality_floor", "failure_classes"]
