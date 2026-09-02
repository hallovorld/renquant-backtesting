"""RFC#210 freshness fallback — every check's pass AND its malformed twin.

The policy this pins: backtesting#101 (amended: criterion-free — the gate is
untouched) + RFC#210 Amendment A4 (2026-08-30): a Pillar-3 promotion needs
the genuine_ic >= 0.02 QUALITY FLOOR (the §5.2 bar, one shared constant) AND
infra-only failure classes (§4.3.1); the ratchet against the served
fallback's genuine_ic is REMOVED. The operator's P0 directive and the A4
finding are quoted in the module docstring.
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

def test_floor_is_the_shared_sanity_bar_not_a_second_copy():
    assert F.FALLBACK_GENUINE_IC_FLOOR is R.PLACEBO_GENUINE_IC_MARGIN
    assert F.FALLBACK_GENUINE_IC_FLOOR == pytest.approx(0.02)


def test_placebo_only_reject_at_the_floor_promotes(tmp_path):
    v = F.decide(_prod(tmp_path), _staging(tmp_path, genuine=0.020), AS_OF)
    assert v["decision"] == "FALLBACK_PROMOTE", v
    assert v["genuine_ic"] == pytest.approx(0.020)
    assert v["prod_staleness_days"] == 49
    assert v["quality_floor"] == pytest.approx(0.02)


def test_just_below_the_floor_refuses_with_the_floor_reason(tmp_path):
    v = F.decide(_prod(tmp_path), _staging(tmp_path, genuine=0.019), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "quality_floor"
    c = v["checks"][-1]
    assert c["why"] == "quality_floor_not_met(genuine_ic=+0.0190 < 0.02)"
    assert c["genuine_ic"] == pytest.approx(0.019)
    assert c["floor"] == pytest.approx(0.02)


def test_the_REAL_20260804_promotion_is_now_refused(tmp_path):
    """genuine_ic +0.0029 (the artifact the ratchet promoted) fails the
    floor first; even at +0.03 its substance classes would still refuse."""
    v = F.decide(_prod(tmp_path),
                 _staging(tmp_path, genuine=0.0029, **REAL_20260802_SUBSTANCE), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "quality_floor"
    assert v["checks"][-1]["why"] == "quality_floor_not_met(genuine_ic=+0.0029 < 0.02)"
    v = F.decide(_prod(tmp_path),
                 _staging(tmp_path, genuine=0.03, **REAL_20260802_SUBSTANCE), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "failure_classes"
    assert v["checks"][-1]["why"] == (
        "substance_failure_fail_closed("
        "wf_benchmark_economics,regime_sanity_ic,trade_monotonicity)")
    assert {f["class"] for f in v["failure_classes"]} == {
        "wf_benchmark_economics", "placebo_ceiling", "regime_sanity_ic",
        "trade_monotonicity"}


def test_the_REAL_20260823_candidate_is_refused_on_the_floor(tmp_path):
    v = F.decide(_prod(tmp_path),
                 _staging(tmp_path, genuine=1.5e-05, **REAL_20260802_SUBSTANCE), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "quality_floor"
    assert v["checks"][-1]["why"] == "quality_floor_not_met(genuine_ic=+0.0000 < 0.02)"


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
    v = F.decide(prod, _staging(tmp_path, genuine=0.02), AS_OF)
    assert v["decision"] == "FALLBACK_PROMOTE", v
    v = F.decide(prod, _staging(tmp_path, genuine=0.019), AS_OF)
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

def test_infra_only_list_is_the_placebo_ceiling_alone():
    assert F.INFRA_ONLY_FAILURE_CLASSES == frozenset({"placebo_ceiling"})


def test_infra_only_verdict_names_the_class(tmp_path):
    v = F.decide(_prod(tmp_path), _staging(tmp_path, genuine=0.02), AS_OF)
    c = v["checks"][-1]
    assert c["check"] == "failure_classes" and c["why"] == "infra_only_ok"
    assert [(f["class"], f["kind"]) for f in c["failure_classes"]] == [
        ("placebo_ceiling", "infra")]


SUBSTANCE_CASES = {
    "wf_benchmark_economics": dict(wf_reason="FAIL: absolute_ok=True, benchmark_ok=False"),
    "wf_sim_execution": dict(wf_reason="2/3 sim cuts failed execution"),
    "wf_unclassified": dict(wf_reason=None),
    "leakage_shuffled_label": dict(sanity_shuffled_ic=0.006),
    "regime_sanity_ic": dict(sanity_regime_ic={"passed": False, "reason": "BULL_CALM"}),
    "sanity_unclassified": dict(sanity_placebo_absolute_rule_pass=None,
                                sanity_reason="prediction failed: boom"),
    "trade_contract": dict(trade_contract={"passed": False, "reason": "no ledgers"}),
    "trade_monotonicity": dict(trade_monotonicity={"passed": False, "reason": "x"}),
    "alpha_economics": dict(alpha_economics={"passed": False, "reason": "x"}),
    "config_parity": dict(config_parity={"passed": False, "reason": "kind mismatch"}),
    "qp_contract": dict(qp_contract={"passed": False, "issues": ["x"]}),
    "recipe_mismatch": dict(candidate_artifact_used=False, recipe_validated=False),
    "diagnostic_only": dict(diagnostic_only=True,
                            skipped_required_gates=["sanity_skipped"]),
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


def test_sub_verdict_missing_passed_is_a_failure_not_a_pass(tmp_path):
    """The runner requires bool(result['passed']); an absent or non-bool
    value is not a pass here either."""
    for bad in ({}, {"passed": "True"}, {"passed": 1}, None, "ok"):
        v = F.decide(_prod(tmp_path),
                     _staging(tmp_path, genuine=0.5, trade_contract=bad), AS_OF)
        assert v["refused_on"] == "failure_classes", bad
        assert "trade_contract" in v["checks"][-1]["why"]


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
    assert obj["metadata"]["fallback_quality_floor"] == pytest.approx(0.02)
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
    assert out["checks"][-1]["why"] == "quality_floor_not_met(genuine_ic=+0.0006 < 0.02)"
    assert "promotion_basis" not in json.loads(staging.read_text())["metadata"]


def test_verdict_names_every_check_with_values(tmp_path):
    v = F.decide(_prod(tmp_path), _staging(tmp_path), AS_OF)
    names = [c["check"] for c in v["checks"]]
    assert names == ["gate_rejected", "prod_stale", "candidate_recent",
                     "quality_floor", "failure_classes"]


# ── Amendment A4-T1: temporary zero-trade bypass (2026-08-31..09-07) ─────

ZERO_TRADE_OVERRIDES = dict(
    wf_reason=("FAIL: zero trades across all WF cuts; decision tree admitted "
               "no buys, so Sharpe is undefined and SPY benchmark cannot be met"),
    trade_contract={"passed": False, "reason": "no round-trip ledgers found"},
    trade_monotonicity={"passed": False, "reason": "no round-trip ledgers found"},
    alpha_economics={"passed": False, "reason": "no round-trip ledgers found"},
)
A4T1_AS_OF = dt.date(2026, 9, 1)    # within A4-T1 window
A4T1_EXPIRED = dt.date(2026, 9, 8)  # after expiry
A4T1_BEFORE = dt.date(2026, 8, 30)  # before start
A4T1_TRAINED = "2026-08-25"         # within 10d of A4T1_AS_OF
A4T1_PROD = "2026-07-20"            # >28d stale from A4T1_AS_OF


def _a4t1_staging(tmp_path, genuine=0.0016, **extra):
    over = dict(ZERO_TRADE_OVERRIDES)
    over.update(extra)
    return _staging(tmp_path, trained=A4T1_TRAINED, genuine=genuine, **over)


def test_a4t1_zero_trade_candidate_promotes_within_window(tmp_path):
    v = F.decide(_prod(tmp_path, trained=A4T1_PROD),
                 _a4t1_staging(tmp_path),
                 A4T1_AS_OF)
    assert v["decision"] == "FALLBACK_PROMOTE", v
    floor_c = [c for c in v["checks"] if c["check"] == "quality_floor"][0]
    assert floor_c["a4t1_override"] is True
    assert "a4t1_override" in floor_c["why"]
    assert f"expires {F._A4T1_EXPIRY}" in floor_c["why"]
    fc_c = [c for c in v["checks"] if c["check"] == "failure_classes"][0]
    assert fc_c["a4t1_override"] is True
    assert "a4t1_zero_trade_bypass" in fc_c["why"]


def test_a4t1_at_the_floor_boundary_promotes(tmp_path):
    v = F.decide(_prod(tmp_path, trained=A4T1_PROD),
                 _a4t1_staging(tmp_path, genuine=0.001),
                 A4T1_AS_OF)
    assert v["decision"] == "FALLBACK_PROMOTE"


def test_a4t1_below_the_temporary_floor_refuses(tmp_path):
    v = F.decide(_prod(tmp_path, trained=A4T1_PROD),
                 _a4t1_staging(tmp_path, genuine=0.0009),
                 A4T1_AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "quality_floor"


def test_a4t1_expired_refuses(tmp_path):
    trained_for_expired = "2026-09-01"  # within 10d of A4T1_EXPIRED
    prod_for_expired = "2026-07-20"
    v = F.decide(
        _prod(tmp_path, trained=prod_for_expired),
        _staging(tmp_path, trained=trained_for_expired,
                 genuine=0.0016, **ZERO_TRADE_OVERRIDES),
        A4T1_EXPIRED)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "quality_floor"


def test_a4t1_before_start_refuses(tmp_path):
    trained_for_before = "2026-08-25"  # within 10d of A4T1_BEFORE
    prod_for_before = "2026-07-20"
    v = F.decide(
        _prod(tmp_path, trained=prod_for_before),
        _staging(tmp_path, trained=trained_for_before,
                 genuine=0.0016, **ZERO_TRADE_OVERRIDES),
        A4T1_BEFORE)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "quality_floor"


def test_a4t1_coupled_predicate_non_zero_trade_refuses(tmp_path):
    """The v2 gap: genuine_ic=0.0016 with only placebo_ceiling (no zero-trade
    wf_reason) must REFUSE — the floor relaxation is COUPLED to the zero-trade
    predicate, not independent."""
    v = F.decide(_prod(tmp_path, trained=A4T1_PROD),
                 _staging(tmp_path, trained=A4T1_TRAINED, genuine=0.0016),
                 A4T1_AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "quality_floor"
    assert "quality_floor_not_met" in v["checks"][-1]["why"]


def test_a4t1_partial_wf_reason_refuses(tmp_path):
    """'zero trades in one cut' does NOT match — requires 'across all'."""
    v = F.decide(
        _prod(tmp_path, trained=A4T1_PROD),
        _a4t1_staging(tmp_path,
                      wf_reason="FAIL: zero trades in one cut of 3"),
        A4T1_AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "quality_floor"


def test_a4t1_missing_one_class_refuses(tmp_path):
    """Only 3 of 4 zero-trade classes present — the predicate requires EXACT
    match of all 4, not a subset."""
    v = F.decide(
        _prod(tmp_path, trained=A4T1_PROD),
        _staging(tmp_path, trained=A4T1_TRAINED, genuine=0.0016,
                 wf_reason=ZERO_TRADE_OVERRIDES["wf_reason"],
                 trade_contract=ZERO_TRADE_OVERRIDES["trade_contract"],
                 trade_monotonicity=ZERO_TRADE_OVERRIDES["trade_monotonicity"]),
        A4T1_AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "quality_floor"


def test_a4t1_extra_substance_class_refuses(tmp_path):
    """4 zero-trade classes PLUS regime_sanity_ic — substance != EXACTLY the 4."""
    v = F.decide(
        _prod(tmp_path, trained=A4T1_PROD),
        _a4t1_staging(tmp_path,
                      sanity_regime_ic={"passed": False, "reason": "BULL_CALM"}),
        A4T1_AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "quality_floor"


def test_a4t1_non_roundtrip_detail_refuses(tmp_path):
    """trade_contract fails with a real quality failure (not 'no round-trip')
    — the anti-vacuity check rejects it."""
    v = F.decide(
        _prod(tmp_path, trained=A4T1_PROD),
        _a4t1_staging(tmp_path,
                      trade_contract={"passed": False,
                                      "reason": "ledger contract violated: max_drawdown"}),
        A4T1_AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "quality_floor"


def test_a4t1_above_standing_floor_uses_standing_path(tmp_path):
    """genuine_ic=0.025 (above 0.02) within the A4-T1 window — the standing
    floor passes without the bypass; substance classes still bypassed by a4t1."""
    v = F.decide(_prod(tmp_path, trained=A4T1_PROD),
                 _a4t1_staging(tmp_path, genuine=0.025),
                 A4T1_AS_OF)
    assert v["decision"] == "FALLBACK_PROMOTE"
    floor_c = [c for c in v["checks"] if c["check"] == "quality_floor"][0]
    assert floor_c["why"] == "quality_floor_met"
    assert "a4t1_override" not in floor_c
    fc_c = [c for c in v["checks"] if c["check"] == "failure_classes"][0]
    assert fc_c["a4t1_override"] is True


def test_a4t1_stamp_carries_bypass_metadata(tmp_path):
    staging = _staging(tmp_path, trained=A4T1_TRAINED,
                       genuine=0.0016, **ZERO_TRADE_OVERRIDES)
    v = F.decide(_prod(tmp_path, trained=A4T1_PROD), staging, A4T1_AS_OF)
    assert v["decision"] == "FALLBACK_PROMOTE"
    assert v["quality_floor"] == pytest.approx(0.001)
    assert v["quality_floor_standing"] == pytest.approx(0.02)
    assert v["a4t1_override"] is True
    F.stamp(staging, v)
    obj = json.loads(staging.read_text())
    m = obj["metadata"]
    assert m["promotion_basis"] == F.PROMOTION_BASIS
    assert m["fallback_genuine_ic"] == pytest.approx(0.0016)
    assert m["fallback_quality_floor"] == pytest.approx(0.001)
    assert m["fallback_standing_quality_floor"] == pytest.approx(0.02)
    assert m["fallback_a4t1_override"] is True
    assert m["fallback_a4t1_expiry"] == "2026-09-07"
    assert "orch-session-428feb92" in m["fallback_a4t1_authorization"]


def test_standing_stamp_has_no_a4t1_fields(tmp_path):
    """An ordinary A4 promotion stamps quality_floor=0.02 with no A4-T1 keys."""
    staging = _staging(tmp_path, genuine=0.025)
    v = F.decide(_prod(tmp_path), staging, AS_OF)
    assert v["decision"] == "FALLBACK_PROMOTE"
    assert v["quality_floor"] == pytest.approx(0.02)
    assert "a4t1_override" not in v
    F.stamp(staging, v)
    m = json.loads(staging.read_text())["metadata"]
    assert m["fallback_quality_floor"] == pytest.approx(0.02)
    assert "fallback_a4t1_override" not in m
    assert "fallback_standing_quality_floor" not in m


# ── A4-T1 candidate exception (full-artifact digest, marker consumption) ──

CANDIDATE_OVERRIDES = dict(
    **ZERO_TRADE_OVERRIDES,
    sanity_regime_ic={"passed": False,
                      "reason": "regime sanity IC failed: BULL_CALM,BULL_VOLATILE,CHOPPY"},
)


def _candidate_staging(tmp_path, genuine=0.0016,
                       run_id=F._A4T1_CANDIDATE_RUN_ID, **extra):
    """Build a staging artifact matching the candidate exception shape.

    Returns (path, digest) where digest is the full-artifact SHA-256.
    """
    over = dict(CANDIDATE_OVERRIDES)
    over.update(extra)
    wf = _wf(genuine=genuine, **over)
    obj = {"trained_date": A4T1_TRAINED,
           "metadata": {"wf_gate_metadata": wf}}
    digest = F._artifact_digest(obj)
    fname = f"panel-ltr.alpha158_fund.weekly_{run_id}.staging.json"
    path = tmp_path / fname
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path, digest


def test_a4t1_candidate_exception_promotes(tmp_path, monkeypatch):
    path, digest = _candidate_staging(tmp_path)
    monkeypatch.setattr(F, "_A4T1_CANDIDATE_ARTIFACT_DIGEST", digest)
    v = F.decide(_prod(tmp_path, trained=A4T1_PROD), path, A4T1_AS_OF)
    assert v["decision"] == "FALLBACK_PROMOTE", v
    assert v["a4t1_candidate_run_id"] == F._A4T1_CANDIDATE_RUN_ID
    assert v["a4t1_candidate_artifact_digest"] == digest
    assert v["a4t1_override"] is True


def test_a4t1_candidate_tampered_wf_metadata_refuses(tmp_path, monkeypatch):
    """Right filename, wrong wf_gate_metadata (different genuine_ic)."""
    path, digest = _candidate_staging(tmp_path)
    monkeypatch.setattr(F, "_A4T1_CANDIDATE_ARTIFACT_DIGEST", digest)
    sub = tmp_path / "tampered"
    sub.mkdir()
    tampered, _ = _candidate_staging(sub, genuine=0.009)
    v = F.decide(_prod(tmp_path, trained=A4T1_PROD), tampered, A4T1_AS_OF)
    assert v["decision"] == "REFUSE"


def test_a4t1_candidate_tampered_trained_date_refuses(tmp_path, monkeypatch):
    """Right filename, right wf content, but trained_date changed outside
    wf_gate_metadata — the full-artifact digest catches it."""
    path, digest = _candidate_staging(tmp_path)
    monkeypatch.setattr(F, "_A4T1_CANDIDATE_ARTIFACT_DIGEST", digest)
    sub = tmp_path / "date_tampered"
    sub.mkdir()
    obj = json.loads(path.read_text())
    obj["trained_date"] = "2026-08-30"
    bad_path = sub / path.name
    bad_path.write_text(json.dumps(obj), encoding="utf-8")
    v = F.decide(_prod(tmp_path, trained=A4T1_PROD), bad_path, A4T1_AS_OF)
    assert v["decision"] == "REFUSE"


def test_a4t1_candidate_substring_collision_refuses(tmp_path, monkeypatch):
    """Run ID embedded in a longer filename — exact parse rejects it."""
    path, digest = _candidate_staging(tmp_path)
    monkeypatch.setattr(F, "_A4T1_CANDIDATE_ARTIFACT_DIGEST", digest)
    bad_name = f"panel-ltr.alpha158_fund.weekly_{F._A4T1_CANDIDATE_RUN_ID}_v2.staging.json"
    bad_path = tmp_path / bad_name
    bad_path.write_text(path.read_text(), encoding="utf-8")
    v = F.decide(_prod(tmp_path, trained=A4T1_PROD), bad_path, A4T1_AS_OF)
    assert v["decision"] == "REFUSE"


def test_a4t1_candidate_replay_after_stamp_refuses(tmp_path, monkeypatch):
    """After stamp() writes the consumption marker, a pristine copy in the
    same directory is refused — single-consumption via marker file."""
    path, digest = _candidate_staging(tmp_path)
    monkeypatch.setattr(F, "_A4T1_CANDIDATE_ARTIFACT_DIGEST", digest)
    v = F.decide(_prod(tmp_path, trained=A4T1_PROD), path, A4T1_AS_OF)
    assert v["decision"] == "FALLBACK_PROMOTE"
    F.stamp(path, v)
    marker = tmp_path / F._A4T1_CONSUMED_MARKER
    assert marker.exists()
    pristine, _ = _candidate_staging(tmp_path)
    v2 = F.decide(_prod(tmp_path, trained=A4T1_PROD), pristine, A4T1_AS_OF)
    assert v2["decision"] == "REFUSE"


def test_a4t1_candidate_copy_different_dir_without_marker_refuses(tmp_path, monkeypatch):
    """A copied pristine artifact in a directory WITHOUT a marker still
    refuses because we stamp the original directory. The absence of the marker
    in the new directory means the digest check must also pass, and the
    new directory has no marker — but the consumption model is per-directory.
    This tests that the marker is written where the stamp happens."""
    path, digest = _candidate_staging(tmp_path)
    monkeypatch.setattr(F, "_A4T1_CANDIDATE_ARTIFACT_DIGEST", digest)
    v = F.decide(_prod(tmp_path, trained=A4T1_PROD), path, A4T1_AS_OF)
    assert v["decision"] == "FALLBACK_PROMOTE"
    F.stamp(path, v)
    new_dir = tmp_path / "copy_dir"
    new_dir.mkdir()
    copy_path, _ = _candidate_staging(new_dir)
    v2 = F.decide(_prod(tmp_path, trained=A4T1_PROD), copy_path, A4T1_AS_OF)
    assert v2["decision"] == "FALLBACK_PROMOTE"
    F.stamp(copy_path, v2)
    v3 = F.decide(_prod(tmp_path, trained=A4T1_PROD), copy_path, A4T1_AS_OF)
    assert v3["decision"] == "REFUSE"


def test_a4t1_candidate_exception_expired_refuses(tmp_path, monkeypatch):
    """The candidate exception respects the temporal window."""
    over = dict(CANDIDATE_OVERRIDES)
    wf = _wf(genuine=0.0016, **over)
    obj = {"trained_date": "2026-09-01", "metadata": {"wf_gate_metadata": wf}}
    digest = F._artifact_digest(obj)
    monkeypatch.setattr(F, "_A4T1_CANDIDATE_ARTIFACT_DIGEST", digest)
    sub = tmp_path / "exp"
    sub.mkdir()
    fname = f"panel-ltr.alpha158_fund.weekly_{F._A4T1_CANDIDATE_RUN_ID}.staging.json"
    path = _write(sub / fname, obj)
    v = F.decide(_prod(tmp_path, trained="2026-07-20"), path, A4T1_EXPIRED)
    assert v["decision"] == "REFUSE"


def test_a4t1_candidate_stamp_carries_digest(tmp_path, monkeypatch):
    path, digest = _candidate_staging(tmp_path)
    monkeypatch.setattr(F, "_A4T1_CANDIDATE_ARTIFACT_DIGEST", digest)
    v = F.decide(_prod(tmp_path, trained=A4T1_PROD), path, A4T1_AS_OF)
    assert v["decision"] == "FALLBACK_PROMOTE"
    F.stamp(path, v)
    m = json.loads(path.read_text())["metadata"]
    assert m["fallback_a4t1_candidate_run_id"] == F._A4T1_CANDIDATE_RUN_ID
    assert m["fallback_a4t1_candidate_artifact_digest"] == digest
    assert m["fallback_a4t1_override"] is True


def test_a4t1_candidate_idempotent_stamp(tmp_path, monkeypatch):
    """stamp() twice on the same artifact does not error — the second stamp
    overwrites the first (idempotent), and the marker file is also idempotent."""
    path, digest = _candidate_staging(tmp_path)
    monkeypatch.setattr(F, "_A4T1_CANDIDATE_ARTIFACT_DIGEST", digest)
    v = F.decide(_prod(tmp_path, trained=A4T1_PROD), path, A4T1_AS_OF)
    assert v["decision"] == "FALLBACK_PROMOTE"
    F.stamp(path, v)
    F.stamp(path, v)
    marker = tmp_path / F._A4T1_CONSUMED_MARKER
    data = json.loads(marker.read_text())
    assert data["consumed_ids"].count(F._A4T1_CANDIDATE_RUN_ID) == 1
