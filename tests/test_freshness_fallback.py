"""RFC#210 freshness fallback — every check's pass AND its malformed twin.

The policy this pins: backtesting#101 (amended: criterion-free — the gate is
untouched; genuine_ic is ordinal/sign only). The operator's P0 directive is
quoted in the module docstring.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from renquant_backtesting.wf_gate import freshness_fallback as F

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


def _staging(tmp_path, trained="2026-08-02", genuine=0.0029, verdict=False,
             stamp=True):
    wf = {}
    if stamp:
        wf = {"passed": verdict,
              "sanity_placebo_genuine_ic": genuine}
    return _write(tmp_path / "staging.json",
                  {"trained_date": trained,
                   "metadata": {"wf_gate_metadata": wf} if stamp else {}})


def test_the_real_shape_promotes(tmp_path):
    """The measured 2026-08-02 state: prod 42d stale (06-21), candidate
    trained 08-02 with genuine_ic +0.0029, gate REJECT."""
    v = F.decide(_prod(tmp_path), _staging(tmp_path), AS_OF)
    assert v["decision"] == "FALLBACK_PROMOTE", v
    assert v["genuine_ic"] == pytest.approx(0.0029)
    assert v["prod_staleness_days"] == 49


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


def test_negative_genuine_ic_is_never_served(tmp_path):
    v = F.decide(_prod(tmp_path), _staging(tmp_path, genuine=-0.0001), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "genuine_ic_nonnegative"


def test_missing_genuine_ic_refuses(tmp_path):
    v = F.decide(_prod(tmp_path), _staging(tmp_path, genuine=None), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "genuine_ic_present"


@pytest.mark.parametrize("bad", ["0.003", True, float("nan"), float("inf")])
def test_non_number_genuine_ic_refuses_not_coerces(tmp_path, bad):
    """The orch#770/pipeline#259 class: a string or bool must never pass a
    numeric gate by coercion."""
    v = F.decide(_prod(tmp_path), _staging(tmp_path, genuine=bad), AS_OF)
    assert v["decision"] == "REFUSE"
    assert v["refused_on"] in ("genuine_ic_present", "genuine_ic_nonnegative")


def test_gate_passed_candidate_uses_the_normal_path(tmp_path):
    v = F.decide(_prod(tmp_path), _staging(tmp_path, verdict=True), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "gate_rejected"


def test_unstamped_staging_refuses(tmp_path):
    v = F.decide(_prod(tmp_path), _staging(tmp_path, stamp=False), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "staging_gate_stamp"


def test_ratchet_refuses_equal_and_lower(tmp_path):
    prod = _prod(tmp_path, basis=F.PROMOTION_BASIS, prior_g=0.0029)
    v = F.decide(prod, _staging(tmp_path, genuine=0.0029), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "ratchet_up_only"
    v = F.decide(prod, _staging(tmp_path, genuine=0.0010), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "ratchet_up_only"


def test_ratchet_promotes_strictly_higher(tmp_path):
    prod = _prod(tmp_path, basis=F.PROMOTION_BASIS, prior_g=0.0029)
    v = F.decide(prod, _staging(tmp_path, genuine=0.0030), AS_OF)
    assert v["decision"] == "FALLBACK_PROMOTE"


def test_fallback_prod_with_missing_prior_refuses(tmp_path):
    """A fallback-promoted prod with no stamped prior cannot prove the chain
    walks up — refuse rather than guess."""
    prod = _prod(tmp_path, basis=F.PROMOTION_BASIS, prior_g=None)
    v = F.decide(prod, _staging(tmp_path, genuine=0.5), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "ratchet_prior_readable"


def test_gate_passed_prod_has_no_ratchet_bar(tmp_path):
    v = F.decide(_prod(tmp_path, basis=None), _staging(tmp_path, genuine=0.0001), AS_OF)
    assert v["decision"] == "FALLBACK_PROMOTE"


def test_unreadable_inputs_refuse(tmp_path):
    v = F.decide(tmp_path / "nope.json", _staging(tmp_path), AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "prod_readable"
    v = F.decide(_prod(tmp_path), tmp_path / "nope.json", AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "staging_readable"


def test_stamp_writes_atomically_and_only_on_promote(tmp_path):
    staging = _staging(tmp_path)
    v = F.decide(_prod(tmp_path), staging, AS_OF)
    F.stamp(staging, v)
    obj = json.loads(staging.read_text())
    assert obj["metadata"]["promotion_basis"] == F.PROMOTION_BASIS
    assert obj["metadata"]["fallback_genuine_ic"] == pytest.approx(0.0029)
    assert obj["metadata"]["fallback_as_of"] == "2026-08-09"
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


def test_verdict_names_every_check_with_values(tmp_path):
    v = F.decide(_prod(tmp_path), _staging(tmp_path), AS_OF)
    names = [c["check"] for c in v["checks"]]
    assert names == ["gate_rejected", "prod_stale", "candidate_recent",
                     "genuine_ic_nonnegative", "ratchet_up_only"]



@pytest.mark.parametrize("bad", [None, "False", 0, 1, "rejected"])
def test_non_boolean_stamped_verdict_refuses_not_permits(tmp_path, bad):
    """[codex on #102] an absent/corrupted/never-recorded gate decision must
    never become permission for a production promotion — only an explicit
    boolean False proceeds. (None here means the producer stamped the key as
    null; a MISSING key is the unstamped case covered above.)"""
    staging = _write(tmp_path / "staging.json",
                     {"trained_date": "2026-08-02",
                      "metadata": {"wf_gate_metadata": {
                          "passed": bad,
                          "sanity_placebo_genuine_ic": 0.0029}}})
    v = F.decide(_prod(tmp_path), staging, AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "gate_rejected"


def test_missing_passed_key_refuses(tmp_path):
    staging = _write(tmp_path / "staging.json",
                     {"trained_date": "2026-08-02",
                      "metadata": {"wf_gate_metadata": {
                          "sanity_placebo_genuine_ic": 0.0029}}})
    v = F.decide(_prod(tmp_path), staging, AS_OF)
    assert v["decision"] == "REFUSE" and v["refused_on"] == "gate_rejected"


def test_the_REAL_rejected_artifact_shape_still_promotes(tmp_path):
    """The 2026-08-02 staging stamp carries passed=False explicitly (bool) —
    the producer contract was already correct; the fix was the consumer's
    key. This pins the real shape end-to-end."""
    v = F.decide(_prod(tmp_path), _staging(tmp_path, verdict=False), AS_OF)
    assert v["decision"] == "FALLBACK_PROMOTE"
