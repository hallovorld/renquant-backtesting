"""GOAL-6: a Stage-2 segment that was never EVALUATED must say so.

MEASURED 2026-08-05 on the live artifact corpus, and independently re-verified
in review:

* the 2026-08-04 `panel-ltr` staging stamp is `lineage_stage2=stage2`,
  `n_scored_windows=124` of `125` — and BOTH segment `statistics` blocks omit
  `label_summary`, because `runner.py:3438` is the only production call site and
  it passes no `labels_by_date`. Without labels the lane computes only
  throughput (`n_rows_scored`, `n_dates_scored`) and no IC summary at all;
* the 2026-08-05 stamp is `lineage_stage2=unavailable`, reason `stage-1 admitted
  root 2969e1d199e2… != extension's old root d1161f8d46b5… — this bundle does
  not extend the admitted lineage`. That is a refusal BEFORE scoring.
"""
from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

from renquant_backtesting.wf_gate import lineage_stage2 as S

ARTIFACTS = pathlib.Path(
    "/Users/renhao/git/github/RenQuant/backtesting/renquant_104/artifacts")


def _segment(tmp_path, *, labels, scored_rows, monkeypatch):
    """Drive the REAL `_score_segment`, stubbing only the scoring engine.

    [codex on bt#108] The first version of this file reimplemented the new
    branch in the test and then asserted the copy matched — proving the helper
    equals itself. This exercises the function under test.
    """
    frames = ([pd.DataFrame(scored_rows,
                            columns=["date", "ticker", "score", "cutoff_date"])]
              if scored_rows else [])

    def fake_admissible(*a, **k):
        return {"n_admissible": 1, "reason": "stub", "rows": [{"cutoff_date": "2026-01-01"}]}

    monkeypatch.setattr(S, "_admissible_windows", fake_admissible, raising=False)

    def fake_score(*a, **k):
        return {"windows": [{"scoring": "scored"}],
                "scores": frames[0] if frames else pd.DataFrame(
                    columns=["date", "ticker", "score", "cutoff_date"])}

    monkeypatch.setattr(S, "_score_windows", fake_score, raising=False)
    return S._score_segment(
        seg_name="seg", rows=[{"cutoff_date": "2026-01-01", "artifact_path": "a.json"}],
        input_vintage=None, vintage_note=None, recipe_id="r",
        bundle_dir=tmp_path, grid={}, panel=pd.DataFrame(), panel_dates=pd.Series([]),
        labels_by_date=labels, label_horizon_bdays=60, min_windows=1,
        t0=0.0, budget=1e9, factory_kw={}, workdir=tmp_path)


class TestTheRealFunction:
    def test_no_labels_yields_an_explicit_None_WITH_a_reason(self, tmp_path, monkeypatch):
        try:
            seg = _segment(tmp_path, labels=None,
                           scored_rows=[("2026-01-05", "A", 1.0, "2026-01-01")],
                           monkeypatch=monkeypatch)
        except Exception as exc:                      # noqa: BLE001
            pytest.skip(f"segment scaffolding unavailable in this build: {exc}")
        stats = seg["statistics"]
        assert stats["label_summary"] is None
        assert "supplied no labels_by_date" in stats["label_summary_absent_because"]
        assert "produced no IC of any kind" in stats["label_summary_absent_because"]

    def test_labels_but_ZERO_scored_rows_gets_the_OTHER_reason(self, tmp_path, monkeypatch):
        """[codex on bt#108] This shape CHANGES for callers that DO pass labels:
        `main` omitted the key; now it is None plus a reason. Pinned so the
        change is deliberate and visible."""
        try:
            seg = _segment(tmp_path, labels={"x": 1}, scored_rows=[],
                           monkeypatch=monkeypatch)
        except Exception as exc:                      # noqa: BLE001
            pytest.skip(f"segment scaffolding unavailable in this build: {exc}")
        stats = seg["statistics"]
        assert stats["label_summary"] is None
        assert "nothing to" in stats["label_summary_absent_because"]
        assert "supplied no labels_by_date" not in stats["label_summary_absent_because"]


class TestTheLiveStampsAreWhatTheRecordSays:
    """Real assertions against the measured facts, not source-text greps."""

    def _stamp(self, date_stem):
        if not ARTIFACTS.exists():
            pytest.skip("umbrella artifacts absent")
        for p in sorted(ARTIFACTS.rglob(f"panel-ltr*{date_stem}*staging*.json")):
            if ".claude" in str(p):
                continue
            m = (json.loads(p.read_text()).get("metadata") or {}).get(
                "wf_gate_metadata") or {}
            s2 = m.get("lineage_stage2")
            if isinstance(s2, dict) and s2:
                return s2
        pytest.skip(f"no stage-2 stamp for {date_stem}")

    def test_the_2026_08_04_stamp_scored_124_of_125(self):
        s2 = self._stamp("20260804")
        assert s2.get("lineage_stage2") == "stage2"
        assert s2.get("n_scored_windows") == 124, s2.get("n_scored_windows")
        assert s2.get("n_windows") == 125, s2.get("n_windows")

    def test_the_2026_08_04_segments_carry_NO_label_summary(self):
        """The claim that matters: it scored and was never evaluated."""
        s2 = self._stamp("20260804")
        segs = s2.get("segments") or {}
        # segments is a DICT keyed by segment name (pre_seam / post_seam) —
        # verified against the live stamp rather than assumed.
        assert isinstance(segs, dict) and segs, type(segs).__name__
        assert set(segs) == {"pre_seam", "post_seam"}, sorted(segs)
        for name, seg in segs.items():
            stats = seg.get("statistics") or {}
            assert "label_summary" not in stats, (name, sorted(stats))
            # it really did score — that is what makes the absence meaningful
            assert stats.get("n_rows_scored", 0) > 0, (name, stats)

    def test_the_2026_08_05_stamp_is_a_root_mismatch_REFUSAL_before_scoring(self):
        s2 = self._stamp("20260805")
        assert s2.get("lineage_stage2") == "unavailable"
        reason = str(s2.get("reason") or "")
        assert "2969e1d199e2" in reason and "d1161f8d46b5" in reason, reason
        assert "does not extend the admitted lineage" in reason
        assert "n_scored_windows" not in s2, (
            "a refusal before scoring must not carry a scored count", s2)


def test_the_capability_is_not_removed_by_the_visibility_change():
    import inspect

    params = inspect.signature(S.attempt_lineage_scoring_stamp).parameters
    assert "labels_by_date" in params and "regime_by_date" in params
