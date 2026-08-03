"""Stage-2 scoring-lane tests: never-raises, seam-separated pooling, content
binding, admission untouched. The module ships UNWIRED — a source guard below
pins runner.py free of it until the operator's stage-2 sign-off on #94."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from renquant_backtesting.wf_gate import lineage_stage2 as L2
from renquant_backtesting.wf_gate.lineage_admissibility import lineage_root_sha

RECIPE = "sha256:testrecipe00"


def _write_artifact(path: Path, cutoff: str) -> str:
    """A minimal admissible snapshot; returns its content sha."""
    path.parent.mkdir(parents=True, exist_ok=True)
    etc = pd.Timestamp(cutoff) - pd.offsets.BDay(85)
    path.write_text(json.dumps({
        "cutoff_date": cutoff, "cutoff_embargo_days": 60,
        "effective_train_cutoff_date": str(etc.date())}))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mk_bundle(tmp_path: Path, n_new: int = 3, n_old: int = 3):
    """A synthetic two-segment extension bundle mirroring run-001's shape:
    new (pre-seam) windows relative to the bundle dir, existing (post-seam)
    windows by absolute path, seam recorded first-class."""
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    new_rows, old_rows = [], []
    # pre-seam: earlier cutoffs, 21-day cadence, stamped with the seam vintage
    for i in range(n_new):
        cut = str((pd.Timestamp("2024-01-01") + pd.Timedelta(days=21 * i)).date())
        rel = f"window_artifacts/{cut}/panel-ltr.json"
        sha = _write_artifact(bundle / rel, cut)
        new_rows.append({"cutoff_date": cut, "artifact_path": rel,
                         "artifact_sha256": sha,
                         "input_vintage": "2026-08-01-rebuild",
                         "provenance": "jobb_depth_extension"})
    # post-seam: later cutoffs, absolute paths (the production-ladder shape)
    prod = tmp_path / "prod_ladder"
    for i in range(n_old):
        cut = str((pd.Timestamp("2024-06-03") + pd.Timedelta(days=21 * i)).date())
        p = prod / cut / "panel-ltr.json"
        sha = _write_artifact(p, cut)
        old_rows.append({"cutoff_date": cut, "artifact_path": str(p),
                         "artifact_sha256": sha,
                         "provenance": "existing_prod_wf_manifest"})
    old_shas = [r["artifact_sha256"] for r in old_rows]
    all_shas = [r["artifact_sha256"] for r in new_rows] + old_shas
    man = {
        "schema": "gbdt-depth-extension-lineage-v1",
        "recipe_id": RECIPE,
        "old_lineage_root_sha": lineage_root_sha(RECIPE, old_shas),
        "old_lineage_n_windows": n_old,
        "new_lineage_root_sha": lineage_root_sha(RECIPE, all_shas),
        "new_lineage_n_windows": n_new + n_old,
        "vintage_seam": {"input_vintage": "2026-08-01-rebuild",
                         "evidence_golden_report_sha256": "e" * 64,
                         "golden_parity_max_abs_delta": 0.649},
        "new_windows": new_rows,
        "existing_windows": old_rows,
    }
    mpath = bundle / "gbdt_depth_extension_manifest.json"
    mpath.write_text(json.dumps(man))
    stage1 = {"lineage_lane": "stage1", "lineage_admissibility": "admissible",
              "lineage_root_sha": man["old_lineage_root_sha"],
              "recipe_id": RECIPE}
    return mpath, man, stage1


def _panel_for(man: dict, tickers=("AAA", "BBB", "CCC")) -> pd.DataFrame:
    """One trading date per non-final window: the first BDay after each
    cutoff (inside (cut, next_cut] for a 21-day cadence)."""
    rows = []
    for w in man["new_windows"] + man["existing_windows"]:
        d = pd.Timestamp(w["cutoff_date"]) + pd.offsets.BDay(1)
        for i, t in enumerate(tickers):
            rows.append({"date": d, "ticker": t, "f1": float(i)})
    return pd.DataFrame(rows)


def _ok_factory(artifact, path):
    return lambda sub: pd.Series(np.arange(len(sub), dtype=float),
                                 index=sub.index)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _attempt(mpath, man, stage1, panel, **kw):
    defaults = dict(stage1=stage1, extension_manifest_path=mpath, panel=panel,
                    label_horizon_bdays=60,
                    min_admissible_windows_per_segment=2,
                    scorer_factory=_ok_factory)
    if "expected_manifest_sha256" not in kw:
        defaults["expected_manifest_sha256"] = _sha(mpath)  # lazy: only when
        # the test does not pin its own bytes (mpath may be deliberately absent)
    defaults.update(kw)
    return L2.attempt_lineage_scoring_stamp(**defaults)


# --------------------------------------------------------------------------
# happy path: two segments, pooled separately, combined pool ABSENT
# --------------------------------------------------------------------------

def test_happy_path_two_segments_pool_separately_no_combined_pool(tmp_path):
    mpath, man, s1 = _mk_bundle(tmp_path)
    out = _attempt(mpath, man, s1, _panel_for(man))
    assert out["lineage_stage2"] == "stage2", out.get("reason")
    assert out["candidate_lineage_used"] is True
    assert out["candidate_artifact_used"] is False       # documented, honest
    segs = out["segments"]
    assert list(segs) == ["pre_seam", "post_seam"]
    # membership: pre_seam windows are exactly the manifest's new_windows
    pre_cuts = {w["cutoff_date"] for w in segs["pre_seam"]["windows"]}
    post_cuts = {w["cutoff_date"] for w in segs["post_seam"]["windows"]}
    assert pre_cuts == {w["cutoff_date"] for w in man["new_windows"]}
    assert post_cuts == {w["cutoff_date"] for w in man["existing_windows"]}
    assert not (pre_cuts & post_cuts)
    # pre-seam scores all 3; post-seam's FINAL window has no closing edge
    assert segs["pre_seam"]["n_scored_windows"] == 3
    assert segs["post_seam"]["n_scored_windows"] == 2
    assert segs["pre_seam"]["scoring_verdict"] == "scored"
    assert segs["post_seam"]["scoring_verdict"] == "scored"
    # vintage labels: seam vintage on pre, none (with the note) on post
    assert segs["pre_seam"]["input_vintage"] == "2026-08-01-rebuild"
    assert segs["post_seam"]["input_vintage"] is None
    assert "vintage_note" in segs["post_seam"]
    # the seam marker is explicit and names the boundary
    seam = out["vintage_seam"]
    assert seam["seam_boundary_cutoffs"] == [
        man["new_windows"][-1]["cutoff_date"],
        man["existing_windows"][0]["cutoff_date"]]
    assert "NO cross-seam" in seam["pooling"]
    # the COMBINED pool is ABSENT: no top-level score statistic exists, only
    # per-segment statistics blocks
    assert "statistics" not in out
    assert "mean_ic" not in out and "label_summary" not in out
    assert "statistics" in segs["pre_seam"] and "statistics" in segs["post_seam"]


def test_labels_summarized_per_segment_never_across_the_seam(tmp_path):
    """Pre-seam scorer aligns with labels (IC +1), post-seam anti-aligns
    (IC -1). Separate pools MUST show +1 and -1; any cross-seam pool would
    average them away — its absence plus these values is the proof."""
    mpath, man, s1 = _mk_bundle(tmp_path)
    panel = _panel_for(man)
    new_cuts = {w["cutoff_date"] for w in man["new_windows"]}

    def _side_factory(artifact, path):
        sign = 1.0 if artifact["cutoff_date"] in new_cuts else -1.0
        return lambda sub: pd.Series(
            sign * np.arange(len(sub), dtype=float), index=sub.index)

    tickers = [f"T{i:02d}" for i in range(25)]   # clear the >=20-name IC floor
    rows = []
    for w in man["new_windows"] + man["existing_windows"]:
        d = pd.Timestamp(w["cutoff_date"]) + pd.offsets.BDay(1)
        rows.extend({"date": d, "ticker": t, "f1": float(i)}
                    for i, t in enumerate(tickers))
    panel = pd.DataFrame(rows)
    labels = {pd.Timestamp(r): pd.Series(np.arange(25, dtype=float),
                                         index=tickers)
              for r in panel["date"].unique()}
    out = _attempt(mpath, man, s1, panel, labels_by_date=labels,
                   scorer_factory=_side_factory)
    assert out["lineage_stage2"] == "stage2", out.get("reason")
    pre = out["segments"]["pre_seam"]["statistics"]["label_summary"]
    post = out["segments"]["post_seam"]["statistics"]["label_summary"]
    assert pre["mean_ic"] == 1.0
    assert post["mean_ic"] == -1.0
    # and there is no key anywhere at the top level pooling the two
    assert "label_summary" not in out and "mean_ic" not in out


# --------------------------------------------------------------------------
# identity: both roots + the manifest content sha, all content-bound
# --------------------------------------------------------------------------

def test_stamp_carries_both_roots_and_the_manifest_sha(tmp_path):
    mpath, man, s1 = _mk_bundle(tmp_path)
    out = _attempt(mpath, man, s1, _panel_for(man))
    assert out["extension_manifest_sha256"] == _sha(mpath)
    assert out["lineage_root_sha_old"] == man["old_lineage_root_sha"]
    assert out["lineage_root_sha_full"] == man["new_lineage_root_sha"]
    assert out["stage1_lineage_root_match"] is True


def test_content_binding_changed_manifest_bytes_refuse(tmp_path):
    mpath, man, s1 = _mk_bundle(tmp_path)
    pin = _sha(mpath)                                   # pin the ORIGINAL bytes
    tampered = json.loads(mpath.read_text())
    tampered["wall_seconds"] = 1.0                      # any byte change at all
    mpath.write_text(json.dumps(tampered))
    out = _attempt(mpath, man, s1, _panel_for(man),
                   expected_manifest_sha256=pin)
    assert out["lineage_stage2"] == "unavailable"
    assert "content pin mismatch" in out["reason"]


def test_missing_content_pin_refuses(tmp_path):
    mpath, man, s1 = _mk_bundle(tmp_path)
    out = _attempt(mpath, man, s1, _panel_for(man),
                   expected_manifest_sha256=None)
    assert out["lineage_stage2"] == "unavailable"
    assert "content pin" in out["reason"]


def test_tampered_window_artifact_refuses_its_whole_segment(tmp_path):
    """Manifest pin still matches (manifest unchanged) but one existing
    artifact's bytes differ from the declared sha: the on-disk re-digest must
    refuse the POST-seam segment while the pre-seam segment still scores."""
    mpath, man, s1 = _mk_bundle(tmp_path)
    victim = Path(man["existing_windows"][1]["artifact_path"])
    art = json.loads(victim.read_text())
    art["cutoff_embargo_days"] = 61                     # wrong-but-plausible
    victim.write_text(json.dumps(art))
    out = _attempt(mpath, man, s1, _panel_for(man))
    assert out["lineage_stage2"] == "stage2"            # a stamped refusal,
    segs = out["segments"]                              # not a lane failure
    assert segs["pre_seam"]["scoring_verdict"] == "scored"
    assert segs["post_seam"]["admissibility_verdict"] == "refused"
    assert segs["post_seam"]["scoring_verdict"] == "refused"
    assert segs["post_seam"]["n_scored_windows"] == 0
    assert all(w["scoring"] == "skipped_segment_refused"
               for w in segs["post_seam"]["windows"])


def test_declared_sha_tamper_breaks_the_root_and_refuses(tmp_path):
    mpath, man, s1 = _mk_bundle(tmp_path)
    m = json.loads(mpath.read_text())
    m["new_windows"][0]["artifact_sha256"] = "0" * 64
    mpath.write_text(json.dumps(m))
    out = _attempt(mpath, man, s1, _panel_for(man))     # pin recomputed fresh
    assert out["lineage_stage2"] == "unavailable"
    assert "root recomputed" in out["reason"]


# --------------------------------------------------------------------------
# stage-1 cross-lane binding
# --------------------------------------------------------------------------

def test_stage1_gate_absent_unavailable_or_root_mismatch_all_refuse(tmp_path):
    mpath, man, s1 = _mk_bundle(tmp_path)
    panel = _panel_for(man)
    a = _attempt(mpath, man, {"lineage_lane": "unavailable", "reason": "x"},
                 panel)
    assert a["lineage_stage2"] == "unavailable" and "stage-1" in a["reason"]
    b = _attempt(mpath, man, {**s1, "lineage_admissibility": "refused"}, panel)
    assert b["lineage_stage2"] == "unavailable" and "not admissible" in b["reason"]
    c = _attempt(mpath, man, {**s1, "lineage_root_sha": "f" * 64}, panel)
    assert c["lineage_stage2"] == "unavailable"
    assert "does not extend the admitted lineage" in c["reason"]
    d = _attempt(mpath, man, {**s1, "recipe_id": "sha256:other"}, panel)
    assert d["lineage_stage2"] == "unavailable" and "recipe_id" in d["reason"]


# --------------------------------------------------------------------------
# structural / seam integrity + never-raises
# --------------------------------------------------------------------------

def test_structural_violations_are_stamped_unavailable(tmp_path):
    mpath, man, s1 = _mk_bundle(tmp_path)
    panel = _panel_for(man)

    def _mutated(fn):
        m = json.loads(mpath.read_text())
        fn(m)
        p2 = mpath.parent / "mut.json"
        p2.write_text(json.dumps(m))
        return _attempt(p2, man, s1, panel, expected_manifest_sha256=_sha(p2))

    a = _mutated(lambda m: m.update(schema="v2-unknown"))
    assert a["lineage_stage2"] == "unavailable" and "schema" in a["reason"]
    b = _mutated(lambda m: m["new_windows"][0].pop("input_vintage"))
    assert b["lineage_stage2"] == "unavailable" and "input_vintage" in b["reason"]
    c = _mutated(lambda m: m["existing_windows"][0].update(
        input_vintage="2026-08-01-rebuild"))
    assert c["lineage_stage2"] == "unavailable" and "seam" in c["reason"]
    d = _mutated(lambda m: m.update(old_lineage_n_windows=99))
    assert d["lineage_stage2"] == "unavailable" and "old_lineage_n_windows" in d["reason"]
    # new windows AFTER existing ones: the append-only-backwards rule broken
    e = _mutated(lambda m: m.update(
        new_windows=[{**m["new_windows"][0], "cutoff_date": "2025-01-06"}]))
    assert e["lineage_stage2"] == "unavailable"


def test_never_raises_on_garbage_inputs(tmp_path):
    mpath, man, s1 = _mk_bundle(tmp_path)
    panel = _panel_for(man)
    # absent manifest file
    a = _attempt(tmp_path / "absent.json", man, s1, panel,
                 expected_manifest_sha256="a" * 64)
    assert a["lineage_stage2"] == "unavailable" and "missing" in a["reason"]
    # unparseable manifest matching its own pin
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    b = _attempt(bad, man, s1, panel, expected_manifest_sha256=_sha(bad))
    assert b["lineage_stage2"] == "unavailable"
    # panel without a 'date' column
    c = _attempt(mpath, man, s1, pd.DataFrame({"ticker": ["A"]}))
    assert c["lineage_stage2"] == "unavailable" and "date" in c["reason"]
    # a window artifact file deleted: below the per-segment minimum -> refused
    Path(man["existing_windows"][0]["artifact_path"]).unlink()
    Path(man["existing_windows"][1]["artifact_path"]).unlink()
    d = _attempt(mpath, man, s1, panel)
    assert d["lineage_stage2"] == "stage2"
    assert d["segments"]["post_seam"]["admissibility_verdict"] == "refused"
    assert d["segments"]["pre_seam"]["scoring_verdict"] == "scored"


def test_final_window_is_refused_never_invented_unless_caller_grids_it(tmp_path):
    mpath, man, s1 = _mk_bundle(tmp_path)
    out = _attempt(mpath, man, s1, _panel_for(man))
    last = out["segments"]["post_seam"]["windows"][-1]
    assert last["scoring"] == "scoring_error"
    assert "no closing edge" in last["scoring_reason"]
    # an explicit caller grid covering every window scores the final one too
    grid = {}
    for w in man["new_windows"] + man["existing_windows"]:
        grid[w["cutoff_date"]] = [pd.Timestamp(w["cutoff_date"])
                                  + pd.offsets.BDay(1)]
    out2 = _attempt(mpath, man, s1, _panel_for(man), oos_dates_by_cutoff=grid)
    assert out2["segments"]["post_seam"]["n_scored_windows"] == 3


def test_time_budget_exceeded_is_a_stamped_unavailable(tmp_path):
    mpath, man, s1 = _mk_bundle(tmp_path)

    def _slow(artifact, path):
        time.sleep(0.05)
        return _ok_factory(artifact, path)

    out = _attempt(mpath, man, s1, _panel_for(man), scorer_factory=_slow,
                   time_budget_seconds=0.02)
    assert out["lineage_stage2"] == "unavailable"
    assert "time budget exceeded" in out["reason"]


def test_slow_FINAL_scoring_call_is_stamped_unavailable(tmp_path, monkeypatch):
    """Review round 2 regression: the budget used to be polled only BEFORE
    each window, so when the FINAL eligible window's scoring call crossed the
    budget there was no subsequent pre-check and the lane returned a NORMAL
    stage-2 stamp with elapsed_seconds over budget. The deadline must bite
    immediately AFTER the call. Fake clock: only the final call advances it,
    so every pre-window check passes deterministically (no sleep flakiness)."""
    mpath, man, s1 = _mk_bundle(tmp_path)
    last_cut = man["existing_windows"][-1]["cutoff_date"]
    # explicit caller grid: the FINAL ladder window is eligible and scored
    # LAST — the exact no-subsequent-pre-check position the review names
    grid = {w["cutoff_date"]: [pd.Timestamp(w["cutoff_date"])
                               + pd.offsets.BDay(1)]
            for w in man["new_windows"] + man["existing_windows"]}
    clock = {"t": 0.0}
    monkeypatch.setattr(L2.time, "monotonic", lambda: clock["t"])

    def _slow_only_final(artifact, path):
        if artifact["cutoff_date"] == last_cut:
            clock["t"] += 999.0        # the call itself crosses the budget
        return _ok_factory(artifact, path)

    out = _attempt(mpath, man, s1, _panel_for(man), oos_dates_by_cutoff=grid,
                   scorer_factory=_slow_only_final, time_budget_seconds=300.0)
    assert out["lineage_stage2"] == "unavailable"
    assert "time budget exceeded" in out["reason"]
    assert "after scoring window" in out["reason"]       # post-call detection
    # never a normal stamp: no elapsed_seconds, no segments, nothing scored
    assert "elapsed_seconds" not in out
    assert "segments" not in out


def test_budget_crossed_after_the_last_call_caught_at_return_boundary(
        tmp_path, monkeypatch):
    """Segment post-processing (the label summary) runs AFTER the last
    scoring call; a budget crossed there has no per-window check left, so the
    successful-return boundary check must convert it into the same stamped
    unavailable."""
    mpath, man, s1 = _mk_bundle(tmp_path)
    panel = _panel_for(man)
    labels = {pd.Timestamp(d): pd.Series([0.0, 1.0, 2.0],
                                         index=["AAA", "BBB", "CCC"])
              for d in panel["date"].unique()}
    clock = {"t": 0.0}
    monkeypatch.setattr(L2.time, "monotonic", lambda: clock["t"])
    calls = {"n": 0}

    def _slow_summary(scores, labels_by_date):
        calls["n"] += 1
        if calls["n"] == 2:            # post_seam's summary: the LAST work
            clock["t"] += 999.0
        return {"stub": True}

    monkeypatch.setattr(L2.LS, "summarize_lineage_scores", _slow_summary)
    out = _attempt(mpath, man, s1, panel, labels_by_date=labels,
                   time_budget_seconds=300.0)
    assert out["lineage_stage2"] == "unavailable"
    assert "time budget exceeded" in out["reason"]
    assert "successful-return boundary" in out["reason"]
    assert "elapsed_seconds" not in out


# --------------------------------------------------------------------------
# admission untouched: behavioural + source-level guards
# --------------------------------------------------------------------------

def test_inputs_are_never_mutated(tmp_path):
    mpath, man, s1 = _mk_bundle(tmp_path)
    panel = _panel_for(man)
    s1_before = json.dumps(s1, sort_keys=True)
    panel_before = panel.copy(deep=True)
    out = _attempt(mpath, man, s1, panel)
    assert out["lineage_stage2"] == "stage2"            # the full path ran
    assert json.dumps(s1, sort_keys=True) == s1_before
    pd.testing.assert_frame_equal(panel, panel_before)


def test_runner_carries_NO_reference_to_stage2_until_signoff():
    """This slice ships the module UNWIRED (#94: per-stage operator sign-off;
    the wiring lands as its own reviewed change after 'approved stage 2').
    The wiring PR must consciously delete this assertion — severability is
    mechanical, not a promise."""
    src = (Path(__file__).resolve().parent.parent /
           "src/renquant_backtesting/wf_gate/runner.py").read_text()
    assert "lineage_stage2" not in src
    assert "attempt_lineage_scoring_stamp" not in src


def test_module_touches_no_admission_surface():
    """Source-level guard in the Stage-1 style: the module never references
    the admission subject (artifact_usage) and never imports the runner."""
    import ast
    src = (Path(__file__).resolve().parent.parent /
           "src/renquant_backtesting/wf_gate/lineage_stage2.py").read_text()
    assert "artifact_usage" not in src
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            assert "runner" not in str(node.module), ast.dump(node)
        if isinstance(node, ast.Import):
            assert all("runner" not in a.name for a in node.names)


# --------------------------------------------------------------------------
# the REAL run-001 bundle (loud env-skips; no panel required)
# --------------------------------------------------------------------------

BUNDLE_REL = "doc/research/data/2026-08-02-jobb-gbdt-depth-extension-run001"
OLD_ROOT = "d1161f8d46b57de77374c82c5916d814c1b72e662ef137d8627cb5ea178e0a3f"
FULL_ROOT = "83496eacf91b218ae0602ea6066e473cab6489262b268d4545cb1542ad460228"
MANIFEST_SHA = "b70119eb00cc9c32e43b8520ca1bf912e630beb040dc8cf2ff333a367a1e800d"


def _model_repo() -> Path:
    return Path(__file__).resolve().parents[2] / "renquant-model"


def test_REAL_run001_bundle_identity_seam_and_admissibility():
    """The full lane against the committed run-001 bundle: 125 on-disk digests
    re-verified, both roots recomputed to the published values, the seam
    boundary stamped, both segments scored (final window refused — no closing
    edge). Scoring uses an injected constant factory (a value-golden needs the
    panel AND a committed expected-score corpus, which run-001 does not carry
    — artifacts only); the recipe-transform default stays covered by the
    micro-test below and #96's golden."""
    import pytest
    repo = _model_repo()
    mp = repo / BUNDLE_REL / "gbdt_depth_extension_manifest.json"
    if not mp.is_file():
        pytest.skip("sibling renquant-model checkout lacks the run-001 bundle")
    # identity is pinned by the MANIFEST_SHA content pin below, not by any git
    # ref — a stale checkout fails the pin loudly instead of testing old bytes
    man = json.loads(mp.read_text())
    if not Path(man["existing_windows"][0]["artifact_path"]).is_file():
        pytest.skip("umbrella-tree production window artifacts absent here")
    stage1 = {"lineage_lane": "stage1", "lineage_admissibility": "admissible",
              "lineage_root_sha": OLD_ROOT,
              "recipe_id": man["recipe_id"]}
    # a synthetic 3-ticker panel: one trading date inside every (cut, next]
    rows = []
    for w in man["new_windows"] + man["existing_windows"]:
        d = pd.Timestamp(w["cutoff_date"]) + pd.offsets.BDay(1)
        rows.extend({"date": d, "ticker": t, "f1": 0.0}
                    for t in ("AAA", "BBB", "CCC"))
    out = L2.attempt_lineage_scoring_stamp(
        stage1=stage1, extension_manifest_path=mp,
        expected_manifest_sha256=MANIFEST_SHA,
        panel=pd.DataFrame(rows), label_horizon_bdays=60,
        scorer_factory=_ok_factory)
    assert out["lineage_stage2"] == "stage2", out.get("reason")
    assert out["extension_manifest_sha256"] == MANIFEST_SHA
    assert out["lineage_root_sha_old"] == OLD_ROOT
    assert out["lineage_root_sha_full"] == FULL_ROOT
    segs = out["segments"]
    assert segs["pre_seam"]["n_windows"] == 82
    assert segs["post_seam"]["n_windows"] == 43
    assert segs["pre_seam"]["n_admissible"] == 82
    assert segs["post_seam"]["n_admissible"] == 43
    assert segs["pre_seam"]["n_scored_windows"] == 82
    assert segs["post_seam"]["n_scored_windows"] == 42   # final: no closing edge
    assert out["vintage_seam"]["seam_boundary_cutoffs"] == [
        "2023-09-11", "2023-10-02"]
    assert segs["pre_seam"]["input_vintage"] == "2026-08-01-rebuild"


def test_default_factory_accepts_a_committed_run001_artifact():
    """Stage-2's default factory (the fail-closed re-keying adapter over the
    #96 normative load_fold_scorer path) loads a REAL committed extension
    window artifact and scores a synthetic ticker-indexed frame — no panel
    anywhere. Loud env-skip when the model checkout is stale. Measured
    2026-08-02: the gbdt window artifacts carry feature_means/stds as ORDERED
    LISTS (172 entries aligned to feature_cols, per the panel_trainer writer),
    which the un-adapted public contract refuses — the adapter is load-bearing."""
    import importlib.util
    import pytest
    if importlib.util.find_spec("renquant_model_gbdt.fold_scoring") is None:
        pytest.skip("resolvable renquant-model predates 0.2.0 (no fold_scoring)")
    art_p = (_model_repo() / BUNDLE_REL /
             "window_artifacts/2019-01-14/panel-ltr.json")
    if not art_p.is_file():
        pytest.skip("sibling renquant-model checkout lacks the run-001 bundle")
    art = json.loads(art_p.read_text())
    assert isinstance(art["feature_means"], list)        # the shape that bit
    score = L2.gbdt_window_scorer_factory(art, art_p)
    frame = pd.DataFrame(np.zeros((5, len(art["feature_cols"]))),
                         columns=art["feature_cols"],
                         index=pd.Index([f"T{i}" for i in range(5)],
                                        name="ticker"))
    s = score(frame)
    assert list(s.index) == list(frame.index)
    assert np.isfinite(s.to_numpy()).all()


def test_adapter_refuses_misaligned_stats_before_any_import():
    """The re-keying adapter must never guess an alignment: a stats list whose
    length disagrees with feature_cols raises BEFORE the model-repo import, so
    this guard runs on any machine."""
    import pytest
    art = {"feature_cols": ["a", "b", "c"],
           "feature_means": [0.0, 1.0],                 # misaligned: 2 vs 3
           "feature_stds": [1.0, 1.0, 1.0]}
    with pytest.raises(ValueError, match="refusing to guess"):
        L2.gbdt_window_scorer_factory(art, Path("x"))
