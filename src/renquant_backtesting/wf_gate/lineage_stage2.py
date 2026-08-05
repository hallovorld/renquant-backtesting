"""Stage-2 scoring-lane slice — #94 implementation slice 4 (UNWIRED module).

ONE entry point, ``attempt_lineage_scoring_stamp``, mirroring Stage-1's
(``lineage_lane.attempt_lineage_stamp``) contract exactly:

* it NEVER raises — every failure yields
  ``{"lineage_stage2": "unavailable", "reason": ...}``;
* admission is NOT touched anywhere in this module; the block is designed to be
  attached by a future caller as a SIBLING key only (``lineage_stage2``), and a
  source-level guard test pins ``runner.py`` free of any reference to this
  module until the operator's stage-2 sign-off on #94 lands — the wiring is a
  separate reviewed change, severable by construction;
* recipe stamps are byte-unchanged: this module receives the Stage-1 block
  read-only and mutates none of its inputs.

Shape decision (the merged #94 design text governs): the design's scoring lane
is IN-RUN — "for each manifest OOS window, score the window's panel rows with
THE CANDIDATE's booster … seconds-to-minutes, far inside the 600 s budget" —
and has no offline-evidence provision; the committed extension bundle carries
ARTIFACTS (boosters), not precomputed scores. So scoring runs in-process
through the #96 engine (``lineage_scoring.score_lineage``); the default scorer
is ``gbdt_window_scorer_factory`` — a fail-closed re-keying adapter over the
#96 normative ``load_fold_scorer`` path (gbdt window artifacts carry
list-shaped feature stats; the public contract wants dicts) — guarded by a
bounded time budget: exceeding the budget is a STAMPED refusal, never an
unbounded gate slowdown. The deadline is enforced at every boundary — before
each scoring call, immediately after each scoring call, and once more before
the successful return — so no stamp ever leaves this module with
``elapsed_seconds`` over budget (the scorer calls are the only long
operations; a hard wall-clock containment boundary was considered and
deferred — see the review round-2 record).

Pooling rule (the stage-2 sign-off's point 2, mechanical here): statistics are
pooled PER input-vintage segment only — the 2026-08-01-rebuild extension
windows (pre-seam, chronologically earlier cutoffs) and the existing
production ladder (post-seam) are pooled SEPARATELY, and this stamp emits NO
cross-seam pooled statistic. A consumer wanting a combined number must compute
it downstream, in the open, from the two segment stamps.

Identity (content binding): the stamp refuses unless
* the extension manifest's bytes hash to the caller-pinned
  ``expected_manifest_sha256`` (the RUN_CLAIM binding);
* the OLD lineage root recomputed over the existing windows' declared shas and
  the FULL-ladder root recomputed over all declared shas both equal the
  manifest's claimed ``old_lineage_root_sha`` / ``new_lineage_root_sha``;
* the Stage-1 block's ``lineage_root_sha`` equals the manifest's OLD root and
  its ``recipe_id`` equals the manifest's — the extension must extend exactly
  the lineage Stage-1 admitted;
* per window, the on-disk artifact bytes re-digest to the manifest's declared
  sha (via ``lineage_admissibility.evaluate_lineage``) — a tampered artifact
  refuses its whole segment (digests attest identity; the root seals it).

The OOS grid is derived from the manifest's own cutoff ladder by the
``(cut, next_cut]`` rule over the CALLER panel's trading dates (the
no-self-attestation rule: the artifacts' own declared windows are never
consulted). The FINAL ladder window has no closing edge in the ladder; it is
REFUSED with a stamped reason, never invented — a caller may score it only by
supplying an explicit ``oos_dates_by_cutoff`` grid.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path

import pandas as pd

from renquant_backtesting.wf_gate import lineage_scoring as LS
from renquant_backtesting.wf_gate.lineage_admissibility import (
    DEFAULT_MIN_ADMISSIBLE_WINDOWS,
    evaluate_lineage,
    lineage_root_sha,
)

#: The one extension-manifest schema this lane understands. Anything else —
#: including a future v2 — is a stamped refusal until this module is revised.
SUPPORTED_SCHEMA = "gbdt-depth-extension-lineage-v1"

#: In-run budget for the whole scoring pass. [推导 — half the 600 s gate budget
#: the #94 design cites as the feasibility envelope, leaving the recipe path
#: its share; overridable per call, never silently.]
DEFAULT_TIME_BUDGET_SECONDS = 300.0

POOLING_RULE = (
    "statistics pooled PER input-vintage segment only; this lane NEVER pools "
    "across the vintage seam — a combined number must be computed downstream, "
    "in the open, from the two segment stamps")


def _unavailable(reason: str) -> dict:
    return {"lineage_stage2": "unavailable", "reason": reason[:300]}


def _budget_guard(*, t0: float, budget: float, where: str) -> None:
    """The whole-pass deadline, enforced at EVERY boundary — before each
    scoring call, immediately after each scoring call, and once more before
    the successful return. (Review round 2: the pre-window poll alone let a
    slow FINAL ``LS.score_lineage`` call — one with no subsequent pre-check —
    ride out of this lane inside a normal stage-2 stamp with
    ``elapsed_seconds`` over budget.) Raises TimeoutError; the lane entry
    point converts it into the one stamped-unavailable budget shape."""
    elapsed = time.monotonic() - t0
    if elapsed > budget:
        raise TimeoutError(
            f"time budget exceeded: {elapsed:.1f}s > {budget:.1f}s "
            f"(detected {where}) — stamped refusal, never an unbounded "
            "gate slowdown")


def gbdt_window_scorer_factory(artifact: dict, artifact_path: Path):
    """Stage-2's default scorer: the #96 normative recipe-transform path
    (``renquant_model_gbdt.fold_scoring.load_fold_scorer``), adapted for the
    gbdt WINDOW artifact shape.

    Measured 2026-08-02: the production ladder's and run-001's window
    artifacts self-carry ``feature_means``/``feature_stds`` as ORDERED LISTS
    aligned to ``feature_cols`` (writer: ``renquant_model_gbdt.panel_trainer``
    — mu/sd arrays are built from ``feat_cols`` by the NormalizationBuilder
    and consumed positionally by ``panel_training_matrix``), while
    ``load_fold_scorer``'s public contract wants dicts keyed by
    ``feature_cols`` (the clf fold shape it was golden-verified on). This
    adapter ONLY re-keys the aligned lists into that dict shape — no transform
    math lives here (the #96 "never reimplemented" rule) — and fail-closes on
    any shape it does not positively recognize. Dict-shaped artifacts pass
    through untouched.
    """
    art = dict(artifact)
    cols = list(art.get("feature_cols") or [])
    for key in ("feature_means", "feature_stds"):
        v = art.get(key)
        if isinstance(v, dict):
            continue                        # already the public-contract shape
        if cols and isinstance(v, list) and len(v) == len(cols):
            art[key] = dict(zip(cols, v))
        else:
            raise ValueError(
                f"{key} is neither a feature_cols-keyed dict nor a list "
                f"aligned to feature_cols (len {len(v) if isinstance(v, list) else '?'}"
                f" vs {len(cols)} cols) — refusing to guess the alignment")
    from renquant_model_gbdt.fold_scoring import load_fold_scorer  # noqa: PLC0415
    return load_fold_scorer(art)


def _structural_check(man: dict) -> str | None:
    """Return a refusal reason, or None if the manifest is structurally sound."""
    if man.get("schema") != SUPPORTED_SCHEMA:
        return (f"unsupported extension-manifest schema {man.get('schema')!r} "
                f"(this lane understands only {SUPPORTED_SCHEMA!r})")
    new = man.get("new_windows") or []
    old = man.get("existing_windows") or []
    if not new or not old:
        return "manifest lacks new_windows or existing_windows"
    if int(man["old_lineage_n_windows"]) != len(old):
        return (f"old_lineage_n_windows {man['old_lineage_n_windows']} != "
                f"{len(old)} existing rows")
    if int(man["new_lineage_n_windows"]) != len(new) + len(old):
        return (f"new_lineage_n_windows {man['new_lineage_n_windows']} != "
                f"{len(new) + len(old)} total rows")
    seam_vintage = str((man.get("vintage_seam") or {}).get("input_vintage") or "")
    if not seam_vintage:
        return "manifest vintage_seam block lacks input_vintage"
    bad = [w["cutoff_date"] for w in new
           if w.get("input_vintage") != seam_vintage]
    if bad:
        return (f"new windows missing the seam input_vintage stamp "
                f"{seam_vintage!r}: {bad[:3]}")
    poisoned = [w["cutoff_date"] for w in old
                if w.get("input_vintage") == seam_vintage]
    if poisoned:
        return (f"existing windows stamped with the SEAM vintage — the seam "
                f"would be fiction: {poisoned[:3]}")
    cuts = [str(w["cutoff_date"])[:10] for w in new + old]
    if len(set(cuts)) != len(cuts):
        return "cutoff ladder has duplicates across the seam"
    if cuts != sorted(cuts):
        return ("cutoff ladder is not chronological with new windows BEFORE "
                "the existing ladder (the manifest's own root_rule)")
    return None


def _derive_oos_grid(cutoffs: list[str],
                     panel_dates: pd.DatetimeIndex) -> dict[str, list | None]:
    """``(cut, next_cut]`` over the caller panel's trading dates; the final
    window maps to ``None`` (no closing edge in the ladder — refused
    downstream, never invented)."""
    grid: dict[str, list | None] = {}
    for i, c in enumerate(cutoffs):
        if i + 1 == len(cutoffs):
            grid[c] = None
            continue
        lo, hi = pd.Timestamp(c), pd.Timestamp(cutoffs[i + 1])
        grid[c] = [d for d in panel_dates if lo < d <= hi]
    return grid


def _score_segment(*, seg_name: str, rows: list[dict], input_vintage: str | None,
                   vintage_note: str | None, recipe_id: str, bundle_dir: Path,
                   grid: dict, panel: pd.DataFrame, panel_dates: pd.Series,
                   labels_by_date: dict | None, label_horizon_bdays: int,
                   regime_by_date: dict | None = None,
                   min_windows: int, t0: float, budget: float,
                   factory_kw: dict, workdir: Path) -> dict:
    """One segment: admissibility (#95) then per-window scoring (#96 engine).

    Returns the segment stamp block, or raises TimeoutError to signal the
    budget refusal (converted by the caller into the lane-level unavailable).
    """
    folds = [{"cutoff_date": str(w["cutoff_date"])[:10],
              "artifact_path": str(Path(bundle_dir, str(w["artifact_path"]))),
              "artifact_sha256": w.get("artifact_sha256")}
             for w in rows]
    seg_man = workdir / f"segment_{seg_name}.json"
    seg_man.write_text(json.dumps({
        "recipe_id": recipe_id,
        "lineage_root_sha": lineage_root_sha(
            recipe_id, [str(f["artifact_sha256"]) for f in folds]),
        "folds": folds}))
    # Admissibility first-OOS mirrors Stage-1: first business day after the
    # cutoff, derived from the LADDER (never from the artifacts).
    first_oos = {f["cutoff_date"]:
                 pd.Timestamp(f["cutoff_date"]) + pd.offsets.BDay(1)
                 for f in folds}
    adm = evaluate_lineage(seg_man, recipe_id_key="recipe_id",
                           label_horizon_bdays=label_horizon_bdays,
                           first_oos_dates=first_oos,
                           min_admissible_windows=min_windows)

    outcomes: list[dict] = []
    frames: list[pd.DataFrame] = []
    if adm["lineage_verdict"] != "admissible":
        # The segment's identity or admissibility failed — a stamped refusal;
        # nothing in this segment is scored (fail-closed, never partial).
        outcomes = [{**w, "scoring": "skipped_segment_refused"}
                    for w in adm["windows"]]
    else:
        for w in adm["windows"]:
            cut = w["cutoff_date"]
            if w["admissibility"] != "admissible":
                outcomes.append({**w, "scoring": "skipped_inadmissible"})
                continue
            _budget_guard(
                t0=t0, budget=budget,
                where=f"before scoring window {cut} in segment {seg_name!r}, "
                      f"after {sum(1 for o in outcomes if o.get('scoring') == 'scored')} "
                      "scored windows")
            # None = derived grid found no closing edge (final ladder window);
            # a missing key (caller-grid case) is an EMPTY grid, not None.
            dates = grid.get(cut, [])
            if dates is None:
                outcomes.append({**w, "scoring": "scoring_error",
                                 "scoring_reason":
                                 "final ladder window has no closing edge; "
                                 "refused, never invented (supply an explicit "
                                 "oos_dates_by_cutoff grid to score it)"})
                continue
            if not dates:
                outcomes.append({**w, "scoring": "scoring_error",
                                 "scoring_reason":
                                 "no OOS dates for this window in the grid "
                                 "(derived (cutoff, next_cutoff] over the "
                                 "caller panel, or the caller-supplied grid)"})
                continue
            sub = panel.loc[panel_dates.isin(dates)]
            out = LS.score_lineage(lineage_manifest=seg_man,
                                   admissibility={**adm, "windows": [w]},
                                   panel=sub,
                                   oos_dates_by_cutoff={cut: dates},
                                   min_admissible_windows=1, **factory_kw)
            # Post-call enforcement (review round 2): the scoring call is the
            # long operation — a slow FINAL eligible call has no subsequent
            # pre-window check, so the deadline must bite HERE too.
            _budget_guard(
                t0=t0, budget=budget,
                where=f"after scoring window {cut} in segment {seg_name!r} "
                      "— the scoring call itself crossed the budget")
            outcomes.extend(out["windows"])
            if len(out["scores"]):
                frames.append(out["scores"])

    n_scored = sum(1 for o in outcomes if o.get("scoring") == "scored")
    scores = (pd.concat(frames, ignore_index=True) if frames
              else pd.DataFrame(columns=["date", "ticker", "score",
                                         "cutoff_date"]))
    statistics: dict = {
        "n_rows_scored": int(len(scores)),
        "n_dates_scored": int(scores["date"].nunique()) if len(scores) else 0,
    }
    if labels_by_date is not None and len(scores):
        # Per-regime split threaded through to the stamp (orch#805): the pooled
        # mean is a regime-mix artifact on this book, so a Stage-2 lane that
        # ranked candidates on it would rank them on that artifact. The regime
        # map is supplied by the CALLER, exactly like labels_by_date — this
        # module never derives a regime.
        # Passed as a KEYWORD and only when supplied, so the two-argument call
        # contract other callers (and test doubles) rely on is unchanged when
        # there is no regime map — adding a capability must not break the seam.
        extra = {"regime_by_date": regime_by_date} if regime_by_date is not None else {}
        statistics["label_summary"] = LS.summarize_lineage_scores(
            scores, labels_by_date, **extra)
    scoring_verdict = "scored" if n_scored >= min_windows else "refused"
    return {
        "segment": seg_name,
        "input_vintage": input_vintage,
        **({"vintage_note": vintage_note} if vintage_note else {}),
        "cutoff_first": folds[0]["cutoff_date"],
        "cutoff_last": folds[-1]["cutoff_date"],
        "n_windows": len(folds),
        "n_admissible": int(adm["n_admissible"]),
        "n_scored_windows": n_scored,
        "admissibility_verdict": adm["lineage_verdict"],
        "admissibility_reason": adm["reason"],
        "scoring_verdict": scoring_verdict,
        "scoring_reason": ("ok" if scoring_verdict == "scored" else
                           f"only {n_scored} scored windows < minimum "
                           f"{min_windows} for this segment"),
        "windows": outcomes,
        "statistics": statistics,
    }


def attempt_lineage_scoring_stamp(*, stage1: dict,
                                  extension_manifest_path: Path | str,
                                  expected_manifest_sha256: str | None,
                                  panel: pd.DataFrame,
                                  label_horizon_bdays: int,
                                  labels_by_date: dict | None = None,
                                  regime_by_date: dict | None = None,
                                  oos_dates_by_cutoff: dict | None = None,
                                  min_admissible_windows_per_segment: int =
                                  DEFAULT_MIN_ADMISSIBLE_WINDOWS,
                                  time_budget_seconds: float =
                                  DEFAULT_TIME_BUDGET_SECONDS,
                                  scorer_factory=None) -> dict:
    """The Stage-2 stamp block. NEVER raises; admission is never touched."""
    t0 = time.monotonic()
    try:
        # -- Stage-1 gate: stage 2 only ever extends an ADMITTED lineage -----
        s1 = stage1 or {}
        if s1.get("lineage_lane") != "stage1":
            return _unavailable("stage-1 lineage lane block absent or "
                                "unavailable — stage 2 scores nothing")
        if s1.get("lineage_admissibility") != "admissible":
            return _unavailable("stage-1 lineage is not admissible — stage 2 "
                                "scores nothing")
        # -- Content pin: the RUN_CLAIM binding, caller-supplied -------------
        if not expected_manifest_sha256:
            return _unavailable("no expected_manifest_sha256 content pin "
                                "supplied — the stamp must be content-bound "
                                "to the exact evidence it scores")
        mpath = Path(extension_manifest_path)
        if not mpath.is_file():
            return _unavailable(f"extension manifest missing: {mpath}")
        raw = mpath.read_bytes()
        actual_sha = hashlib.sha256(raw).hexdigest()
        if actual_sha != expected_manifest_sha256:
            return _unavailable(
                f"extension manifest content pin mismatch: sha256 "
                f"{actual_sha[:12]}… != expected "
                f"{str(expected_manifest_sha256)[:12]}…")
        man = json.loads(raw.decode("utf-8"))
        # -- Structure + seam integrity --------------------------------------
        reason = _structural_check(man)
        if reason:
            return _unavailable(reason)
        recipe_id = str(man.get("recipe_id") or "")
        if not recipe_id:
            return _unavailable("manifest lacks recipe_id")
        new, old = man["new_windows"], man["existing_windows"]
        # -- Identity: both roots recomputed from DECLARED shas --------------
        old_shas = [str(w.get("artifact_sha256")) for w in old]
        all_shas = [str(w.get("artifact_sha256")) for w in new] + old_shas
        root_old = lineage_root_sha(recipe_id, old_shas)
        root_full = lineage_root_sha(recipe_id, all_shas)
        if root_old != str(man.get("old_lineage_root_sha")):
            return _unavailable(
                f"old lineage root recomputed {root_old[:12]}… != claimed "
                f"{str(man.get('old_lineage_root_sha'))[:12]}…")
        if root_full != str(man.get("new_lineage_root_sha")):
            return _unavailable(
                f"full-ladder root recomputed {root_full[:12]}… != claimed "
                f"{str(man.get('new_lineage_root_sha'))[:12]}…")
        # -- Cross-lane binding: the extension must extend the ADMITTED root -
        if s1.get("lineage_root_sha") != root_old:
            return _unavailable(
                f"stage-1 admitted root {str(s1.get('lineage_root_sha'))[:12]}… "
                f"!= extension's old root {root_old[:12]}… — this bundle does "
                "not extend the admitted lineage")
        if str(s1.get("recipe_id")) != recipe_id:
            return _unavailable(
                f"stage-1 recipe_id {s1.get('recipe_id')!r} != extension "
                f"manifest recipe_id {recipe_id!r}")
        # -- OOS grid: ladder-derived, or the caller's explicit grid ---------
        cutoffs = [str(w["cutoff_date"])[:10] for w in new + old]
        if "date" not in panel.columns:
            return _unavailable("caller panel lacks a 'date' column")
        panel_dates = pd.to_datetime(panel["date"])
        if oos_dates_by_cutoff is not None:
            grid = {c: [pd.Timestamp(d) for d in ds]
                    for c, ds in oos_dates_by_cutoff.items()}
        else:
            uniq = pd.DatetimeIndex(sorted(panel_dates.unique()))
            grid = _derive_oos_grid(cutoffs, uniq)
        factory_kw = {"scorer_factory": (scorer_factory if scorer_factory
                                         is not None
                                         else gbdt_window_scorer_factory)}
        seam = man["vintage_seam"]
        with tempfile.TemporaryDirectory(prefix="lineage_stage2_") as td:
            workdir = Path(td)
            common = dict(recipe_id=recipe_id, bundle_dir=mpath.parent,
                          grid=grid, panel=panel, panel_dates=panel_dates,
                          labels_by_date=labels_by_date,
                          regime_by_date=regime_by_date,
                          label_horizon_bdays=label_horizon_bdays,
                          min_windows=min_admissible_windows_per_segment,
                          t0=t0, budget=float(time_budget_seconds),
                          factory_kw=factory_kw, workdir=workdir)
            pre = _score_segment(seg_name="pre_seam", rows=new,
                                 input_vintage=str(seam["input_vintage"]),
                                 vintage_note=None, **common)
            post = _score_segment(
                seg_name="post_seam", rows=old, input_vintage=None,
                vintage_note=("existing production ladder predates the "
                              "input rebuild; the manifest stamps no "
                              "input_vintage on these rows by design"),
                **common)
        # Successful-return boundary (review round 2): segment post-processing
        # (label summaries, frame concat) runs AFTER the last scoring call —
        # a stamp that leaves this function over budget is a refusal, always.
        _budget_guard(t0=t0, budget=float(time_budget_seconds),
                      where="at the successful-return boundary, after all "
                            "segment scoring")
        return {
            "lineage_stage2": "stage2",
            "candidate_lineage_used": True,
            "candidate_artifact_used": False,   # documented meaning per #94:
            # this lane never direct-scores the served booster
            "recipe_id": recipe_id,
            "extension_manifest_sha256": actual_sha,
            "lineage_root_sha_old": root_old,
            "lineage_root_sha_full": root_full,
            "stage1_lineage_root_match": True,
            "pooling_rule": POOLING_RULE,
            "vintage_seam": {
                "input_vintage": str(seam["input_vintage"]),
                "seam_boundary_cutoffs": [pre["cutoff_last"],
                                          post["cutoff_first"]],
                "evidence_golden_report_sha256":
                    seam.get("evidence_golden_report_sha256"),
                "golden_parity_max_abs_delta":
                    seam.get("golden_parity_max_abs_delta"),
                "pooling": "segment-only; this stamp emits NO cross-seam "
                           "pooled statistic",
            },
            "segments": {"pre_seam": pre, "post_seam": post},
            "n_windows": len(cutoffs),
            "n_scored_windows": (pre["n_scored_windows"]
                                 + post["n_scored_windows"]),
            "label_horizon_bdays": int(label_horizon_bdays),
            "time_budget_seconds": float(time_budget_seconds),
            "elapsed_seconds": round(time.monotonic() - t0, 3),
        }
    except TimeoutError as exc:
        return _unavailable(str(exc))
    except Exception as exc:  # noqa: BLE001 — the lane must never break the gate
        return _unavailable(str(exc))
