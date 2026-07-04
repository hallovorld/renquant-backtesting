"""Campaign B1 pins: the WF loader's verification IS the M6 dispatch.

RQ#444 F-2 / orchestrator#295 F2 / orchestrator#296 BT-1: this repo used
to carry a forked ``WalkForwardModelLoader`` with its own 12-char-prefix
fingerprint matcher and a venv-coupled bare ``model_content_sha256``
recompute. These tests pin the收编 (consolidation):

1. is-identity — the verification surface is IMPORTED from the pipeline
   dispatch module, never re-implemented (a re-fork breaks these).
2. incident-fixture regressions — a mismatched stamp still fails closed;
   a legacy-stamped artifact still passes via the dispatch legacy route.
3. the historical 12-char prefix acceptance survives ONLY behind the
   explicit ``accept_legacy_stamps`` migration-window flag (default ON);
   flag OFF retires it together with every versionless stamp.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import renquant_backtesting.walk_forward.loader as bt_loader
from renquant_backtesting.walk_forward.loader import WalkForwardModelLoader
from renquant_pipeline.kernel.panel_pipeline import fingerprint_dispatch
from renquant_pipeline.kernel.panel_pipeline.global_calibrator import (
    GlobalPanelCalibration,
)
from renquant_pipeline.kernel.walk_forward.loader import (
    WalkForwardModelLoader as PipelineWalkForwardModelLoader,
)


def _write_manifest(tmp_path, scorer_path, cal_path):
    p = tmp_path / "walkforward_manifest.json"
    p.write_text(json.dumps({
        "retrains": [{
            "cutoff_date": "2024-01-01T00:00:00",
            "trained_date": "2024-01-02T03:00:00",
            "artifact_uri": str(scorer_path),
            "calibrator_uri": str(cal_path),
        }],
    }))
    return p


def _write_calibrator(cal_path, metadata):
    GlobalPanelCalibration(
        prob_x=np.array([-1.0, 1.0]),
        prob_y=np.array([0.25, 0.75]),
        er_x=np.array([-1.0, 1.0]),
        er_y=np.array([-0.01, 0.01]),
        metadata=metadata,
    ).save(cal_path)


_LEGACY_STAMP = "sha256:" + "a1" * 32


def _write_stamped_scorer(tmp_path, stamp=_LEGACY_STAMP):
    """A versionless (legacy) stamped fold artifact."""
    scorer_path = tmp_path / "panel-ltr.json"
    scorer_path.write_text(json.dumps({
        "kind": "panel_ltr_xgboost",
        "feature_cols": ["f0"],
        "model_content_fingerprint": stamp,
    }))
    return scorer_path


class TestVerificationIsPipelineDispatch:
    """Is-identity: no local matcher fork can silently come back."""

    def test_loader_is_pipeline_loader_subclass(self):
        assert issubclass(WalkForwardModelLoader, PipelineWalkForwardModelLoader)

    def test_verification_methods_are_inherited_not_overridden(self):
        for name in (
            "_assert_calibrator_matches_entry",
            "_scorer_claim_for_entry",
            "_scorer_fingerprints_for_entry",
            "calibrator_as_of",
            "entry_as_of",
            "model_as_of",
        ):
            assert name not in vars(WalkForwardModelLoader), (
                f"{name} is overridden locally — the fourth fork; only the "
                "URI-resolution layer may live in renquant-backtesting"
            )

    def test_matcher_helpers_are_the_dispatch_functions(self):
        assert bt_loader._fingerprints_match is fingerprint_dispatch.fingerprints_match
        assert bt_loader._any_fingerprints_match is fingerprint_dispatch.any_fingerprints_match
        assert bt_loader._normalize_fingerprint is fingerprint_dispatch.normalize_fingerprint
        assert (
            bt_loader._scorer_claim_from_payload
            is fingerprint_dispatch.scorer_claim_from_payload
        )


class TestIncidentFixtureRegressions:
    """The 05-27/06-22/07-01 incident shapes, pinned on THIS repo's loader."""

    def test_mismatched_stamp_still_fails_closed(self, tmp_path):
        scorer_path = _write_stamped_scorer(tmp_path)
        cal_path = tmp_path / "cal.json"
        _write_calibrator(cal_path, {
            "scorer_model_content_fingerprint": "sha256:" + "0f" * 32,
        })
        loader = WalkForwardModelLoader(
            _write_manifest(tmp_path, scorer_path, cal_path),
        )
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            loader.calibrator_as_of("2024-01-15")

    def test_legacy_stamped_artifact_still_passes_via_dispatch(self, tmp_path):
        scorer_path = _write_stamped_scorer(tmp_path)
        cal_path = tmp_path / "cal.json"
        _write_calibrator(cal_path, {
            "scorer_model_content_fingerprint": _LEGACY_STAMP,
        })
        loader = WalkForwardModelLoader(
            _write_manifest(tmp_path, scorer_path, cal_path),
        )
        cal = loader.calibrator_as_of("2024-01-15")
        assert cal.metadata["scorer_model_content_fingerprint"] == _LEGACY_STAMP

    def test_corrupt_v1_stamp_fails_closed_at_fold_read(self, tmp_path):
        """A v1-stamped fold whose content does not reproduce its own stamp
        is corrupt regardless of any flag (dispatch ``verify()``)."""
        scorer_path = tmp_path / "panel-ltr.json"
        scorer_path.write_text(json.dumps({
            "kind": "panel_ltr_xgboost",
            "feature_cols": ["f0"],
            "booster_raw_json": "{\"learner\":{}}",
            "model_content_fingerprint": "sha256:" + "d0" * 32,
            "fingerprint_schema_version": 1,
        }))
        cal_path = tmp_path / "cal.json"
        _write_calibrator(cal_path, {
            "scorer_model_content_fingerprint": "sha256:" + "d0" * 32,
            "scorer_fingerprint_schema_version": 1,
        })
        loader = WalkForwardModelLoader(
            _write_manifest(tmp_path, scorer_path, cal_path),
        )
        with pytest.raises(ValueError):
            loader.calibrator_as_of("2024-01-15")


class TestPrefixAcceptanceIsFlagGoverned:
    """The fork's 12-char prefix acceptance now lives ONLY in the dispatch
    legacy route: ON during the migration window (default), retired by the
    explicit ``accept_legacy_stamps=False`` flip (M6 stage-2 step 4)."""

    def _prefix_fixture(self, tmp_path):
        scorer_path = _write_stamped_scorer(tmp_path)
        cal_path = tmp_path / "cal.json"
        # Historical short-sha declaration: a 12-char prefix of the stamp.
        _write_calibrator(cal_path, {
            "scorer_model_content_fingerprint": _LEGACY_STAMP[:len("sha256:") + 12],
        })
        return _write_manifest(tmp_path, scorer_path, cal_path)

    def test_prefix_accepted_while_flag_on(self, tmp_path):
        loader = WalkForwardModelLoader(self._prefix_fixture(tmp_path))
        cal = loader.calibrator_as_of("2024-01-15")
        assert cal is not None

    def test_flag_off_retires_versionless_stamps_and_prefixes(self, tmp_path):
        loader = WalkForwardModelLoader(
            self._prefix_fixture(tmp_path), accept_legacy_stamps=False,
        )
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            loader.calibrator_as_of("2024-01-15")
