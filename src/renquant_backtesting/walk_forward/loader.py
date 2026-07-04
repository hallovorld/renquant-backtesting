"""Walk-forward model loader — backtesting URI resolution over the
pipeline-owned canonical loader.

Campaign B1 (RQ#444 F-2 / orchestrator#295 F2 / orchestrator#296 BT-1):
this module used to carry a FULL fork of the pipeline
``WalkForwardModelLoader`` — 266 drifted lines including a local
12-char-prefix fingerprint matcher and a venv-coupled bare
``model_content_sha256`` recompute lazy-imported from the pipeline's
``panel_scorer`` (semantics-follows-the-venv, the pipeline#160 hazard).
Three divergent verifiers of ONE WF-stamp contract (this fork, the
umbrella copy, the pipeline original) are the structural generator of the
2026-05-27 / 06-22 / 07-01 stamp-mismatch incident class.

Now the loader IS the pipeline loader. Verification routes through the M6
stage-2 dispatch module
(``renquant_pipeline.kernel.panel_pipeline.fingerprint_dispatch``):
schema-version dispatch, fail-closed ``verify()`` on v1 stamps, and the
legacy route preserved byte-for-byte during the migration window — the
historical 12-char-prefix acceptance survives ONLY inside that legacy
route and is retired fleet-wide by the explicit
``accept_legacy_stamps`` flag at M6 stage-2 step 4 (never by this repo
diverging again).

The ONLY backtesting-specific behavior kept local is manifest-URI
resolution (the strategy-dir inference layer), preserved verbatim below.

Contract (DO NOT CHANGE without P1 / P2 sync): see
``renquant_pipeline.kernel.walk_forward.loader`` — the single owner.
"""
from __future__ import annotations

from pathlib import Path

# Verification internals are IMPORTS ONLY from the pipeline-owned modules
# (M6 design §5 row 3: hash/match logic is never re-implemented locally).
# The historical private names stay importable from this module so no
# consumer re-forks them to keep an old import path alive.
from renquant_pipeline.kernel.panel_pipeline.fingerprint_dispatch import (  # noqa: F401
    IdentityClaim,
    any_fingerprints_match as _any_fingerprints_match,
    fingerprints_match as _fingerprints_match,
    normalize_fingerprint as _normalize_fingerprint,
    scorer_claim_from_payload as _scorer_claim_from_payload,
)
from renquant_pipeline.kernel.walk_forward.loader import (  # noqa: F401
    RetrainEntry,
    WalkForwardModelLoader as _PipelineWalkForwardModelLoader,
    _calibrator_claim,
    _calibrator_scorer_fingerprints,
    _optional_timestamp,
    _parse_entry,
    _resolve_manifest_path,
    _scorer_fingerprints_from_payload,
)

__all__ = ["RetrainEntry", "WalkForwardModelLoader"]


class WalkForwardModelLoader(_PipelineWalkForwardModelLoader):
    """Pipeline loader + the backtesting manifest-URI resolution layer.

    Everything else — manifest parsing, the leakage guards, point-in-time
    entry selection, scorer/calibrator loading, and the fail-closed
    scorer/calibrator fingerprint contract
    (``_assert_calibrator_matches_entry`` via the M6 fingerprint
    dispatch) — is inherited from the canonical implementation and MUST
    NOT be overridden here (that override would be the fourth fork).
    """

    def _resolve_uri(self, uri: str):
        """Resolve local relative manifest URIs (backtesting layer).

        Legacy manifests may store paths relative to the manifest folder.
        Production WF manifests store strategy-dir-relative paths such as
        ``artifacts/walkforward_v2/...`` while the manifest itself lives under
        ``artifacts/sim``. Prefer existing manifest-relative files, then avoid
        the ``artifacts/sim/artifacts/...`` double-resolution failure mode by
        resolving strategy artifact paths against the inferred strategy root.
        """
        if "://" in uri:
            return uri
        p = Path(uri)
        if p.is_absolute():
            return p
        candidate = self._manifest_path.parent / p
        if candidate.exists():
            return candidate
        strategy_dir = self._strategy_dir_from_manifest_path()
        if strategy_dir is not None:
            strategy_candidate = strategy_dir / p
            if p.parts[:1] == ("artifacts",) or strategy_candidate.exists():
                return strategy_candidate
        return candidate

    def _strategy_dir_from_manifest_path(self) -> Path | None:
        """Infer ``<strategy_dir>`` from ``<strategy_dir>/artifacts/sim/*.json``."""
        parent = self._manifest_path.parent
        if parent.name == "sim" and parent.parent.name == "artifacts":
            return parent.parent.parent
        return None
