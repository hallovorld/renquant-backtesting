from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from renquant_backtesting.wf_gate import runner


def test_manifest_sanity_date_map_is_manifest_driven(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLoader:
        def __init__(self, manifest_path: Path) -> None:
            self.manifest_path = manifest_path

        def has_walkforward_model(self) -> bool:
            return True

        def entry_as_of(self, d):
            d = pd.Timestamp(d)
            if d < pd.Timestamp("2024-01-10"):
                raise ValueError("pre-manifest")
            return SimpleNamespace(
                cutoff_date=pd.Timestamp("2023-12-01"),
                effective_train_cutoff_date=pd.Timestamp("2023-12-01"),
                lookahead_days=20,
                artifact_uri="artifacts/hf_patchtst_seed42.pt",
            )

    fake_loader_mod = SimpleNamespace(WalkForwardModelLoader=FakeLoader)
    monkeypatch.setitem(
        sys.modules,
        "renquant_backtesting.walk_forward.loader",
        fake_loader_mod,
    )
    monkeypatch.setattr(
        runner,
        "_manifest_uri_to_path",
        lambda manifest_path, uri: Path("/tmp") / Path(uri).name,
    )

    date_to_artifact, safe_dates, skipped = runner._manifest_sanity_date_map(
        [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-15")],
        Path("/tmp/wf_manifest.json"),
        20,
    )

    assert safe_dates == [pd.Timestamp("2024-01-15")]
    assert skipped == [pd.Timestamp("2024-01-05")]
    assert date_to_artifact[pd.Timestamp("2024-01-15")] == "/tmp/hf_patchtst_seed42.pt"


def test_manifest_sanity_date_map_fails_closed_on_lookahead_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLoader:
        def __init__(self, manifest_path: Path) -> None:
            self.manifest_path = manifest_path

        def has_walkforward_model(self) -> bool:
            return True

        def entry_as_of(self, d):
            return SimpleNamespace(
                cutoff_date=pd.Timestamp("2023-12-01"),
                effective_train_cutoff_date=pd.Timestamp("2023-12-01"),
                lookahead_days=60,
                artifact_uri="artifacts/hf_patchtst_seed42.pt",
            )

    fake_loader_mod = SimpleNamespace(WalkForwardModelLoader=FakeLoader)
    monkeypatch.setitem(
        sys.modules,
        "renquant_backtesting.walk_forward.loader",
        fake_loader_mod,
    )

    with pytest.raises(ValueError, match="lookahead mismatch"):
        runner._manifest_sanity_date_map(
            [pd.Timestamp("2024-02-15")],
            Path("/tmp/wf_manifest.json"),
            20,
        )
