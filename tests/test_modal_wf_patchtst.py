"""Unit tests for the Modal-scaled walk-forward PatchTST re-score.

No cloud calls: a fake ``modal`` module is injected into ``sys.modules`` before
the app module is (re-)imported — mirroring the orchestrator ``cloud/`` test
strategy. Covers: image-spec lockstep, gpu/timeout/retries decoration-time
baking, recipe-id determinism, the 43-fold target window, staged selection,
dispatch fan-out + artifact collection + manifest assembly, the provenance
envelope stamps (provenance_schema_version + recipe_id + effective_train_cutoff),
and the Modal-readiness precheck.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import importlib
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

from renquant_backtesting.wf_gate.modal import executor as ex

# The Modal worker is a STANDALONE top-level module (not under the
# renquant_backtesting package) so the container entrypoint can import it with
# only os + modal — see wf_patchtst_modal_app's docstring.
APP_MODULE = "wf_patchtst_modal_app"
ENV_VARS = (
    "RENQUANT_WF_MODAL_GPU",
    "RENQUANT_WF_MODAL_TIMEOUT_SECONDS",
    "RENQUANT_WF_MODAL_RETRIES",
)


# ── Fake Modal SDK ───────────────────────────────────────────────────────────
def _install_fake_modal(monkeypatch, *, map_results=None):
    """Inject a fake ``modal`` module; return a captured-state dict.

    Captures the ``@app.function`` decorator kwargs and the ``Image`` build
    inputs so tests can assert the real decoration-time values. When
    ``map_results`` is given, the decorated worker gets a ``.map()`` that yields
    those canned per-pod result strings and records the dispatched payloads.
    """
    fake = types.ModuleType("modal")
    captured: dict = {"dispatched": []}

    class _FakeImage:
        def pip_install(self, *a, **k):
            captured["image_pip_packages"] = list(a)
            return self

        def run_commands(self, *a, **k):
            captured["image_run_commands"] = list(a)
            return self

    class _FakeImageNS:
        @staticmethod
        def debian_slim(python_version=None):
            captured["image_python_version"] = python_version
            return _FakeImage()

    class _FakeVolume:
        @staticmethod
        def from_name(name, create_if_missing=False):
            captured["volume_name"] = name
            v = _FakeVolume()
            return v

        def commit(self):
            captured["volume_committed"] = True

        @contextmanager
        def batch_upload(self, force=False):
            captured["batch_force"] = force
            yield self

        def put_file(self, local, remote):
            captured.setdefault("uploaded", []).append((local, remote))

    class _MappedFn:
        def __init__(self, kwargs):
            captured["function_kwargs"] = kwargs
            self._modal_function_kwargs = kwargs

        def map(self, requests, order_outputs=None, return_exceptions=None,
                **extra):
            captured["dispatched"] = list(requests)
            return iter(map_results if map_results is not None else [])

    class _FakeApp:
        def __init__(self, name):
            captured["app_name"] = name

        def function(self, **kwargs):
            def deco(fn):
                return _MappedFn(kwargs)
            return deco

        @contextmanager
        def run(self):
            captured["app_ran"] = True
            yield self

    fake.App = _FakeApp
    fake.Volume = _FakeVolume
    fake.Image = _FakeImageNS
    monkeypatch.setitem(sys.modules, "modal", fake)
    # Force a fresh app import against this fake SDK (gpu/timeout/retries are
    # baked at import time; the executor's re-import guard otherwise trips).
    monkeypatch.delitem(sys.modules, APP_MODULE, raising=False)
    return captured


def _reimport_app(monkeypatch, *, gpu=None, timeout=None, retries=None):
    for var, val in (
        ("RENQUANT_WF_MODAL_GPU", gpu),
        ("RENQUANT_WF_MODAL_TIMEOUT_SECONDS", timeout),
        ("RENQUANT_WF_MODAL_RETRIES", retries),
    ):
        if val is None:
            monkeypatch.delenv(var, raising=False)
        else:
            monkeypatch.setenv(var, str(val))
    monkeypatch.delitem(sys.modules, APP_MODULE, raising=False)
    return importlib.import_module(APP_MODULE)


# ── Image spec / recipe / fold-window ────────────────────────────────────────
def test_image_spec_fingerprint_is_deterministic_sha256():
    fp = ex.image_spec_fingerprint()
    assert fp.startswith("sha256:") and len(fp) == len("sha256:") + 64
    assert fp == ex.image_spec_fingerprint()


def test_app_image_is_built_from_image_spec(monkeypatch):
    captured = _install_fake_modal(monkeypatch)
    _reimport_app(monkeypatch)
    assert captured["image_python_version"] == ex.IMAGE_SPEC["python_version"]
    assert captured["image_pip_packages"] == list(ex.IMAGE_SPEC["pip_packages"])
    # IMAGE_SPEC declares no run_commands; app.py must not add any.
    assert list(ex.IMAGE_SPEC["run_commands"]) == []
    assert "image_run_commands" not in captured


def test_gpu_timeout_retries_bake_into_decorator(monkeypatch):
    captured = _install_fake_modal(monkeypatch)
    mod = _reimport_app(monkeypatch, gpu="A10G", timeout=1234, retries=3)
    assert mod.WORKER_GPU == "A10G"
    assert mod.WORKER_TIMEOUT_SECONDS == 1234
    assert mod.WORKER_RETRIES == 3
    kw = captured["function_kwargs"]
    assert kw["gpu"] == "A10G"
    assert kw["timeout"] == 1234
    assert kw["retries"] == 3
    assert kw["volumes"]["/data"] is mod.data_volume


def test_cpu_gpu_means_no_gpu_kwarg(monkeypatch):
    captured = _install_fake_modal(monkeypatch)
    _reimport_app(monkeypatch, gpu="cpu")
    assert "gpu" not in captured["function_kwargs"]


def test_recipe_id_deterministic_and_hyperparam_sensitive():
    base = {k: 1 for k in ex.RECIPE_FIELDS}
    rid = ex.compute_recipe_id(base)
    assert rid.startswith("sha256:") and len(rid) == len("sha256:") + 16
    assert rid == ex.compute_recipe_id(dict(base))  # order-independent copy
    bumped = dict(base, epochs=2)
    assert ex.compute_recipe_id(bumped) != rid
    # A field NOT in RECIPE_FIELDS must not change the id.
    assert ex.compute_recipe_id(dict(base, device="cuda")) == rid


def test_target_window_is_43_folds():
    cutoffs = ex.compute_retrain_cutoffs("2023-10-02", "2026-03-02", 21)
    assert len(cutoffs) == 43
    assert cutoffs[0] == "2023-10-02"
    assert cutoffs[-1] == "2026-03-02"


def test_staged_selection_takes_recent_folds():
    cutoffs = ex.compute_retrain_cutoffs("2023-10-02", "2026-03-02", 21)
    staged = ex.select_staged_cutoffs(cutoffs, 8)
    assert staged == cutoffs[-8:]
    assert ex.select_staged_cutoffs(cutoffs, None) == cutoffs
    assert ex.select_staged_cutoffs(cutoffs, 999) == cutoffs


# ── Readiness precheck ───────────────────────────────────────────────────────
def test_modal_readiness_reports_missing_token(monkeypatch, tmp_path):
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    monkeypatch.setattr(ex.Path, "home", staticmethod(lambda: tmp_path))
    _install_fake_modal(monkeypatch)  # SDK importable
    report = ex.modal_readiness()
    assert report["sdk_importable"] is True
    assert report["token_present"] is False
    assert report["ready"] is False
    assert any("credentials" in m for m in report["missing"])


def test_modal_readiness_ok_with_env_token(monkeypatch, tmp_path):
    monkeypatch.setenv("MODAL_TOKEN_ID", "id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "secret")
    monkeypatch.setattr(ex.Path, "home", staticmethod(lambda: tmp_path))
    _install_fake_modal(monkeypatch)
    report = ex.modal_readiness()
    assert report["ready"] is True


# ── Plan build ───────────────────────────────────────────────────────────────
def _default_args(**over):
    ns = ex.parse_args([])
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_build_plan_full_and_staged():
    plan = ex.build_plan(_default_args())
    assert len(plan.cutoffs) == 43
    assert len(plan.fold_requests) == 43
    assert plan.recipe_id.startswith("sha256:")
    for req in plan.fold_requests:
        assert req["recipe_id"] == plan.recipe_id
        assert req["image_spec_sha256"] == ex.image_spec_fingerprint()
        assert req["container_repo_root"] == ex.CONTAINER_REPO_ROOT
    staged = ex.build_plan(_default_args(staged=5))
    assert len(staged.cutoffs) == 5
    assert staged.cutoffs == plan.cutoffs[-5:]


# ── Dispatch + collection + manifest + provenance (end to end, mocked) ───────
def _canned_fold_result(cutoff, trained, effective):
    pt_bytes = b"PYTORCH-FAKE-STATE-DICT"
    return json.dumps({
        "ok": True,
        "cutoff_date": cutoff,
        "recipe_id": "sha256:deadbeefdeadbeef",
        "worker_id": f"ta-{cutoff}",
        "code_image_id": "im-123",
        "device": "cuda",
        "elapsed_seconds": 42.0,
        "result_checksum": "sha256:abc123",
        "entry": {
            "cutoff_date": cutoff,
            "trained_date": trained,
            "artifact_uri": f"/data/backtesting/renquant_104/artifacts/"
                            f"walkforward_patchtst/{cutoff}/"
                            f"hf_patchtst_all_seed44_model.pt",
            "lookahead_days": 60,
            "calibrator_uri": f"/data/.../{cutoff}/hf_patchtst-calibration.json",
            "effective_train_cutoff_date": effective,
        },
        "artifacts": {
            "model_pt_b64gz": base64.b64encode(gzip.compress(pt_bytes)).decode(),
            "sidecar_json": json.dumps({"training_contract": {
                "trained_date": trained,
                "effective_train_cutoff_date": effective}}),
            "calibrator_json": json.dumps({"method": "platt", "a": 1.0, "b": 0.0}),
        },
    })


def test_dispatch_fans_out_and_collects(monkeypatch, tmp_path):
    # Two folds: trained_date >= cutoff (write_manifest validates this).
    results_json = [
        _canned_fold_result("2026-02-09", "2026-04-10", "2026-01-12"),
        _canned_fold_result("2026-03-02", "2026-05-01", "2026-02-02"),
    ]
    captured = _install_fake_modal(monkeypatch, map_results=results_json)

    plan = ex.build_plan(_default_args(staged=2))
    got = ex.dispatch_folds(plan, timeout_s=1800, retries=1,
                            volume_commit_id="sha256:vol01")
    assert captured["app_ran"] is True
    assert len(captured["dispatched"]) == 2
    # volume_commit_id threaded into every payload
    for p in captured["dispatched"]:
        assert json.loads(p)["volume_commit_id"] == "sha256:vol01"
    assert len(got) == 2 and all(r["ok"] for r in got)

    repo_root = tmp_path
    out = ex.collect_and_write(
        plan, got, repo_root=repo_root,
        code_heads={"renquant-model": "abc123"},
        staging={"volume_name": ex.VOLUME_NAME,
                 "volume_commit_id": "sha256:vol01"})
    assert out["n_folds"] == 2

    # Artifacts materialised on disk.
    art = (repo_root / "backtesting" / "renquant_104" / "artifacts"
           / "walkforward_patchtst")
    pts = list(art.rglob("*.pt"))
    assert len(pts) == 2
    assert pts[0].read_bytes() == b"PYTORCH-FAKE-STATE-DICT"
    assert list(art.rglob("*-calibration.json"))

    # Manifest via the reviewed writer.
    manifest = json.loads(Path(out["manifest"]).read_text())
    assert len(manifest["retrains"]) == 2
    assert all("effective_train_cutoff_date" in r for r in manifest["retrains"])

    # Provenance envelope carries the required GOAL-2 stamps.
    prov = out["provenance_obj"]
    assert prov["provenance_schema_version"] == ex.PROVENANCE_SCHEMA_VERSION
    assert prov["recipe_id"] == plan.recipe_id
    assert prov["modal"]["image_spec_sha256"] == ex.image_spec_fingerprint()
    assert prov["modal"]["gpu"] == plan.gpu
    assert prov["modal"]["volume_commit_id"] == "sha256:vol01"
    assert prov["n_folds_succeeded"] == 2
    for f in prov["folds"]:
        assert f["effective_train_cutoff_date"]
        assert f["cutoff_date"] in ("2026-02-09", "2026-03-02")
    # Per-pod facts recorded.
    assert prov["pod_facts"]["2026-03-02"]["worker_id"] == "ta-2026-03-02"


def test_dispatch_handles_partial_failure(monkeypatch, tmp_path):
    good = _canned_fold_result("2026-03-02", "2026-05-01", "2026-02-02")
    bad = json.dumps({"ok": False, "cutoff_date": "2026-02-09",
                      "error": "train_one_cutoff exit=1"})
    _install_fake_modal(monkeypatch, map_results=[good, bad])
    plan = ex.build_plan(_default_args(staged=2))
    got = ex.dispatch_folds(plan, timeout_s=60, retries=0,
                            volume_commit_id=None)
    out = ex.collect_and_write(
        plan, got, repo_root=tmp_path, code_heads={},
        staging={"volume_name": ex.VOLUME_NAME, "volume_commit_id": None})
    assert out["n_folds"] == 1  # only the good fold materialised
    prov = out["provenance_obj"]
    assert prov["n_folds_succeeded"] == 1
    assert len(prov["failed_folds"]) == 1
    assert prov["failed_folds"][0]["cutoff_date"] == "2026-02-09"


def test_dispatch_map_exception_is_captured(monkeypatch, tmp_path):
    _install_fake_modal(monkeypatch,
                        map_results=[RuntimeError("pod OOM")])
    plan = ex.build_plan(_default_args(staged=1))
    got = ex.dispatch_folds(plan, timeout_s=60, retries=0, volume_commit_id=None)
    assert len(got) == 1 and got[0]["ok"] is False
    assert "pod OOM" in got[0]["error"]


# ── Fresh-driver staleness guard ─────────────────────────────────────────────
def _write_driver(bundle_dir, body):
    drv = (bundle_dir / "renquant-backtesting" / "src" / "renquant_backtesting"
           / "wf_gate" / "train_walkforward_patchtst.py")
    drv.parent.mkdir(parents=True, exist_ok=True)
    drv.write_text(body)
    return drv


def test_assert_fresh_driver_passes_on_module_invocation(tmp_path):
    _write_driver(tmp_path,
                  'TRAIN_MODULE = "renquant_model_patchtst.hf_trainer"\n')
    ex._assert_fresh_driver(tmp_path)  # must not raise


def test_assert_fresh_driver_rejects_stale_script_driver(tmp_path):
    _write_driver(tmp_path,
                  'TRAIN_SCRIPT = REPO_ROOT / "scripts" / "patchtst_hf.py"\n')
    with pytest.raises(RuntimeError, match="STALE"):
        ex._assert_fresh_driver(tmp_path)


def test_assert_fresh_driver_rejects_missing_driver(tmp_path):
    with pytest.raises(RuntimeError, match="missing WF driver"):
        ex._assert_fresh_driver(tmp_path)


# ── CLI plan-only path ───────────────────────────────────────────────────────
def test_cli_dry_run_makes_no_cloud_calls(monkeypatch, capsys):
    # No fake modal installed: a dry run must not import modal at all.
    monkeypatch.delitem(sys.modules, "modal", raising=False)
    rc = ex.main(["--start-date", "2023-10-02", "--end-date", "2026-03-02",
                  "--cadence-days", "21", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "recipe_id" in out
    assert "43" in out
    assert "modal" not in sys.modules  # never imported on the dry-run path
