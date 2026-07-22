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
def _canned_sidecar(cutoff, trained, effective, *, sidecar_cutoff=None):
    """A metadata sidecar faithful to the renquant-model hf_trainer contract:
    ``training_contract`` carries the requested cutoff (``train_cutoff_date``),
    ``trained_date`` + ``effective_train_cutoff_date``, plus provenance/recipe
    (``dataset`` + ``hyperparameters``)."""
    return json.dumps({"training_contract": {
        "contract_version": 1,
        "trained_date": trained,
        "effective_train_cutoff_date": effective,
        "train_cutoff_date": sidecar_cutoff if sidecar_cutoff is not None else cutoff,
        "dataset": "data/transformer_v4_wl200_clean.parquet",
        "hyperparameters": {"seq_len": 32, "epochs": 5, "lr": 3e-4},
    }})


def _canned_fold_result(cutoff, trained, effective, *, with_model=True,
                        with_sidecar=True, with_calibrator=True,
                        sidecar_cutoff=None):
    """A worker ``ok`` result. Flags drop individual payloads / corrupt the
    sidecar cutoff so tests can exercise the fail-closed promotion gate."""
    pt_bytes = b"PYTORCH-FAKE-STATE-DICT"
    entry = {
        "cutoff_date": cutoff,
        "trained_date": trained,
        "artifact_uri": f"/data/backtesting/renquant_104/artifacts/"
                        f"walkforward_patchtst/{cutoff}/"
                        f"hf_patchtst_all_seed44_model.pt",
        "lookahead_days": 60,
        "effective_train_cutoff_date": effective,
    }
    if with_calibrator:
        entry["calibrator_uri"] = \
            f"/data/.../{cutoff}/hf_patchtst-calibration.json"
    artifacts = {}
    if with_model:
        artifacts["model_pt_b64gz"] = base64.b64encode(
            gzip.compress(pt_bytes)).decode()
    if with_sidecar:
        artifacts["sidecar_json"] = _canned_sidecar(
            cutoff, trained, effective, sidecar_cutoff=sidecar_cutoff)
    if with_calibrator:
        artifacts["calibrator_json"] = json.dumps(
            {"method": "platt", "a": 1.0, "b": 0.0})
    return json.dumps({
        "ok": True,
        "cutoff_date": cutoff,
        "recipe_id": "sha256:deadbeefdeadbeef",
        "worker_id": f"ta-{cutoff}",
        "code_image_id": "im-123",
        "device": "cuda",
        "elapsed_seconds": 42.0,
        "result_checksum": "sha256:abc123",
        "entry": entry,
        "artifacts": artifacts,
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
                 "volume_commit_id": "sha256:vol01",
                 "data_digests": {"/data/panel.parquet": "sha256:d00d"}})
    assert out["n_folds"] == 2

    # Artifacts materialised on disk — under the quarantined run namespace.
    art_root = repo_root / "backtesting" / "renquant_104" / "artifacts"
    pts = list(art_root.rglob("*.pt"))
    assert len(pts) == 2
    assert pts[0].read_bytes() == b"PYTORCH-FAKE-STATE-DICT"
    assert list(art_root.rglob("*-calibration.json"))
    # Nothing lands at the canonical serving location.
    assert not (art_root / ex.CANONICAL_SERVING_MANIFEST).exists()
    assert all(ex.RUN_NAMESPACE_ROOT in str(p) for p in pts)

    # Manifest via the reviewed writer, in the run namespace.
    assert ex.RUN_NAMESPACE_ROOT in out["manifest"]
    assert plan.run_id in out["manifest"]
    manifest = json.loads(Path(out["manifest"]).read_text())
    assert len(manifest["retrains"]) == 2
    assert all("effective_train_cutoff_date" in r for r in manifest["retrains"])

    # Provenance envelope carries the required GOAL-2 stamps.
    prov = out["provenance_obj"]
    assert prov["provenance_schema_version"] == ex.PROVENANCE_SCHEMA_VERSION
    assert prov["recipe_id"] == plan.recipe_id
    assert prov["run_id"] == plan.run_id
    assert prov["modal"]["image_spec_sha256"] == ex.image_spec_fingerprint()
    assert prov["modal"]["gpu"] == plan.gpu
    assert prov["modal"]["volume_commit_id"] == "sha256:vol01"
    assert prov["modal"]["data_digests"] == {"/data/panel.parquet": "sha256:d00d"}
    assert prov["modal"]["resolved_image_ids"] == ["im-123"]
    assert prov["n_folds_succeeded"] == 2
    # All requested folds succeeded → promotable (still a separate reviewed step).
    assert prov["promotion_ready"] is True
    assert prov["quarantined"] is False
    assert out["promotion_ready"] is True
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
    # A partial corpus is quarantined, NOT promotable (codex #76 blocker 3).
    assert prov["promotion_ready"] is False
    assert prov["quarantined"] is True
    assert out["promotion_ready"] is False


def test_dispatch_map_exception_is_captured(monkeypatch, tmp_path):
    _install_fake_modal(monkeypatch,
                        map_results=[RuntimeError("pod OOM")])
    plan = ex.build_plan(_default_args(staged=1))
    got = ex.dispatch_folds(plan, timeout_s=60, retries=0, volume_commit_id=None)
    assert len(got) == 1 and got[0]["ok"] is False
    assert "pod OOM" in got[0]["error"]


# ── Fail-closed promotion gate (codex #76 CR) ────────────────────────────────
def _collect_single(monkeypatch, tmp_path, *canned, skip_calibrators=False):
    """Dispatch the given canned fold(s) and collect them under ``tmp_path``.

    Returns the ``collect_and_write`` output dict (carries provenance_obj).
    """
    _install_fake_modal(monkeypatch, map_results=list(canned))
    plan = ex.build_plan(_default_args(staged=len(canned),
                                       skip_calibrators=skip_calibrators))
    got = ex.dispatch_folds(plan, timeout_s=60, retries=0, volume_commit_id=None)
    return ex.collect_and_write(
        plan, got, repo_root=tmp_path, code_heads={},
        staging={"volume_name": ex.VOLUME_NAME, "volume_commit_id": None})


def test_all_payloads_valid_is_promotion_ready(monkeypatch, tmp_path):
    # Happy path: every requested fold has a non-empty model + valid sidecar
    # (cutoff agrees, provenance/recipe present) + a valid calibrator.
    out = _collect_single(
        monkeypatch, tmp_path,
        _canned_fold_result("2026-02-09", "2026-04-10", "2026-01-12"),
        _canned_fold_result("2026-03-02", "2026-05-01", "2026-02-02"))
    prov = out["provenance_obj"]
    assert out["promotion_ready"] is True
    assert prov["promotion_ready"] is True
    assert prov["quarantined"] is False
    assert prov["n_folds_promotable"] == 2
    assert prov["promotion_gate"]["quarantine_reasons"] == []
    assert all(f["promotable"] for f in prov["folds"])


def test_missing_model_pt_is_quarantined(monkeypatch, tmp_path):
    # Worker reports ok but omits the model .pt blob → nothing materialises.
    out = _collect_single(
        monkeypatch, tmp_path,
        _canned_fold_result("2026-03-02", "2026-05-01", "2026-02-02",
                            with_model=False))
    prov = out["provenance_obj"]
    assert out["promotion_ready"] is False
    assert prov["promotion_ready"] is False
    assert prov["quarantined"] is True
    assert prov["n_folds_promotable"] == 0
    assert "model_pt_missing" in prov["promotion_gate"]["quarantine_reasons"]
    # No phantom .pt referenced in a manifest.
    assert out["n_folds"] == 0


def test_missing_sidecar_is_quarantined(monkeypatch, tmp_path):
    out = _collect_single(
        monkeypatch, tmp_path,
        _canned_fold_result("2026-03-02", "2026-05-01", "2026-02-02",
                            with_sidecar=False))
    prov = out["provenance_obj"]
    assert out["promotion_ready"] is False
    assert prov["quarantined"] is True
    assert "sidecar_missing" in prov["promotion_gate"]["quarantine_reasons"]
    # Model still materialised (quarantined) → manifest still written.
    assert out["n_folds"] == 1


def test_wrong_cutoff_sidecar_is_quarantined(monkeypatch, tmp_path):
    # Sidecar's train_cutoff_date does NOT agree with the requested fold.
    out = _collect_single(
        monkeypatch, tmp_path,
        _canned_fold_result("2026-03-02", "2026-05-01", "2026-02-02",
                            sidecar_cutoff="2020-01-01"))
    prov = out["provenance_obj"]
    assert out["promotion_ready"] is False
    assert prov["quarantined"] is True
    reasons = prov["promotion_gate"]["quarantine_reasons"]
    assert any(r.startswith("sidecar_cutoff_mismatch") for r in reasons)
    assert out["n_folds"] == 1  # model materialised, manifest still written


def test_missing_calibrator_is_quarantined(monkeypatch, tmp_path):
    out = _collect_single(
        monkeypatch, tmp_path,
        _canned_fold_result("2026-03-02", "2026-05-01", "2026-02-02",
                            with_calibrator=False))
    prov = out["provenance_obj"]
    assert out["promotion_ready"] is False
    assert prov["quarantined"] is True
    assert "calibrator_missing" in prov["promotion_gate"]["quarantine_reasons"]
    assert out["n_folds"] == 1  # model materialised, manifest still written


def test_skip_calibrators_never_promotable(monkeypatch, tmp_path):
    # All models materialise, but a --skip-calibrators run is diagnostic by
    # definition → ALWAYS quarantined, never promotion_ready.
    out = _collect_single(
        monkeypatch, tmp_path,
        _canned_fold_result("2026-02-09", "2026-04-10", "2026-01-12",
                            with_calibrator=False),
        _canned_fold_result("2026-03-02", "2026-05-01", "2026-02-02",
                            with_calibrator=False),
        skip_calibrators=True)
    prov = out["provenance_obj"]
    assert out["promotion_ready"] is False
    assert prov["promotion_ready"] is False
    assert prov["quarantined"] is True
    assert prov["promotion_gate"]["skip_calibrators"] is True
    assert "skip_calibrators_diagnostic" in \
        prov["promotion_gate"]["quarantine_reasons"]
    # Models + manifest still written (quarantined), just not promotable.
    assert out["n_folds"] == 2
    assert ex.RUN_NAMESPACE_ROOT in out["manifest"]


def test_validate_fold_promotable_unit(tmp_path):
    # Direct unit coverage of the gate helper against on-disk artifacts.
    model = tmp_path / "hf_patchtst_all_seed44_model.pt"
    model.write_bytes(b"MODEL")
    (tmp_path / (model.name + ".metadata.json")).write_text(
        _canned_sidecar("2026-03-02", "2026-05-01", "2026-02-02"))
    cal = tmp_path / "hf_patchtst-calibration.json"
    cal.write_text(json.dumps({"method": "platt", "a": 1.0}))
    entry = {"cutoff_date": "2026-03-02", "artifact_uri": str(model),
             "calibrator_uri": str(cal)}
    ok, reasons = ex.validate_fold_promotable(entry, skip_calibrators=False)
    assert ok is True and reasons == []
    # Empty model file → not promotable.
    model.write_bytes(b"")
    ok, reasons = ex.validate_fold_promotable(entry, skip_calibrators=False)
    assert ok is False and "model_pt_empty" in reasons


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


# ── Single-pinned-assembly bundling (codex #76 blocker 1 + strategy config) ──
import subprocess as _sp  # noqa: E402


def _git(*args, cwd):
    _sp.run(["git", *args], cwd=str(cwd), check=True,
            capture_output=True, text=True)


def _make_assembly(root, *, repos=None, with_config=True, fresh_driver=True):
    """Build a fake pinned assembly of git checkouts; return {repo: head_sha}."""
    repos = list(repos if repos is not None else ex.BUNDLE_REPOS)
    heads = {}
    for repo in repos:
        checkout = root / repo
        src = checkout / "src"
        src.mkdir(parents=True)
        (src / "__init__.py").write_text("")
        if repo == "renquant-backtesting":
            drv = (src / "renquant_backtesting" / "wf_gate"
                   / "train_walkforward_patchtst.py")
            drv.parent.mkdir(parents=True)
            drv.write_text(
                'TRAIN_MODULE = "renquant_model_patchtst.hf_trainer"\n'
                if fresh_driver else
                'TRAIN_SCRIPT = REPO_ROOT / "scripts" / "patchtst_hf.py"\n')
        if repo == "renquant-strategy-104" and with_config:
            cfg = checkout / "configs" / "strategy_config.json"
            cfg.parent.mkdir(parents=True)
            cfg.write_text("{}")
        _git("init", "-q", cwd=checkout)
        _git("config", "user.email", "t@t", cwd=checkout)
        _git("config", "user.name", "t", cwd=checkout)
        _git("add", "-A", cwd=checkout)
        _git("commit", "-qm", "init", cwd=checkout)
        heads[repo] = _sp.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip()
    return heads


def test_bundle_code_stages_all_and_returns_heads(tmp_path):
    root = tmp_path / "assembly"
    expected = _make_assembly(root)
    bundle = tmp_path / "bundle"
    heads = ex.bundle_code(bundle, root)
    assert heads == expected
    # Every required repo's src is staged...
    for repo in ex.BUNDLE_REPOS:
        assert (bundle / repo / "src" / "__init__.py").exists()
    # ...and the trainer's strategy config rides along (EXTRA_BUNDLE_SUBDIRS).
    assert (bundle / "renquant-strategy-104" / "configs"
            / "strategy_config.json").exists()


def test_bundle_code_fails_closed_on_missing_repo(tmp_path):
    root = tmp_path / "assembly"
    _make_assembly(root, repos=[r for r in ex.BUNDLE_REPOS
                                if r != "renquant-pipeline"])
    with pytest.raises(RuntimeError, match="missing required"):
        ex.bundle_code(tmp_path / "bundle", root)


def test_bundle_code_no_home_fallback_signature():
    # A single explicit root only — no list of candidate roots (which is how the
    # ~/git/github fallback used to sneak an arbitrary checkout in).
    import inspect
    params = inspect.signature(ex.bundle_code).parameters
    assert "code_root" in params and "code_roots" not in params


def test_bundle_code_rejects_lock_drift(tmp_path):
    root = tmp_path / "assembly"
    heads = _make_assembly(root)
    lock = dict(heads)
    lock["renquant-model"] = "0" * 40  # a stale/wrong pin
    with pytest.raises(RuntimeError, match="drift"):
        ex.bundle_code(tmp_path / "bundle", root, assembly_lock=lock)


def test_bundle_code_accepts_matching_lock(tmp_path):
    root = tmp_path / "assembly"
    heads = _make_assembly(root)
    # Exact match → no raise.
    ex.bundle_code(tmp_path / "bundle", root, assembly_lock=dict(heads))


def test_bundle_code_missing_strategy_config_fails_closed(tmp_path):
    root = tmp_path / "assembly"
    _make_assembly(root, with_config=False)
    with pytest.raises(RuntimeError, match="strategy_config.json"):
        ex.bundle_code(tmp_path / "bundle", root)


def test_assert_strategy_config_passes_when_present(tmp_path):
    cfg = tmp_path / "renquant-strategy-104" / "configs" / "strategy_config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{}")
    ex._assert_strategy_config(tmp_path)  # must not raise


def test_assert_strategy_config_rejects_missing(tmp_path):
    with pytest.raises(RuntimeError, match="strategy_config.json"):
        ex._assert_strategy_config(tmp_path)


# ── Content-addressed Volume digest (codex #76 blocker 2) ────────────────────
def _stage_two_files(monkeypatch, tmp_path, dataset_bytes, rawlabel_bytes):
    _install_fake_modal(monkeypatch)  # fake Volume batch_upload
    bundle = tmp_path / "bundle"
    (bundle / "renquant-backtesting" / "src").mkdir(parents=True)
    (bundle / "renquant-backtesting" / "src" / "x.py").write_bytes(b"CODE")
    ds = tmp_path / "ds.parquet"
    rl = tmp_path / "rl.parquet"
    ds.write_bytes(dataset_bytes)
    rl.write_bytes(rawlabel_bytes)
    plan = ex.build_plan(_default_args(staged=1, dataset="data/ds.parquet",
                                       raw_label_panel="data/rl.parquet"))
    return ex.stage_inputs_to_volume(
        plan, bundle_dir=bundle, dataset_path=ds, raw_label_path=rl)


def test_volume_commit_hashes_content_not_size(monkeypatch, tmp_path):
    a = _stage_two_files(monkeypatch, tmp_path / "a", b"AAAA", b"BBBB")
    # Same SIZES, different CONTENT → the commit id (and data digests) must differ.
    b = _stage_two_files(monkeypatch, tmp_path / "b", b"XXXX", b"YYYY")
    assert a["volume_commit_id"] != b["volume_commit_id"]
    assert a["data_digests"]["/data/ds.parquet"] != \
        b["data_digests"]["/data/ds.parquet"]
    # Both leakage-relevant panels get an explicit content digest.
    assert set(a["data_digests"]) == {"/data/ds.parquet", "/data/rl.parquet"}
    assert all(v.startswith("sha256:") for v in a["data_digests"].values())


# ── Run namespace + canonical-manifest refusal (codex #76 blocker 3) ─────────
def test_build_plan_quarantines_under_run_namespace():
    plan = ex.build_plan(_default_args(staged=2))
    assert plan.run_id.startswith("wf-pt-")
    assert plan.artifact_root.startswith(ex.RUN_NAMESPACE_ROOT + "/")
    assert plan.run_id in plan.artifact_root
    fixed = ex.build_plan(_default_args(staged=2, run_id="my-run"))
    assert fixed.run_id == "my-run"
    assert fixed.artifact_root == f"{ex.RUN_NAMESPACE_ROOT}/my-run"


def test_collect_refuses_canonical_serving_manifest(tmp_path):
    plan = ex.build_plan(_default_args(staged=1))
    canonical = (tmp_path / "backtesting" / "renquant_104" / "artifacts"
                 / ex.CANONICAL_SERVING_MANIFEST)
    plan.manifest_output = str(canonical)
    with pytest.raises(RuntimeError, match="canonical serving manifest"):
        ex.collect_and_write(
            plan, [], repo_root=tmp_path, code_heads={},
            staging={"volume_name": ex.VOLUME_NAME, "volume_commit_id": None})


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
