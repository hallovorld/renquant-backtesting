"""Modal-scaled walk-forward PatchTST re-score (GOAL-2 AC2/AC3).

This subpackage runs the walk-forward PatchTST training driver
(:mod:`renquant_backtesting.wf_gate.train_walkforward_patchtst`) on Modal cloud
GPUs, one fold per pod, IN PARALLEL. It produces a FRESH, provenance-stamped
per-fold corpus (``.pt`` + calibrators + a walk-forward manifest) suitable as
the 2nd expert for the GOAL-4 ensemble.

Repo-boundary rationale (see ``docs/progress/2026-07-21-modal-wf-patchtst-rescore.md``):

* **Model-training internals stay in renquant-model.** The Modal worker never
  reimplements training; it invokes ``python -m renquant_model_patchtst.hf_trainer``
  / ``renquant_model_patchtst.fit_calibrator`` exactly as the local driver does
  (the driver *is* the per-fold unit of work — the worker calls
  ``train_one_cutoff``).
* **The Modal orchestration lives in renquant-backtesting**, next to the rest of
  the WF-gate infrastructure (``wf_gate/``), because the walk-forward gate is
  backtesting's subject. It is NOT homed in renquant-orchestrator — orchestrator
  consumes backtesting, not the other way round.

The pieces:

* :mod:`.executor` (here) — the modal-free driver side: staging, dispatch,
  provenance, manifest assembly, and the CLI. ``modal`` is imported lazily.
* ``wf_patchtst_modal_app`` — a **standalone top-level module** (deliberately NOT
  under the ``renquant_backtesting`` package) holding the module-scope
  ``modal.App`` + the ``@app.function`` GPU worker. Modal's container entrypoint
  imports the worker's defining module at load time, so it must import with only
  ``os + modal``; a submodule of ``renquant_backtesting`` would drag the heavy
  package ``__init__`` (→ ``renquant_common``) and fail to load in-container. See
  that module's docstring. This two-part split mirrors the proven orchestrator
  ``cloud/`` sweep pattern without importing it.

Importing this package does NOT import ``modal`` — the Modal SDK is only imported
lazily by :mod:`.executor` at dispatch time. This keeps the provenance / recipe
helpers unit-testable with no cloud dependency.

    from renquant_backtesting.wf_gate.modal import executor   # driver side
    import wf_patchtst_modal_app                              # the Modal app
"""
from __future__ import annotations
