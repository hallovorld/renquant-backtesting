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
  consumes backtesting, not the other way round. The two-file split
  (:mod:`.executor` = modal-free staging/dispatch/provenance;
  :mod:`.app` = module-scope Modal app that imports ``modal``) mirrors the proven
  orchestrator ``cloud/`` sweep pattern without importing it.

Importing this package does NOT import ``modal`` — the Modal SDK is only imported
lazily by :mod:`.executor` at dispatch time, and eagerly by :mod:`.app` (which is
only imported when a real/fake Modal is present). This keeps the provenance /
recipe helpers unit-testable with no cloud dependency.

Import the pieces directly:

    from renquant_backtesting.wf_gate.modal import executor   # driver side
    from renquant_backtesting.wf_gate.modal import app        # the Modal app

The submodules are intentionally NOT re-exported here so that
``python -m renquant_backtesting.wf_gate.modal.executor`` does not trip the
"module found in sys.modules after import of its package" runpy warning.
"""
from __future__ import annotations
