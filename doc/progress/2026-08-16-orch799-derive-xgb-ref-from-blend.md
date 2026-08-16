# orch#799: derive an xgb reference from a blend prod's component[0]

STATUS:    feat — implements the backtesting half of the approved orch#799 design
           (renquant-orchestrator #982). Paired with the umbrella weekly_wf_promote.sh
           wiring PR. Deploy order: pin this subrepo first, then the umbrella.

WHAT:      `wf_gate/wf_config_builder.py`: `derive_xgb_reference_from_blend(blend_config,
           *, strategy_dir, component_index=0)` + helper `_load_artifact_declared_kind`.
           Given a kind=blend prod config, returns an xgb-shaped reference built from
           `components[component_index]` (its artifact + config-fingerprint pin), with
           the blend structure stripped. Fails closed (ValueError) unless the component
           artifact's DECLARED kind normalizes to xgb.

WHY/DIR:   Operator-directed ("解决所有问题" 2026-08-16). The 2026-08-04 z-blend switch
           made the pinned primary kind=blend, so `_find_gbdt_config` finds no top-level
           kind=xgb reference and the weekly promote gate fails closed (exit 2) — the
           root of the wf-promote/retrain-panel104/silent-refusal alarm cluster, which
           fired again today. Design #982 (approved+merged) chose Option A: derive the
           xgb reference from component[0] (the production xgb scorer), keeping the blend.

EVIDENCE:
  artifact:      `wf_gate/wf_config_builder.py` (the new function + helper) +
                 `tests/wf_gate/test_wf_config_builder.py` (6 new tests) + this doc.
  prod or exp:   neither — pure library function + unit tests; no live/production write,
                 no gate run. Wired to the runtime only by the paired umbrella PR, then
                 an operator-gated deploy.
  existing data: [VERIFIED] pinned strategy_config.json kind=blend; strategy_config.shadow.json
                 kind=hf_patchtst (no pinned xgb reference → the freeze); blend
                 component[0] artifact `artifacts/prod/panel-ltr.alpha158_fund.json`
                 DECLARES kind=panel_ltr_xgboost (→ xgb) with fingerprint
                 sha256:f8fb2259b2bf1537 — so the derivation is adversarial-safe.
  best-known?:   yes — behaviour-invariance is PROVEN (12 pre-existing builder/parity tests
                 pass unchanged); the parity guard's intent is preserved (kind read from
                 the genuine artifact, never mutated); 4 fail-closed paths tested. Honest
                 scope: validates the xgb leg's recipe vs the kind-matched manifest; does
                 NOT gate the served z-sum (blend-level gating deferred per the design).
  scope:         "adds a pure derivation function so a blend prod can supply the xgb
                 reference the WF gate requires. Changes NO gate threshold, does NOT run
                 the gate, does NOT touch production. Inert until the paired umbrella
                 weekly_wf_promote.sh wiring calls it, then operator-gated deploy."

TESTS:     `pytest tests/wf_gate/test_wf_config_builder.py` → 18 passed (12 pre-existing
           unchanged + 6 new: xgb-shaped output, parity-kind match, non-xgb component
           refused, missing artifact refused, non-blend refused, no-components refused).

NEXT:      codex review → merge → pin from the umbrella alongside the paired
           weekly_wf_promote.sh PR → operator-gated live-tree deploy. Deferred (named in
           #982): real blend-level gating; the gate's booster-blindness.
