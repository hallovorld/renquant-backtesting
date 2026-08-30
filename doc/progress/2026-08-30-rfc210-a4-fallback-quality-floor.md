# RFC#210 A4 — the freshness fallback requires the genuine_ic ≥ 0.02 quality floor; the ratchet is removed

**Date:** 2026-08-30 · `renquant-backtesting` · branch `fix/freshness-fallback-quality-floor`
**Governance:** RFC#210 Amendment A4, orchestrator design doc
`doc/design/2026-06-30-model-freshness-governance.md` (orchestrator PR opening in
the same batch, branch `design/rfc210-amendment-honest-fallback`).

STATUS:    policy change in `wf_gate/freshness_fallback.py` + tests (52);
           consumer contract (exit codes, `decision`/`refused_on`/`checks`,
           the stamp keys `fallback_pair_promote.py` reads) UNCHANGED.
WHAT:      check 4 is now the QUALITY FLOOR — stamped
           `sanity_placebo_genuine_ic` finite and `>= FALLBACK_GENUINE_IC_FLOOR`,
           which IS `runner.PLACEBO_GENUINE_IC_MARGIN` (0.02, the §5.2 bar; one
           constant, one place, not a second copy). Check 5 is the §4.3.1
           FAILURE-CLASS filter: every failing sub-verdict in the candidate's
           stamp must be in the enumerated infra-only list; any substance
           class or anything unclassified is fail-closed regardless of
           genuine_ic. The ratchet against the served fallback's
           `fallback_genuine_ic` is REMOVED; the served value is REPORTED in
           the verdict (`served_fallback_genuine_ic`, `served_promotion_basis`)
           and never compared.
WHY:       [VERIFIED 2026-08-30, read-only] the served prod artifact
           (trained 08-02) was fallback-promoted on 2026-08-04 with
           `fallback_genuine_ic=+0.00289` — a Fix-3 placebo-floor failure that
           A3.2 classifies as QUALITY/SUBSTANCE, fail-closed. Its stamp also
           carries ΔSharpe −0.479 vs SPY (Fix-4), regime sanity IC FAIL, and
           trade monotonicity FAIL. Later candidates (08-18..23) scored
           +0.0006..+0.0000 and were refused by `ratchet_up_only`. A
           comparison against a model that itself failed the floor is not a
           quality predicate.

## Verdict reasons (consumer-visible text; exit codes unchanged)

| check | refuse `why` | pass `why` |
|---|---|---|
| `quality_floor` | `quality_floor_not_met(genuine_ic=+0.0006 < 0.02)` | `quality_floor_met` |
| `failure_classes` | `substance_failure_fail_closed(<class>[,<class>...])` | `infra_only_ok` |

Removed check names: `genuine_ic_nonnegative`, `ratchet_up_only`,
`ratchet_prior_readable` (a negative genuine_ic now refuses on
`quality_floor`; `genuine_ic_present` is unchanged).

## Failure classes (§4.3.1) — how the stamp is read

The runner stamps no failure class; it stamps each sub-verdict and composes
`passed` from them (`runner._compute_overall_pass`). `classify_gate_failures`
reads those same fields back and tags each failure:

| class | source field | kind |
|---|---|---|
| `placebo_ceiling` | `sanity_placebo_absolute_rule_pass is False` | **infra** — the ONLY bypassable class; its §4.3.1 predicate is the difference test = check 4 |
| `wf_benchmark_economics` (Fix-4) | `wf_reason` starts with `FAIL` | substance |
| `wf_sim_execution` | `wf_reason` contains `sim cuts failed` | substance (WF economics never measured; §4.3.3's OOS floor has no implemented predicate) |
| `leakage_shuffled_label` | `abs(sanity_shuffled_ic) >= runner.SHUF_IC_MAX` | substance |
| `regime_sanity_ic` | `sanity_regime_ic.passed is False` | substance |
| `trade_contract` / `trade_monotonicity` / `alpha_economics` / `qp_contract` | sub-verdict `passed` is not `True` | substance |
| `config_parity` | `passed` present and not `True` (runner defaults a MISSING key to pass) | substance (Fix-2 is "once reconfirmed" in §4.3.1 — not reconfirmed) |
| `recipe_mismatch` | neither `candidate_artifact_used` nor `recipe_validated` is `True` | substance |
| `diagnostic_only` | `diagnostic_only` / non-empty `skipped_required_gates` | substance |
| `wf_unclassified` / `sanity_unclassified` / `unclassified` | nothing attributable | substance (fail-closed) |

The other infra classes §4.3.1 names (phase/timeout, path-not-found) abort the
runner before `wf_gate_metadata` is written, so they never reach the fallback
as a `passed=False` stamp (refused earlier on `staging_gate_stamp`).

## Consumer-visible changes (umbrella `scripts/weekly_wf_promote.sh` Step 4b, read-only audit)

- Contract kept: exit 0 ⇔ `FALLBACK_PROMOTE`, exit 1 ⇔ `REFUSE`; JSON on
  stdout; `--stamp` only on promote; stamp keys `promotion_basis`,
  `fallback_genuine_ic`, `fallback_as_of`, `fallback_prod_staleness_days`
  unchanged (`fallback_pair_promote.py` reads `promotion_basis` + `passed`).
- Additive: verdict keys `quality_floor`, `served_fallback_genuine_ic`,
  `served_promotion_basis`, `failure_classes`; stamp key
  `fallback_quality_floor`.
- Step 4b's comment block (lines ~480-483) still describes the old rule
  ("non-negative stamped genuine_ic (ordinal/sign only) AND no downward
  ratchet") — text only, no logic; to be refreshed in the umbrella when it
  next changes. Module import stays `renquant_backtesting.wf_gate.freshness_fallback`;
  it now imports `runner` (pandas) at module load, which the pinned runtime
  already carries (the gate itself runs there).
- Expected live effect at the 2026-08-31 lapse: every current candidate
  refuses on `quality_floor` (measured below), the 28-day ceiling lapses the
  served artifact, P-WF-GATE hard-fails and buys stay blocked — A4 §3's
  designed safe state, not an incident.

EVIDENCE:

```
tests:     tests/test_freshness_fallback.py 52 passed (0.019 → quality_floor
           refuse with the exact reason; 0.020 → FALLBACK_PROMOTE on a
           placebo-only reject; 13 substance classes each refuse at
           genuine_ic=0.5 and name the class; served fallback higher/lower/
           equal/absent/gate-passed → identical verdicts; missing-`passed`
           sub-verdict = failure; passed=False with nothing attributable =
           `unclassified`; old ratchet tokens absent from the module source;
           floor `is` runner.PLACEBO_GENUINE_IC_MARGIN; stamp/CLI contract).
           [本次实测]
suite:     749 passed, 2 failed, 11 skipped (CI command `pytest -q` via the
           umbrella .venv + sibling src on PYTHONPATH). The 2 fails are
           tests/test_lineage_scoring.py resolving `../renquant-model`
           relative to the worktree (operator-disk class); identical on the
           origin/main baseline (725 passed, same 2 failed). [本次实测]
dry-run:   REAL artifacts, read-only, no --stamp: prod panel-ltr.alpha158_fund
           (trained 08-02, served fallback_genuine_ic +0.00289) × staging
           weekly_20260823T170004Z (genuine_ic +0.0000):
             --as-of 2026-08-30 → REFUSE prod_stale (28d, SLA boundary)
             --as-of 2026-08-31 → REFUSE quality_floor
               quality_floor_not_met(genuine_ic=+0.0000 < 0.02); rc=1
           classify_gate_failures on both real stamps:
             [wf_benchmark_economics, placebo_ceiling(infra),
              regime_sanity_ic, trade_monotonicity]  [本次实测]
scope:     one module + its tests + this doc; runner untouched; no umbrella
           write (verified: only pre-existing untracked files under
           artifacts/prod).
```
