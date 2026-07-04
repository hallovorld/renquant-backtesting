# 2026-07-04 — Re-point ntfy sender to the canonical (campaign B6, audit XC-4)

`analysis/backtest_and_analyze.notify_ntfy` local urllib transport deleted;
now a thin seam over `renquant_common.notify.send` (keeps the "ntfy
notification sent" stdout line on success). renquant-common floor bumped to
0.10 (lockstep).

A/B deltas (enumerated): timeout 10 s → the standardized 5 s (this module was
the fleet's lone 10 s outlier, audit #296 §4.2); failure now logged via the
`renquant_common.notify` logger (counted) instead of a stdout print;
suppression accepts the truthy set 1/true/yes/on (superset of the previous
`== "1"`). `RENQUANT_NO_NOTIFY` honoring itself is unchanged — this repo
already had it.

Suite: 315 passed; the 2 pre-existing umbrella byte-equivalence failures
(`test_b1_lift` / `test_import_lift`) fail identically on main in the same
environment (umbrella-sibling drift, unrelated).
