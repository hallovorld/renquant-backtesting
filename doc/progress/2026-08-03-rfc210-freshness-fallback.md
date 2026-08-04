# RFC#210 freshness fallback — the criterion-free unblock (operator P0)

**Date:** 2026-08-03 · `renquant-backtesting` · backtesting#101

STATUS:    policy module + CLI + 21 tests; UNWIRED (the promote script's
           REJECT branch consumes it in a follow-up umbrella PR before
           2026-08-09). Gate criterion UNTOUCHED — the v3 enforcement stays
           blocked on its own shadow-eval protocol.
WHAT:      decide(prod, staging, as_of) → FALLBACK_PROMOTE iff gate-rejected
           + prod >28d stale + candidate ≤10d old + stamped genuine_ic
           finite ≥0 + no-downward-ratchet across chained fallbacks; every
           check named with measured values; stamp() writes promotion_basis
           "freshness_fallback_rfc210" + inputs atomically, PROMOTE-only.
WHY:       Operator P0, verbatim in the module docstring. Fifth identical
           Sunday REJECT otherwise lands 2026-08-09 while the served model
           is 42d+ past the 28d SLA.

EVIDENCE:

```
tests:     21/21 (every check's pass AND its malformed twin: string/bool/
           NaN genuine_ic refuse-not-coerce; boundary 28d strict; ratchet
           equal/lower refuse; unstamped/unreadable refuse; stamp atomic +
           refuse-on-REFUSE; CLI exit codes).  [本次实测]
dry-run:   REAL artifacts (read-only): prod trained 06-21 (44d stale at
           08-04), staging weekly_20260802 genuine_ic +0.00289 →
           FALLBACK_PROMOTE, all five checks measured.  [本次实测]
suite:     617 passed; the 2 fails are the pre-existing umbrella
           byte-equivalence pair (operator-disk drift class), untouched.
scope:     "new module + tests only; no runner change, no script change,
            nothing consumes this yet."
```

## Review round 1 resolution (the design note above, resolved better than drafted)

Codex required an explicit rejected value and refusal of None/missing/
non-boolean. Investigating the producer revealed the first draft read the
WRONG KEY: `gate_verdict_before_override` is an ORCHESTRATOR promotion-time
provenance field, absent from staging stamps — while the runner's own
contract ALREADY stamps `passed: False` (explicit bool, verified on the real
2026-08-02 reject). The consumer now reads `passed`, only an explicit False
proceeds, and None/missing/"False"/0/1 each refuse with regressions. The
producer needed no repair; the real-artifact dry-run still yields
FALLBACK_PROMOTE with `stamped_verdict: False` recorded.

## Revert

git revert (nothing consumes it). The wiring PR carries its own revert.
