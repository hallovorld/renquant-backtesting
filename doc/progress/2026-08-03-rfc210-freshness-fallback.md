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

## Honest design note for review

`gate_verdict_before_override` on REAL stamps is None (not False); the check
refuses only a POSITIVE True (gate passed → use the normal path) because the
script-flow (REJECT branch) is the primary signal and a None-stamped
candidate is measurably the real REJECT shape. The stamped value is recorded
in the verdict either way. If review prefers fail-closed-on-None, the runner
must first start stamping an explicit False — flagging rather than hiding
the choice.

## Revert

git revert (nothing consumes it). The wiring PR carries its own revert.
