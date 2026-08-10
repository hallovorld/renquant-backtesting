"""Regime-series injection seam for the BEAR-exit confirmatory arms (orch#962 B2).

The frozen prereg (orch repo ``doc/design/2026-08-08-bear-exit-prereg.md``
sections 3/3.1) requires book simulations under INJECTED regime series:

* **D2 placebo** — the amendment keyed to an episode-block-permuted regime
  series, 200 seeds;
* **D4 shifts** — the regime series lagged +5/+10/+20 days.

The simulator computes regimes internally (umbrella
``kernel/pipeline/task_regime.py``: Hurst -> CUSUM -> GMM -> BEAROverride ->
TrendOverlay -> ``RegimeFinalizeTask``, which commits ``ctx.regime``). This
module provides the injection seam the confirmatory runner needs, WITHOUT
touching the umbrella repo: it wraps ``RegimeFinalizeTask.run`` so that after
the computed finalize, the per-date label from a caller-supplied series
REPLACES ``ctx.regime`` (and its Track-C alias ``ctx.final_regime``) before
any downstream decision task reads it. Both sim arms in one process (candidate
+ golden comparison leg) therefore run under the SAME injected series — the
prereg's paired-arm contract.

Fail-closed contract (the guards-validate-the-wrong-object lesson):

* an injected series missing a date the sim needs is a hard
  :class:`RegimeInjectionError` at that bar — never a silent fallback to the
  computed regime;
* installation errors loudly when the kernel task module is not importable —
  never "sim ran on computed regimes while claiming injection";
* after patching, the install is verified by reading the marker back off the
  class.

When no series is injected this module is inert: importing it has no side
effects, and ``RegimeFinalizeTask.run`` is untouched.
"""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

_KERNEL_TASK_MODULE = "kernel.pipeline.task_regime"
_KERNEL_CONFIG_MODULE = "kernel.config"
_CSV_HEADER_LABELS = {"regime", "label"}
_MARKER = "__regime_injection__"


class RegimeInjectionError(RuntimeError):
    """Any failure of the regime-injection seam. Always fail-closed."""


@dataclass(frozen=True)
class RegimeSeries:
    """A per-date regime label series + the provenance of its source file.

    ``by_date`` maps ISO ``YYYY-MM-DD`` date keys to non-empty label strings.
    ``sha256`` is the digest of the source file's exact bytes — recorded in
    the sim output metadata so a placebo/shift arm's results are bound to the
    literal series that produced them.
    """

    by_date: Mapping[str, str]
    sha256: str
    source: str


def _parse_iso_date(raw: str, *, context: str) -> str:
    text = str(raw).strip()
    try:
        return _dt.date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise RegimeInjectionError(
            f"regime series {context}: unparseable date {raw!r} "
            f"(expected YYYY-MM-DD)") from exc


def _validate_label(raw: object, *, context: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise RegimeInjectionError(
            f"regime series {context}: empty or non-string label {raw!r}")
    return raw.strip()


def _add_entry(by_date: dict, date_key: str, label: str, *, context: str) -> None:
    if date_key in by_date:
        raise RegimeInjectionError(
            f"regime series {context}: duplicate date {date_key}")
    by_date[date_key] = label


def _load_csv(text: str, *, context: str) -> dict:
    by_date: dict = {}
    rows = [row for row in csv.reader(text.splitlines()) if row]
    for i, row in enumerate(rows):
        if len(row) != 2:
            raise RegimeInjectionError(
                f"regime series {context}: row {i + 1} has {len(row)} "
                f"column(s), expected exactly 2 (date,label)")
        first, second = row[0].strip(), row[1].strip()
        if i == 0:
            try:
                _dt.date.fromisoformat(first)
            except ValueError:
                # Not a data row — accept ONLY a known header, else error
                # (silently skipping a malformed first row would hide data).
                if second.lower() in _CSV_HEADER_LABELS and first.lower() == "date":
                    continue
                raise RegimeInjectionError(
                    f"regime series {context}: first row {row!r} is neither "
                    f"a YYYY-MM-DD data row nor a 'date,regime' header")
        date_key = _parse_iso_date(first, context=context)
        label = _validate_label(second, context=context)
        _add_entry(by_date, date_key, label, context=context)
    return by_date


def _load_json(text: str, *, context: str) -> dict:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RegimeInjectionError(
            f"regime series {context}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RegimeInjectionError(
            f"regime series {context}: JSON must be an object of "
            f"{{date: label}}, got {type(payload).__name__}")
    by_date: dict = {}
    for raw_date, raw_label in payload.items():
        date_key = _parse_iso_date(raw_date, context=context)
        label = _validate_label(raw_label, context=context)
        _add_entry(by_date, date_key, label, context=context)
    return by_date


def load_regime_series(path: "str | Path") -> RegimeSeries:
    """Load a date->label series from a ``.csv`` or ``.json`` file.

    CSV: exactly two columns, optional ``date,regime`` (or ``date,label``)
    header. JSON: one object of ``{"YYYY-MM-DD": "LABEL", ...}``. Any
    structural defect (missing file, unknown extension, duplicate date, bad
    date, empty label, empty series) raises :class:`RegimeInjectionError`.
    """
    p = Path(path)
    context = f"({p})"
    if not p.is_file():
        raise RegimeInjectionError(f"regime series {context}: file not found")
    raw = p.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")
    suffix = p.suffix.lower()
    if suffix == ".csv":
        by_date = _load_csv(text, context=context)
    elif suffix == ".json":
        by_date = _load_json(text, context=context)
    else:
        raise RegimeInjectionError(
            f"regime series {context}: unsupported extension {suffix!r} "
            f"(expected .csv or .json)")
    if not by_date:
        raise RegimeInjectionError(f"regime series {context}: empty series")
    return RegimeSeries(by_date=dict(by_date), sha256=digest, source=str(p))


def date_key(today: object) -> str:
    """Normalize a sim bar date (date/datetime/Timestamp/str) to YYYY-MM-DD."""
    if isinstance(today, str):
        return _parse_iso_date(today, context="(bar date)")
    if isinstance(today, _dt.datetime):
        return today.date().isoformat()
    if isinstance(today, _dt.date):
        return today.isoformat()
    date_attr = getattr(today, "date", None)
    if callable(date_attr):  # pd.Timestamp without importing pandas here
        return date_attr().isoformat()
    raise RegimeInjectionError(
        f"cannot derive a YYYY-MM-DD key from bar date {today!r} "
        f"({type(today).__name__})")


def required_sim_dates(spy_df, start: object, end: object) -> "list[str]":
    """The bar dates a driver sim will iterate: ``spy_df.loc[start:end].index``.

    Mirrors ``run_backtest``'s own bar derivation so the driver can preflight
    series coverage before any compute. An empty/absent frame yields ``[]``
    (the per-bar guard inside the seam remains the hard backstop).
    """
    if spy_df is None or getattr(spy_df, "empty", True):
        return []
    idx = spy_df.loc[str(start):str(end)].index
    return [date_key(d) for d in idx]


def uncovered_dates(series: RegimeSeries, needed: Sequence[str]) -> "list[str]":
    """Sorted needed dates that the series does NOT cover."""
    return sorted(d for d in set(needed) if d not in series.by_date)


@dataclass
class InstalledRegimeInjection:
    """Handle for an installed seam; carries the metadata the sim outputs record."""

    series: RegimeSeries
    label_set_validated: bool

    def metadata(self) -> dict:
        return {
            "active": True,
            "sha256": self.series.sha256,
            "source": self.series.source,
            "n_dates": len(self.series.by_date),
            "label_set_validated": self.label_set_validated,
        }


_installed_state: "dict | None" = None


def _validate_label_set(series: RegimeSeries) -> bool:
    """Check labels against ``kernel.config.REGIMES`` when importable.

    Returns True when the check RAN and passed; False when the kernel config
    module is unavailable (stub environments) — recorded in the metadata so a
    reader can tell 'validated' from 'not validatable'. A label outside the
    kernel's regime set is a hard error: downstream ``dict.get(ctx.regime)``
    lookups would silently behave as 'no params for this regime'.
    """
    try:
        cfg_mod = importlib.import_module(_KERNEL_CONFIG_MODULE)
    except ImportError:
        return False
    regimes = getattr(cfg_mod, "REGIMES", None)
    if not regimes:
        return False
    allowed = set(regimes)
    unknown = sorted({v for v in series.by_date.values() if v not in allowed})
    if unknown:
        raise RegimeInjectionError(
            f"injected regime series ({series.sha256[:12]}) contains label(s) "
            f"outside kernel.config.REGIMES {sorted(allowed)}: {unknown}")
    return True


def install_regime_injection(series: RegimeSeries) -> InstalledRegimeInjection:
    """Patch ``RegimeFinalizeTask.run`` so ``ctx.regime`` follows the series.

    The wrapper calls the computed finalize first (regime state machinery,
    confidence, evidence all still run), then REPLACES ``ctx.regime`` and
    ``ctx.final_regime`` with the injected label for ``ctx.today``. The
    class-level patch covers every pipeline instance in the process — both
    the candidate and the golden comparison leg.
    """
    global _installed_state
    if _installed_state is not None:
        raise RegimeInjectionError(
            "regime injection already installed; refusing a second install "
            "(one series per process)")
    try:
        task_mod = importlib.import_module(_KERNEL_TASK_MODULE)
    except ImportError as exc:
        raise RegimeInjectionError(
            f"cannot import {_KERNEL_TASK_MODULE} — the strategy dir must be "
            f"on sys.path BEFORE installing regime injection; refusing to run "
            f"a sim that would silently use computed regimes") from exc
    task_cls = getattr(task_mod, "RegimeFinalizeTask", None)
    if task_cls is None:
        raise RegimeInjectionError(
            f"{_KERNEL_TASK_MODULE} has no RegimeFinalizeTask — the seam's "
            f"anchor is gone; refusing to run without injection")
    label_set_validated = _validate_label_set(series)
    original_run = task_cls.run

    def _injected_run(self, ctx):
        out = original_run(self, ctx)
        today = getattr(ctx, "today", None)
        if today is None:
            raise RegimeInjectionError(
                "regime injection: context has no 'today' date; cannot key "
                "the injected series")
        key = date_key(today)
        if key not in series.by_date:
            raise RegimeInjectionError(
                f"injected regime series ({series.sha256[:12]}) has no label "
                f"for sim date {key}; refusing to fall back to the computed "
                f"regime")
        label = series.by_date[key]
        computed = getattr(ctx, "regime", None)
        ctx.regime = label
        ctx.final_regime = label
        counts = getattr(ctx, "regime_counts", None)
        if isinstance(counts, dict) and computed != label:
            if counts.get(computed, 0) > 0:
                counts[computed] -= 1
            counts[label] = counts.get(label, 0) + 1
        evidence = getattr(ctx, "_regime_evidence", None)
        if isinstance(evidence, dict):
            evidence["regime_injection"] = {
                "active": True,
                "sha256": series.sha256,
                "computed_regime": computed,
                "injected_regime": label,
            }
        return out

    setattr(_injected_run, _MARKER, True)
    _injected_run.__regime_injection_sha256__ = series.sha256
    task_cls.run = _injected_run
    # Verify the install by reading the marker back off the class — the
    # object downstream pipelines will actually call.
    if not getattr(task_cls.run, _MARKER, False):
        raise RegimeInjectionError(
            "regime injection install verification failed: "
            "RegimeFinalizeTask.run does not carry the injection marker")
    _installed_state = {
        "module": task_mod,
        "cls": task_cls,
        "original_run": original_run,
    }
    return InstalledRegimeInjection(
        series=series, label_set_validated=label_set_validated)


def uninstall_regime_injection() -> None:
    """Restore the original ``RegimeFinalizeTask.run`` (test hygiene)."""
    global _installed_state
    if _installed_state is None:
        return
    _installed_state["cls"].run = _installed_state["original_run"]
    _installed_state = None
