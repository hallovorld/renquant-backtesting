"""EXPLORATORY diagnostic (NOT a config proposal): realized-vol admission cap vs downstream
vol-aware sizing, on a SURVIVORSHIP-BIASED panel (291/291 survive to 2026 -> high-vol biased UP).
Upper-bound only. Theory: Kelly f*=mu/sigma^2 (continuous, no binary threshold); low-vol anomaly/
BAB = the real continuous vol-penalty case; Moreira-Muir is PORTFOLIO vol-timing, NOT a
cross-sectional single-security gate (background only). Proxy XGB ranker (not live PatchTST;
omits Kelly numerator/QP/concentration/daily-rebalance/live-gate-order).

Re-homed from renquant-orchestrator (model-fitting research belongs in the backtest-evaluation
repo, not the orchestrator). Two correctness fixes applied during the move:
  * LEAKAGE: the forward label ``fwd_60d_excess`` is ``close.shift(-60)`` = 60 TRADING sessions,
    but the original purge subtracted ``Timedelta(days=60)`` (~42 trading sessions) -> training
    labels near the cutoff overlapped the test interval. The embargo is now counted in TRADING
    SESSIONS off the sorted unique date index, with STRICT non-overlap (EMB_SESSIONS = horizon+1
    = 61, so the last training label ends strictly before test_lo; see ``purged_test_windows``).
  * REGIME UNIFORMITY: the per-date regime is now asserted uniform across names (fail closed)
    before it is used; the prior code silently took ``.groupby("date").first()``.
"""
from __future__ import annotations
import glob
import numpy as np
import pandas as pd

R = "/Users/renhao/git/github/RenQuant"
LABEL = "fwd_60d_excess"
# fwd_60d_excess = close.shift(-60): the label horizon is 60 TRADING SESSIONS, so the embargo
# between a training label-end and the first test date must also be 60 trading sessions.
LABEL_HORIZON_SESSIONS = 60
REGCOL = {"BULL_CALM": "regime_p_bull_calm", "BEAR": "regime_p_bear", "BULL_VOLATILE": "regime_p_bull_volatile"}
REGNAME = ["BULL_CALM", "BEAR", "BULL_VOLATILE"]
# EMB_SESSIONS strictly exceeds the 60-session label horizon so the last training label ends
# STRICTLY BEFORE the first test date (no boundary overlap) — see purged_test_windows().
N_TEST_FOLDS, EMB_SESSIONS = 5, LABEL_HORIZON_SESSIONS + 1
CAPS = [0.6, 0.8, 1.0, 1.2, 1.5, np.inf]
VOL_FLOOR, VOL_CEIL = 0.05, 1.5
COST_BASE, COST_K = 0.0005, 0.0020


def annualize(monthly):
    if len(monthly) < 6:
        return {"n": int(len(monthly)), "sharpe": np.nan}
    mu, sd = monthly.mean() * 12, monthly.std() * np.sqrt(12)
    dn = monthly[monthly < 0].std() * np.sqrt(12)
    cum = (1 + monthly).cumprod()
    return {"n": int(len(monthly)), "ann_ret": mu, "ann_vol": sd,
            "sharpe": (mu / sd if sd else np.nan), "sortino": (mu / dn if dn else np.nan),
            "maxDD": float((cum / cum.cummax() - 1).min()),
            "cvar5": float(monthly[monthly <= monthly.quantile(0.05)].mean()),
            "median_m": float(monthly.median()), "hit": float((monthly > 0).mean())}


def block_bootstrap_ci(diff, block=3, n_boot=2000, seed=0, lo=2.5, hi=97.5):
    diff = np.asarray([d for d in diff if np.isfinite(d)], dtype=float)
    n = len(diff)
    if n < block + 1:
        return (float(np.mean(diff)) if n else np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=nb)
        means[i] = np.concatenate([diff[s:s + block] for s in starts])[:n].mean()
    return float(diff.mean()), float(np.percentile(means, lo)), float(np.percentile(means, hi))


def purged_test_windows(dates, n_folds, embargo_sessions, label_horizon_sessions=None):
    """Non-overlapping purged walk-forward test windows with a TRADING-SESSION embargo.

    ``dates`` is the panel's full set of observation timestamps (may contain duplicates per
    date across names). The embargo is enforced on the sorted UNIQUE date index, in *trading
    sessions* (not calendar days), to match the forward label ``fwd_60d_excess`` =
    ``close.shift(-60)`` = 60 trading sessions.

    Contract (STRICT non-overlap): a training row dated ``train_end`` carries a label whose
    realisation window ends ``label_horizon_sessions`` sessions later, at index
    ``idx(train_end) + label_horizon_sessions``. That index must be **strictly less** than the
    first test index ``lo_i`` so the label's last observed price falls *outside* the test
    interval. We therefore require ``embargo_sessions > label_horizon_sessions`` and pick the
    largest ``train_end`` satisfying ``idx(train_end) + label_horizon_sessions < lo_i``.

    This matches the production WF-gate convention in
    ``wf_gate/train_walkforward_panel.py`` (``data_end = cutoff - BDay(lookahead_days)`` with a
    strict ``data_end<`` cut): the final training label ends *before*, never *on*, the test
    boundary.

    Returns a list of ``(train_end, test_lo, test_hi)`` timestamps. ``train_end`` is inclusive:
    callers keep rows with ``date <= train_end`` for training.
    """
    if label_horizon_sessions is None:
        label_horizon_sessions = LABEL_HORIZON_SESSIONS
    embargo_sessions = int(embargo_sessions)
    label_horizon_sessions = int(label_horizon_sessions)
    # Strict non-overlap is only achievable if the embargo strictly exceeds the label horizon;
    # a 60-session embargo on a 60-session label leaves the last label ending ON test_lo.
    if embargo_sessions <= label_horizon_sessions:
        raise ValueError(
            f"embargo_sessions={embargo_sessions} must be STRICTLY greater than "
            f"label_horizon_sessions={label_horizon_sessions} to keep the last training label "
            f"end strictly before the first test date (no boundary overlap)."
        )
    udates = np.array(sorted(pd.to_datetime(pd.unique(dates))))
    n_u = len(udates)
    b = np.linspace(0, n_u, n_folds + 2).astype(int)
    out = []
    for k in range(1, n_folds + 1):
        lo_i, hi_i = b[k], b[k + 1] - 1
        if lo_i > hi_i or lo_i >= n_u:
            continue
        test_lo, test_hi = udates[lo_i], udates[hi_i]
        # Embargo cutoff index = the unique-date position `embargo_sessions` trading sessions
        # BEFORE the test-start position. With embargo > horizon, the last training label
        # (ending at cutoff_i + horizon) lands strictly before lo_i.
        cutoff_i = lo_i - embargo_sessions
        if cutoff_i < 0:
            # Not enough history to embargo a full horizon: drop this fold rather than leak.
            continue
        # Defensive invariant: the last training label must end strictly before the test start.
        assert cutoff_i + label_horizon_sessions < lo_i, (
            f"purge invariant violated: label end {cutoff_i + label_horizon_sessions} "
            f">= test start {lo_i}"
        )
        train_end = udates[cutoff_i]
        out.append((pd.Timestamp(train_end), pd.Timestamp(test_lo), pd.Timestamp(test_hi)))
    return out


def assert_regime_uniform_per_date(df, regime_col="regime", date_col="date"):
    """Fail closed if any date carries more than one distinct regime label across names.

    The monthly regime slice assumes the regime is a market-level label that is identical for
    every name on a given date. The original code claimed this ("verified uniform per date") but
    never checked it. We now verify it and raise on any conflict before assigning regimes.
    """
    nun = df.groupby(date_col)[regime_col].nunique()
    bad = nun[nun > 1]
    if len(bad):
        sample = ", ".join(str(pd.Timestamp(d).date()) for d in bad.index[:5])
        raise ValueError(
            f"regime label is NOT uniform across names on {len(bad)} date(s) "
            f"(e.g. {sample}); the monthly regime slice assumes a market-level label. "
            f"Refusing to silently .first() over conflicting per-name regimes."
        )
    return df.groupby(date_col)[regime_col].first()


def _load_px(t):
    fs = glob.glob(f"{R}/data/ohlcv/{t}/1d.parquet")
    if not fs:
        return None
    d = pd.read_parquet(fs[0]).sort_index()
    d.index = pd.to_datetime(d.index)
    return d["close"]


def main():
    from xgboost import XGBRegressor
    df = pd.read_parquet(f"{R}/data/alpha158_291_fund_regime_dataset.parquet").dropna(subset=[LABEL]).copy()
    df["date"] = pd.to_datetime(df["date"]); df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    df["regime"] = df[[REGCOL[r] for r in REGNAME]].values.argmax(1)
    regime_by_date = assert_regime_uniform_per_date(df)  # fail closed on conflicting per-date regimes
    tickers = sorted(str(t).upper() for t in df["ticker"].unique())
    meta = {"ticker", "date", "regime", "fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess", "split_label", *REGCOL.values()}
    feats = [c for c in df.columns if c not in meta]
    dts = np.sort(df["date"].unique())
    preds = []
    for train_end, tlo, thi in purged_test_windows(dts, N_TEST_FOLDS, EMB_SESSIONS):
        tr = df[df.date <= train_end]; te = df[(df.date >= tlo) & (df.date <= thi)]
        if len(tr) < 1000 or len(te) < 200:
            continue
        m = XGBRegressor(n_estimators=180, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=0, verbosity=0)
        m.fit(tr[feats].values, tr[LABEL].values)
        p = te[["date", "ticker"]].copy(); p["score"] = m.predict(te[feats].values); preds.append(p)
    P = pd.concat(preds, ignore_index=True)
    print(f"OOS rows={len(P)} (5 non-overlapping purged folds, {EMB_SESSIONS}-session embargo)", flush=True)
    px = {}
    for t in tickers + ["SPY"]:
        s = _load_px(t)
        if s is not None:
            px[t] = s
    spy = px["SPY"]; PX = pd.DataFrame(px).sort_index()
    VOL = PX.pct_change().rolling(60).std() * np.sqrt(252)
    me = pd.Series(PX.index).groupby([PX.index.year, PX.index.month]).max().values
    me = pd.to_datetime([d for d in me if P["date"].min() <= d <= P["date"].max()])
    score_by_date = {pd.Timestamp(d): g.set_index("ticker")["score"] for d, g in P.groupby("date")}
    oos = np.sort(P["date"].unique())

    def fwd_excess(t, d0, d1):
        try:
            p0, p1, s0, s1 = PX[t].asof(d0), PX[t].asof(d1), spy.asof(d0), spy.asof(d1)
            if any(pd.isna(x) or x <= 0 for x in (p0, p1, s0, s1)):
                return np.nan
            return (p1 / p0 - 1) - (s1 / s0 - 1)
        except Exception:
            return np.nan

    def run(cap, drop_top_frac=0.0, winsor=False):
        rets, dates, regs, prev = [], [], [], {}
        for j in range(len(me) - 1):
            d0, d1 = me[j], me[j + 1]
            sdl = [x for x in oos if x <= np.datetime64(d0)]
            if not sdl or not len(VOL.loc[:d0]):
                continue
            sc = score_by_date.get(pd.Timestamp(sdl[-1]))
            if sc is None:
                continue
            vol = VOL.loc[:d0].iloc[-1]
            cand = pd.DataFrame({"score": sc}); cand["vol"] = vol.reindex(cand.index); cand = cand.dropna()
            if len(cand) < 20:
                continue
            top = cand[(cand["score"] >= cand["score"].quantile(0.80)) & (cand["vol"] <= cap)]
            if not len(top):
                rets.append(0.0); dates.append(d1); regs.append(int(regime_by_date.asof(d0))); prev = {}; continue
            sig = top["vol"].clip(VOL_FLOOR, VOL_CEIL); w = (1.0 / sig ** 2); w = (w / w.sum()).to_dict()
            fr = {t: (0.0 if pd.isna(v := fwd_excess(t, d0, d1)) else v) for t in w}
            if winsor and fr:
                cut = np.quantile(list(fr.values()), 1 - drop_top_frac); fr = {t: min(v, cut) for t, v in fr.items()}
            gross = sum(w[t] * fr[t] for t in w)
            cost = sum(abs(w.get(t, 0) - prev.get(t, 0)) * (COST_BASE + COST_K * float(sig.get(t, 0.5))) for t in set(w) | set(prev))
            rets.append(gross - cost); dates.append(d1); regs.append(int(regime_by_date.asof(d0))); prev = w
        s = pd.Series(rets, index=pd.to_datetime(dates)); rg = pd.Series(regs, index=pd.to_datetime(dates))
        if drop_top_frac > 0 and not winsor and len(s):
            keep = s < s.quantile(1 - drop_top_frac); s, rg = s[keep], rg[keep]
        return s, rg

    print("\n=== cap sweep OVERALL ===", flush=True)
    series = {}
    print(f"{'cap':>5s} {'n':>4s} {'annRet':>7s} {'Sharpe':>7s} {'maxDD':>7s} {'CVaR5':>7s} {'medM':>8s}", flush=True)
    for c in CAPS:
        s, rg = run(c); series[c] = (s, rg); m = annualize(s)
        cs = "inf" if np.isinf(c) else f"{c:.1f}"
        print(f"{cs:>5s} {m['n']:4d} {m['ann_ret']:+.3f} {m['sharpe']:+.3f} {m['maxDD']:+.3f} {m['cvar5']:+.3f} {m['median_m']:+.5f}", flush=True)

    print("\n=== per ACTUAL REGIME Sharpe by cap (n) — BEAR small-sample ===", flush=True)
    print("regime        " + "".join(f"{('inf' if np.isinf(c) else f'{c:.1f}'):>11s}" for c in CAPS), flush=True)
    for ri, rn in enumerate(REGNAME):
        row = f"{rn:13s} "
        for c in CAPS:
            s, rg = series[c]; m = annualize(s[rg == ri])
            cell = f"{m['sharpe']:+.2f}({m['n']})" if m.get("n", 0) >= 6 else f"~({m.get('n',0)})"
            row += f"{cell:>11s}"
        print(row, flush=True)

    print("\n=== paired block-bootstrap CI: monthly delta vs cap 0.6 ===", flush=True)
    print("(NOTE: per-comparison CIs; any FUTURE positive inference must pre-register the primary", flush=True)
    print(" cap or apply a family-wise/FDR correction — see the research doc.)", flush=True)
    base = series[0.6][0]
    for c in [0.8, 1.0, 1.2, np.inf]:
        s = series[c][0]; idx = base.index.intersection(s.index)
        mean, lo, hi = block_bootstrap_ci((s.reindex(idx) - base.reindex(idx)).values)
        cs = "inf" if np.isinf(c) else f"{c:.1f}"
        sig = "" if (np.isnan(lo) or lo <= 0 <= hi) else "  *CI excludes 0"
        print(f"  cap {cs:>4s} - 0.6: dMean={mean:+.5f} 95%CI=[{lo:+.5f},{hi:+.5f}]{sig}", flush=True)

    print("\n=== robustness: TRUE-exclude vs WINSORIZE top-1% winner months ===", flush=True)
    for c in [0.6, 1.0, np.inf]:
        cs = "inf" if np.isinf(c) else f"{c:.1f}"
        se, _ = run(c, 0.01, False); sw, _ = run(c, 0.01, True)
        print(f"  cap {cs:>4s}: true-exclude Sharpe={annualize(se).get('sharpe'):+.3f}  winsorize Sharpe={annualize(sw).get('sharpe'):+.3f}", flush=True)
    print("NOTE: neither op removes survivorship (both keep only 2026 survivors).", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
