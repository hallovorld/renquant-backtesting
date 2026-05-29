"""Dollar-alpha backtest of model predictions vs SPY (frictionless quantile portfolio).

Answers "does the model's IC translate into $ alpha vs SPY?" — the question IC and
the standardized-label decile spread cannot answer (the research label is z-scored,
so its quantile spread is relative, not dollars). This joins TICKER-TAGGED prediction
dumps → RAW forward returns from ``data/ohlcv`` and forms a top-quintile (long) and
long-short portfolio per rebalance date, reporting realized return vs SPY's same-period
return.

Scope: this is the SIGNAL's portfolio alpha — frictionless, equal-weight, no QP / Kelly
sizing / stops / taxes / turnover costs of the full strategy. It is the cheap, honest
"is there $ edge over SPY here at all" read. The full strategy P&L requires the runtime
PatchTST scorer (which does not exist yet) run through the InferencePipeline; see the
multi-repo SOP. Mirrors the method behind the GBDT "top-quintile edge +1.56%/60d" read.

    python -m renquant_backtesting.forensics.patchtst_alpha \
        --preds-glob 'artifacts/patchtst_research/C_xstock_cut*_s42/*val_preds.parquet' \
        --ohlcv-dir data/ohlcv --horizon 60 --quantile 5

Requires preds with a ``ticker`` column (hf_trainer dumps it as of 2026-05-29).
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def _fwd_return(close: pd.Series, horizon: int) -> pd.Series:
    """h-trading-day forward simple return: close[t+h]/close[t] - 1."""
    return close.shift(-horizon) / close - 1.0


def load_raw_fwd(ohlcv_dir: Path, tickers: set[str], horizon: int) -> pd.DataFrame:
    """Long frame (date, ticker, fwd_ret) of RAW forward returns for the tickers."""
    rows = []
    for tkr in sorted(tickers):
        p = ohlcv_dir / tkr / "1d.parquet"
        if not p.exists():
            continue
        s = pd.read_parquet(p)["close"].sort_index()
        fwd = _fwd_return(s, horizon).rename("fwd_ret")
        df = fwd.reset_index(); df["ticker"] = tkr
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"no ohlcv parquets found under {ohlcv_dir} for given tickers")
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def load_preds(preds_glob: str) -> pd.DataFrame:
    fs = sorted(glob.glob(preds_glob))
    if not fs:
        raise FileNotFoundError(f"no preds match {preds_glob}")
    parts = []
    for f in fs:
        d = pd.read_parquet(f)
        if "ticker" not in d.columns or d["ticker"].isna().all():
            raise ValueError(f"{f} has no ticker column — re-dump with the 2026-05-29 hf_trainer")
        d["cut"] = Path(f).parent.name
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def backtest(preds: pd.DataFrame, raw: pd.DataFrame, spy_fwd: pd.Series,
             quantile: int) -> dict:
    """Per rebalance date: top-quantile (long) + long-short raw return vs SPY."""
    m = preds.merge(raw, on=["date", "ticker"], how="inner").dropna(subset=["fwd_ret"])
    long_ret, ls_ret, spy_ret, dates = [], [], [], []
    for d, g in m.groupby("date"):
        if len(g) < quantile * 2 or d not in spy_fwd.index:
            continue
        g = g.assign(q=pd.qcut(g["pred"].rank(method="first"), quantile, labels=False))
        top = g.loc[g["q"] == quantile - 1, "fwd_ret"].mean()
        bot = g.loc[g["q"] == 0, "fwd_ret"].mean()
        long_ret.append(top); ls_ret.append(top - bot)
        spy_ret.append(float(spy_fwd.loc[d])); dates.append(d)
    long_ret, ls_ret, spy_ret = map(np.array, (long_ret, ls_ret, spy_ret))
    alpha = long_ret - spy_ret
    n = len(alpha)
    return {
        "n_rebalances": n,
        "long_ret_mean": float(long_ret.mean()) if n else float("nan"),
        "spy_ret_mean": float(spy_ret.mean()) if n else float("nan"),
        "alpha_vs_spy_mean": float(alpha.mean()) if n else float("nan"),
        "alpha_hit_rate": float((alpha > 0).mean()) if n else float("nan"),
        "alpha_IR": float(alpha.mean() / alpha.std(ddof=1)) if n > 1 and alpha.std() else float("nan"),
        "long_short_mean": float(ls_ret.mean()) if n else float("nan"),
        "long_short_IR": float(ls_ret.mean() / ls_ret.std(ddof=1)) if n > 1 and ls_ret.std() else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("===")[0])
    ap.add_argument("--preds-glob", required=True)
    ap.add_argument("--ohlcv-dir", default="data/ohlcv")
    ap.add_argument("--spy", default=None, help="default <ohlcv-dir>/SPY/1d.parquet")
    ap.add_argument("--horizon", type=int, default=60, help="forward trading days")
    ap.add_argument("--quantile", type=int, default=5, help="5=quintiles, 10=deciles")
    a = ap.parse_args(argv)

    ohlcv = Path(a.ohlcv_dir)
    preds = load_preds(a.preds_glob)
    raw = load_raw_fwd(ohlcv, set(preds["ticker"].unique()), a.horizon)
    spy_close = pd.read_parquet(a.spy or ohlcv / "SPY" / "1d.parquet")["close"].sort_index()
    spy_fwd = _fwd_return(spy_close, a.horizon)
    spy_fwd.index = pd.to_datetime(spy_fwd.index)

    ann = 252 / a.horizon
    print(f"=== PatchTST dollar-alpha vs SPY ({a.horizon}d fwd, Q{a.quantile}, frictionless) ===")
    print(f"{'cut':16} {'n':>4} {'longRet':>9} {'SPYRet':>9} {'alpha':>9} {'alphaAnn':>9} {'hit':>5} {'IR':>6} {'L-S':>8}")
    pooled = preds.copy()
    for cut, g in list(preds.groupby("cut")) + [("POOLED", pooled)]:
        r = backtest(g, raw, spy_fwd, a.quantile)
        print(f"{cut:16} {r['n_rebalances']:>4} {r['long_ret_mean']:>+9.4f} {r['spy_ret_mean']:>+9.4f} "
              f"{r['alpha_vs_spy_mean']:>+9.4f} {r['alpha_vs_spy_mean']*ann:>+9.4f} "
              f"{r['alpha_hit_rate']:>5.2f} {r['alpha_IR']:>+6.2f} {r['long_short_mean']:>+8.4f}")
    print("\nalpha = long-leg raw return − SPY same-period return. >0 ⇒ the signal beats SPY.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
