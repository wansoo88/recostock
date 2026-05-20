#!/usr/bin/env python3
"""Downside-predictor IC screen.

Question: does ANY feature predict NEGATIVE forward returns on the index well
enough to build a dedicated short model? Every short attempt so far reused the
long model's proba and failed because index ETFs drift up ("weak != falling").
Here we screen purpose-built downside features against forward 5d returns,
causally (feature at T uses data <= T; target is T -> T+5).

For each feature we report:
  IC        — Spearman corr(feature_t, fwd5_ret) over full sample
  IC_hold   — same, holdout 2024+ (robustness)
  botMean   — mean fwd5 return when feature is in its most-bearish decile
  botNet    — botMean for a SHORT position net of cost (= -botMean - cost)
  hitShort  — P(fwd5 < 0) in the bearish decile (short win rate, gross)

A short edge exists if botNet > 0 AND consistent across full/holdout AND
hitShort > 0.5. Gate: |IC| > config.IC_MIN_VIABLE.

Run: python scripts/screen_downside_ic.py
"""
from __future__ import annotations
import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import config

RAW = Path("data/raw")
MACRO = RAW / "macro"
HOLDOUT = pd.Timestamp("2024-01-01")
FWD = 5


def _s(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    return df.iloc[:, 0].dropna()


def zscore(s: pd.Series, w: int = 60) -> pd.Series:
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_features(target_tk: str = "SPY") -> tuple[pd.DataFrame, pd.Series]:
    ohlcv = pd.read_parquet(RAW / "etf_ohlcv.parquet")
    close = ohlcv["Close"] if isinstance(ohlcv.columns, pd.MultiIndex) else ohlcv
    spy = close[target_tk].dropna()

    vix = _s(MACRO / "vix.parquet")
    vix9d = _s(MACRO / "vix9d.parquet")
    vix3m = _s(MACRO / "vix3m.parquet")
    vvix = _s(MACRO / "vvix.parquet")
    move = _s(MACRO / "move.parquet")
    skew = _s(MACRO / "skew.parquet")
    hyg = _s(MACRO / "hyg.parquet")
    lqd = _s(MACRO / "lqd.parquet")
    tlt = _s(MACRO / "tlt.parquet")
    dxy = _s(MACRO / "dxy.parquet")
    gold = _s(MACRO / "gold.parquet")
    y10 = _s(MACRO / "yield_10y.parquet")
    y2 = _s(MACRO / "yield_2y.parquet")

    sectors = [c for c in ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI"] if c in close.columns]
    breadth = (close[sectors] > close[sectors].rolling(50).mean()).mean(axis=1)

    f = pd.DataFrame(index=spy.index)
    f["vix_level"] = vix.reindex(spy.index)
    f["vix_chg5"] = vix.reindex(spy.index).pct_change(5)
    f["vix_term_9d"] = (vix9d / vix).reindex(spy.index)        # >1 = backwardation = stress
    f["vix_term_3m"] = (vix / vix3m).reindex(spy.index)        # >1 = stress
    f["vvix_z"] = zscore(vvix).reindex(spy.index)
    f["move_z"] = zscore(move).reindex(spy.index)
    f["skew_z"] = zscore(skew).reindex(spy.index)
    f["hyg_ret5"] = hyg.pct_change(5).reindex(spy.index)        # credit stress = HYG falls
    f["hyg_lqd"] = (hyg / lqd).pct_change(5).reindex(spy.index) # HY underperforms IG = stress
    f["tlt_ret5"] = tlt.pct_change(5).reindex(spy.index)        # bond rally = risk off
    f["yc_10_2"] = (y10 - y2).reindex(spy.index)               # inversion
    f["y10_chg5"] = y10.diff(5).reindex(spy.index)
    f["dxy_chg5"] = dxy.pct_change(5).reindex(spy.index)       # dollar up = risk off
    f["gold_spy"] = (gold.pct_change(5) - spy.pct_change(5)).reindex(spy.index)
    f["spy_dist_sma200"] = (spy / spy.rolling(200).mean() - 1)  # overbought
    f["spy_rsi14"] = rsi(spy)
    f["spy_mom20"] = spy.pct_change(20)

    fwd = spy.pct_change(FWD).shift(-FWD)
    return f, fwd


def screen(target_tk: str = "SPY") -> pd.DataFrame:
    f, fwd = build_features(target_tk)
    rows = []
    cost = config.TOTAL_COST_ROUNDTRIP
    for col in f.columns:
        s = f[col]
        valid = s.notna() & fwd.notna()
        x, y = s[valid], fwd[valid]
        if len(x) < 250:
            continue
        ic = x.corr(y, method="spearman")
        hold = x.index >= HOLDOUT
        ic_h = x[hold].corr(y[hold], method="spearman") if hold.sum() > 60 else np.nan
        # bearish decile: a feature predicts drops via low OR high extreme.
        # Pick the extreme whose mean fwd return is most negative.
        lo = x <= x.quantile(0.10)
        hi = x >= x.quantile(0.90)
        lo_mean, hi_mean = y[lo].mean(), y[hi].mean()
        if lo_mean <= hi_mean:
            bear, side = lo, "low"
            bear_mean = lo_mean
        else:
            bear, side = hi, "high"
            bear_mean = hi_mean
        short_ret_net = -bear_mean - cost          # short position net of cost
        hit_short = float((y[bear] < 0).mean())
        rows.append({
            "feature": col, "side": side, "n": int(len(x)),
            "IC": round(float(ic), 4),
            "IC_hold": round(float(ic_h), 4) if not np.isnan(ic_h) else np.nan,
            "botMean%": round(float(bear_mean) * 100, 3),
            "botNetShort%": round(float(short_ret_net) * 100, 3),
            "hitShort": round(hit_short, 3),
        })
    df = pd.DataFrame(rows).sort_values("botNetShort%", ascending=False)
    return df


if __name__ == "__main__":
    for tk in ["SPY", "QQQ"]:
        print("=" * 100)
        print(f"DOWNSIDE IC SCREEN — target {tk}, fwd {FWD}d, cost {config.TOTAL_COST_ROUNDTRIP:.2%}, holdout {HOLDOUT.date()}")
        print(f"  IC gate = {config.IC_MIN_VIABLE} | short edge needs botNetShort% > 0 AND hitShort > 0.5 AND consistent")
        print("=" * 100)
        df = screen(tk)
        print(df.to_string(index=False))
        print()
        viable = df[(df["botNetShort%"] > 0) & (df["hitShort"] > 0.5) & (df["IC"].abs() > config.IC_MIN_VIABLE)]
        print(f"  >>> {tk}: {len(viable)} feature(s) clear short-edge screen")
        if len(viable):
            print(viable[["feature", "side", "IC", "IC_hold", "botNetShort%", "hitShort"]].to_string(index=False))
        print()
