#!/usr/bin/env python3
"""Research: faster entry/exit variants + dynamic fear-depth tilt.

Compares the current 200SMA core (with cash yield baseline) against:
  A) Golden-cross zone: SPY>50SMA AND 50SMA>200SMA  (fewer whipsaws)
  B) Faster re-entry:    SPY>200SMA OR (SPY>50SMA AND 50SMA rising)
  C) Donchian breakout:  SPY > N-day high (enter), < M-day low (exit)
  D) Dynamic tilt:       SPXL weight scaled by fear-dip percentile

All use REAL SPXL with decay, turnover cost, and IRX cash yield.
"""
from __future__ import annotations
import io, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import config
from scripts.simulate_trading import feardip_trades, close
from scripts.screen_downside_ic import build_features

ONEWAY = config.TOTAL_COST_ROUNDTRIP / 2
_IRX = pd.read_parquet("data/raw/macro/yield_2y.parquet").iloc[:, 0].dropna()
spy = close["SPY"].dropna(); spxl = close["SPXL"].dropna()
idx = spy.index.intersection(spxl.index)
spy, spxl = spy.reindex(idx), spxl.reindex(idx)
sret, xret = spy.pct_change(), spxl.pct_change()
sma200 = spy.rolling(200).mean(); sma50 = spy.rolling(50).mean()
hi100 = spy.rolling(100).max(); lo50 = spy.rolling(50).min()


def _fd_active(idx_):
    m = pd.Series(False, index=idx_); s = list(idx_)
    for t in feardip_trades():
        if t["entry_date"] in idx_:
            i = s.index(t["entry_date"])
            for h in range(10):
                if i + h < len(s):
                    m.iloc[i + h] = True
    return m


def _fd_pct(idx_):
    # bear composite percentile for dynamic tilt
    from signals.fear_dip import build_bear_score
    spy_df = close[["SPY"]]
    score = build_bear_score(spy_df).reindex(idx_)
    # rolling expanding percentile (causal)
    pct = pd.Series(np.nan, index=idx_)
    vals = score.values
    for i in range(252, len(idx_)):
        hist = vals[:i+1]; hist = hist[~np.isnan(hist)]
        if len(hist) > 252 and not np.isnan(vals[i]):
            pct.iloc[i] = (hist <= vals[i]).mean()
    return pct


fd_on = _fd_active(idx)


def core_signal(kind):
    if kind == "200sma":
        return (spy.shift(1) > sma200.shift(1)).fillna(False)
    if kind == "golden":  # both 50 and 200, AND 50>200
        return ((spy.shift(1) > sma50.shift(1)) & (sma50.shift(1) > sma200.shift(1))).fillna(False)
    if kind == "fast_reentry":
        base = spy.shift(1) > sma200.shift(1)
        # fast re-entry: when above 50SMA AND 50SMA rising 5d
        sma50_rising = sma50.diff(5).shift(1) > 0
        above50 = spy.shift(1) > sma50.shift(1)
        return (base | (above50 & sma50_rising)).fillna(False)
    if kind == "donchian":
        # long when SPY > 100d high (yesterday); exit when SPY < 50d low. Stateful.
        s = pd.Series(False, index=idx); inpos = False
        for i in range(100, len(idx)):
            p = spy.iloc[i-1]; h = hi100.iloc[i-1]; l = lo50.iloc[i-1]
            if not inpos and p >= h * 0.999:  inpos = True
            elif inpos and p <= l * 1.001:    inpos = False
            s.iloc[i] = inpos
        return s


def simulate(core_kind, tilt_kind, since):
    core_on = core_signal(core_kind)
    fd_pct = _fd_pct(idx) if tilt_kind == "dynamic" else None
    idx_s = idx[idx >= since]
    cap = 10000.0; peak = cap; mdd = 0; prev_spy = prev_xl = 0; rets = []
    for d in idx_s:
        c_on = bool(core_on.loc[d]); f_on = bool(fd_on.loc[d])
        if tilt_kind == "fixed":
            w = 0.15
        elif tilt_kind == "dynamic" and f_on and not np.isnan(fd_pct.loc[d] if d in fd_pct.index else np.nan):
            # 0.85->0.10, 0.99->0.25 (linear); clamp
            p = fd_pct.loc[d]
            w = float(np.clip(0.10 + (p - 0.85) / 0.14 * 0.15, 0.10, 0.25))
        else:
            w = 0.15
        if c_on and f_on:  spy_w, xl_w = 1 - w, w
        elif c_on:         spy_w, xl_w = 1.0, 0.0
        elif f_on:         spy_w, xl_w = 1.0, 0.0
        else:              spy_w, xl_w = 0.0, 0.0
        turn = abs(spy_w - prev_spy) + abs(xl_w - prev_xl)
        cash_w = max(0, 1 - prev_spy - prev_xl)
        iy = _IRX.asof(d); cy = (iy/100/252) if pd.notna(iy) else 0
        d_ret = prev_spy*(sret.get(d, 0) or 0) + prev_xl*(xret.get(d, 0) or 0) + cash_w*cy - turn*ONEWAY
        cap *= (1 + d_ret); peak = max(peak, cap); mdd = min(mdd, cap/peak - 1); rets.append(d_ret)
        prev_spy, prev_xl = spy_w, xl_w
    r = pd.Series(rets); sh = r.mean()/r.std()*np.sqrt(252) if r.std() > 0 else 0
    return cap/10000 - 1, sh, mdd


def line(tag, since):
    vals = []
    for core, tilt in [("200sma","fixed"), ("golden","fixed"), ("fast_reentry","fixed"),
                       ("donchian","fixed"), ("200sma","dynamic")]:
        tot, sh, mdd = simulate(core, tilt, since)
        vals.append((f"{core}+{tilt}", tot, sh, mdd))
    print(f"\n=== {tag} ===")
    print(f"  {'variant':<22} {'수익률':>8} {'Sharpe':>7} {'MDD':>7}")
    for name, t, s, m in vals:
        print(f"  {name:<22} {t*100:>+7.1f}% {s:>+6.2f}  {m*100:>+6.1f}%")


if __name__ == "__main__":
    line("FULL OOS 2021+", pd.Timestamp("2021-01-01"))
    line("HOLDOUT 2024+",  pd.Timestamp("2024-01-01"))
