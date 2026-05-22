#!/usr/bin/env python3
"""Backtest: trend-following core + fear-dip leverage tilt (the primary engine).

Reproduces the 2026-05-22 leverage decision. Core = long SPY when SPY > 200d
SMA, else cash. Tilt = on fear-dip days in an uptrend, shift TILT_SPXL_WEIGHT
into SPXL (3x). Validated with REAL SPXL prices (decay included) vs the
idealized constant-leverage model. Turnover cost applied on exposure changes.

Run: python scripts/backtest_trend_tilt.py
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
from scripts.simulate_trading import feardip_trades, close

ONEWAY = config.TOTAL_COST_ROUNDTRIP / 2


def _fd_mask(idx):
    m = pd.Series(False, index=idx); s = list(idx)
    for t in feardip_trades():
        if t["entry_date"] in idx:
            i = s.index(t["entry_date"])
            for h in range(10):
                if i + h < len(s):
                    m.iloc[i + h] = True
    return m


def run():
    spy = close["SPY"].dropna(); spxl = close["SPXL"].dropna()
    idx = spy.index.intersection(spxl.index)
    spy, spxl = spy.reindex(idx), spxl.reindex(idx)
    sret, xret = spy.pct_change(), spxl.pct_change()
    sma = spy.rolling(200).mean()
    core_on = (spy.shift(1) > sma.shift(1)).fillna(False)
    fd = _fd_mask(idx)

    def sim(w_spxl, real, since):
        spy_w = pd.Series(0.0, index=idx); xl_w = pd.Series(0.0, index=idx)
        boost = core_on & fd
        spy_w[core_on & ~boost] = 1.0
        spy_w[fd & ~core_on] = 1.0
        if real:
            spy_w[boost] = 1 - w_spxl; xl_w[boost] = w_spxl
        else:
            spy_w[boost] = 1 + 2 * w_spxl
        es, ex = spy_w[spy_w.index >= since], xl_w[xl_w.index >= since]
        r = es * sret.reindex(es.index) + ex * xret.reindex(es.index)
        turn = es.diff().abs().fillna(0) + ex.diff().abs().fillna(0)
        d = (r - turn * ONEWAY).dropna()
        eq = (1 + d).cumprod()
        return (eq.iloc[-1] - 1) * 100, d.mean() / d.std() * np.sqrt(252), \
               ((eq - eq.cummax()) / eq.cummax()).min() * 100

    for since, lab in [(pd.Timestamp("2021-01-01"), "FULL OOS 2021+"),
                       (pd.Timestamp("2024-01-01"), "HOLDOUT 2024+")]:
        bh = (spy[spy.index >= since].iloc[-1] / spy[spy.index >= since].iloc[0] - 1) * 100
        print(f"\n=== {lab} === [buy&hold {bh:+.1f}%]")
        for w, real, tag in [(0.0, True, "core only"),
                             (0.15, True, "1.3x tilt (real SPXL)"),
                             (0.25, True, "1.5x tilt (real SPXL)")]:
            tot, sh, mdd = sim(w, real, since)
            print(f"  {tag:26} {tot:>+6.1f}%  Sharpe {sh:>+4.2f}  MDD {mdd:>+6.1f}%")


if __name__ == "__main__":
    run()
