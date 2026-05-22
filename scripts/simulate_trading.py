#!/usr/bin/env python3
"""End-to-end 'what the user would actually experience' trading simulation.

Walks the OOS period day by day, following the all-weather ensemble verdict
(conviction = LIVE trend signal, fear-dip = EXPERIMENTAL mean-reversion),
applying volatility-targeted sizing, round-trip cost, and capital compounding.
Prints a trade-by-trade ledger + summary. Holdout (2024+) is the honest OOS.

Run: python scripts/simulate_trading.py
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
from scripts.conviction_winrate_buckets import regime_pass, LONG, HOLD, TP, SL, COST
from scripts.screen_downside_ic import build_features
from signals.sizing import position_size_pct

START_CAPITAL = 10_000.0
HOLDOUT = pd.Timestamp("2024-01-01")

o = pd.read_parquet("data/raw/etf_ohlcv.parquet")
close, high, low = o["Close"], o["High"], o["Low"]
proba = pd.read_parquet("data/logs/phase3_v3_oos_proba.parquet")["proba"].unstack(level=1)
e5 = proba.ewm(span=5).mean(); e3 = proba.ewm(span=3).mean(); e7 = proba.ewm(span=7).mean()
rp = regime_pass(close); idx = close.index
fri = [d for d in idx if d.dayofweek == 4]
cols = [c for c in LONG if c in e5.columns]
THR = config.CONVICTION_THRESHOLD


def conviction_trades():
    out = []
    for d in fri:
        if d not in e5.index or not rp.get(d, False):
            continue
        i = idx.get_loc(d)
        if i + HOLD >= len(idx):
            continue
        cand = [(tk, e5[tk].loc[d]) for tk in cols if not np.isnan(e5[tk].get(d, np.nan))
                and e3[tk].loc[d] >= THR and e7[tk].loc[d] >= THR and e5[tk].loc[d] >= THR]
        if not cand:
            continue
        tk = max(cand, key=lambda x: x[1])[0]
        entry = close[tk].iloc[i]; tp_px = entry * (1 + TP); sl_px = entry * (1 - SL)
        ret, exit_i = None, i + HOLD
        for hh in range(1, HOLD + 1):
            if low[tk].iloc[i + hh] <= sl_px:
                ret = -SL - COST; exit_i = i + hh; break
            if high[tk].iloc[i + hh] >= tp_px:
                ret = TP - COST; exit_i = i + hh; break
            if hh == HOLD:
                ret = (close[tk].iloc[i + hh] / entry - 1) - COST
        out.append({"entry_date": d, "exit_date": idx[exit_i], "strat": "conviction",
                    "ticker": tk, "entry": entry, "ret": ret})
    return out


def feardip_trades():
    spy = close["SPY"].dropna()
    f, _ = build_features("SPY")
    orient = {'vix_term_3m': -1, 'vix_term_9d': -1, 'vix_level': -1, 'tlt_ret5': -1,
              'gold_spy': -1, 'dxy_chg5': -1, 'move_z': +1, 'spy_dist_sma200': +1,
              'spy_rsi14': +1, 'spy_mom20': +1, 'hyg_lqd': +1, 'y10_chg5': +1,
              'vvix_z': +1, 'skew_z': -1, 'yc_10_2': +1, 'hyg_ret5': +1, 'vix_chg5': +1}
    z = pd.DataFrame(index=f.index)
    for c in f.columns:
        z[c] = ((f[c] - f[c].rolling(252).mean()) / f[c].rolling(252).std()) * orient.get(c, 1)
    bear = z.mean(axis=1); thr = bear.expanding(min_periods=252).quantile(0.85)
    out = []; sidx = list(spy.index); k = 252; H10 = 10
    while k < len(sidx) - H10:
        d = sidx[k]
        if d in bear.index and not np.isnan(thr.get(d, np.nan)) and bear.loc[d] >= thr.loc[d]:
            entry = spy.iloc[k]; ret = (spy.iloc[k + H10] / entry - 1) - COST
            out.append({"entry_date": d, "exit_date": sidx[k + H10], "strat": "fear-dip",
                        "ticker": "SPY", "entry": entry, "ret": ret})
            k += H10
        else:
            k += 1
    return out


def simulate(trades, label, since=None):
    trades = sorted(trades, key=lambda t: t["entry_date"])
    if since is not None:
        trades = [t for t in trades if t["entry_date"] >= since]
    cap = START_CAPITAL; peak = cap; mdd = 0.0; wins = 0
    print(f"\n{'='*92}\n{label}  (시작자본 ${START_CAPITAL:,.0f})\n{'='*92}")
    print(f"{'진입일':>11} {'청산일':>11} {'전략':>10} {'종목':>5} {'사이즈':>6} "
          f"{'수익률':>7} {'손익($)':>9} {'자본($)':>10}")
    for t in trades:
        szinfo = position_size_pct(close[t["ticker"]].dropna().loc[:t["entry_date"]])
        size = szinfo["sizePct"] or 0.0
        pnl = cap * size * t["ret"]
        cap += pnl
        peak = max(peak, cap); mdd = min(mdd, cap / peak - 1)
        if t["ret"] > 0:
            wins += 1
        print(f"{t['entry_date'].date()!s:>11} {t['exit_date'].date()!s:>11} "
              f"{t['strat']:>10} {t['ticker']:>5} {size*100:>5.0f}% "
              f"{t['ret']*100:>+6.2f}% {pnl:>+9.2f} {cap:>10,.0f}")
    n = len(trades)
    if n:
        print(f"{'-'*92}")
        print(f"거래 {n}건 · 승률 {wins/n:.0%} · 최종자본 ${cap:,.0f} "
              f"(수익률 {(cap/START_CAPITAL-1)*100:+.1f}%) · MDD {mdd*100:.1f}%")
    return cap


if __name__ == "__main__":
    conv = conviction_trades(); fd = feardip_trades()
    combined = conv + fd
    print("\n########## 실전 시뮬레이션 — HOLDOUT 2024+ (OOS) ##########")
    simulate(conv, "A) conviction 단독 (실전 신호만)", HOLDOUT)
    simulate(combined, "B) 올웨더 앙상블 (conviction 실전 + fear-dip 실험)", HOLDOUT)
    # buy & hold SPY benchmark over same window
    spy = close["SPY"].dropna(); spy_h = spy[spy.index >= HOLDOUT]
    bh = (spy_h.iloc[-1] / spy_h.iloc[0] - 1) * 100
    print(f"\n[벤치마크] 같은 기간 SPY 바이앤홀드: {bh:+.1f}%")
