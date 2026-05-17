"""Validate candidate strategy across multiple windows.

Best candidate from prior experiments: K=1, thr=0.65, SL 1.5%.
Compare to production (top-5 thr=0.58) and a few alternatives across:
- Full OOS period (2020-12 onwards)
- Pre-2024 (training-like)
- 2024+ Holdout
- Recent 12m

Run:  python scripts/experiment_final_strategy.py
"""
from __future__ import annotations
import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import config

proba_raw = pd.read_parquet('data/logs/phase3_v3_oos_proba.parquet')['proba']
ohlcv = pd.read_parquet('data/raw/etf_ohlcv.parquet')
close = ohlcv['Close']
low = ohlcv['Low']
high = ohlcv['High']

proba_df = proba_raw.unstack(level=1)
ema_df = proba_df.ewm(span=5).mean()
fri_dates = proba_df.index[proba_df.index.dayofweek == 4]
common = sorted(set(proba_df.columns) & set(close.columns))


def realized(entry_date, ticker, sl_pct, tp_pct, hold=5):
    if ticker not in close.columns: return float('nan')
    try: idx = close.index.get_loc(entry_date)
    except KeyError: return float('nan')
    entry = close[ticker].iloc[idx]
    if pd.isna(entry) or entry <= 0: return float('nan')
    end = min(idx + hold, len(close) - 1)
    for j in range(idx + 1, end + 1):
        if sl_pct:
            dl = low[ticker].iloc[j]
            if not pd.isna(dl) and dl <= entry * (1 - sl_pct):
                return -sl_pct
        if tp_pct:
            dh = high[ticker].iloc[j]
            if not pd.isna(dh) and dh >= entry * (1 + tp_pct):
                return tp_pct
    ex = close[ticker].iloc[end]
    if pd.isna(ex): return float('nan')
    return (ex - entry) / entry


def run_strategy(thr, top_k, sl, tp, date_min=None, date_max=None):
    p = ema_df.loc[fri_dates, common]
    if date_min is not None: p = p[p.index >= date_min]
    if date_max is not None: p = p[p.index < date_max]
    pnls = []; weekly_returns = []
    for fri in p.index:
        elig = p.loc[fri][p.loc[fri] >= thr].sort_values(ascending=False).head(top_k)
        if elig.empty:
            weekly_returns.append(0.0)
            continue
        trade_pnls = []
        for t in elig.index:
            r = realized(fri, t, sl, tp)
            if not pd.isna(r):
                net = r - config.TOTAL_COST_ROUNDTRIP
                pnls.append(net)
                trade_pnls.append(net)
        weekly_returns.append(np.mean(trade_pnls) if trade_pnls else 0.0)
    if not pnls: return None
    pnl_s = pd.Series(pnls)
    wins = pnl_s[pnl_s > 0]; losses = pnl_s[pnl_s <= 0]
    wr = len(wins)/len(pnl_s)
    avg_win = float(wins.mean()) if len(wins) else 0
    avg_loss = float(abs(losses.mean())) if len(losses) else 0
    payoff = avg_win/avg_loss if avg_loss > 0 else float('inf')
    expectancy = wr*avg_win - (1-wr)*avg_loss
    wkly = pd.Series(weekly_returns)
    wkly_active = wkly[wkly != 0]
    sharpe = wkly_active.mean()/wkly_active.std()*np.sqrt(52) if len(wkly_active) > 1 and wkly_active.std() > 0 else 0
    eq = (1 + wkly).cumprod()
    mdd = float(((eq - eq.cummax()) / eq.cummax()).min())
    return {'n': len(pnl_s), 'wr': round(wr, 4),
            'avg_W_pct': round(avg_win*100, 3), 'avg_L_pct': round(avg_loss*100, 3),
            'payoff': round(payoff, 2), 'E_pct': round(expectancy*100, 4),
            'sharpe': round(float(sharpe), 3),
            'mdd_pct': round(mdd*100, 2),
            'total_pct': round((float(eq.iloc[-1]) - 1)*100, 2)}


WINDOWS = [
    ('FULL', pd.Timestamp('2020-01-01'), None),
    ('Pre-2024', pd.Timestamp('2020-01-01'), pd.Timestamp('2024-01-01')),
    ('2024+', pd.Timestamp('2024-01-01'), None),
    ('Last 12m', pd.Timestamp('2025-05-15'), None),
]

STRATEGIES = [
    ('Production (top-5 thr=0.58)', 0.58, 5, None, None),
    ('Production + SL 1.5%', 0.58, 5, 0.015, None),
    ('Production + SL 1.5% TP 3%', 0.58, 5, 0.015, 0.03),
    ('K=3 thr=0.58', 0.58, 3, None, None),
    ('K=3 thr=0.58 + SL 1.5%', 0.58, 3, 0.015, None),
    ('K=1 thr=0.65 (candidate)', 0.65, 1, None, None),
    ('K=1 thr=0.65 + SL 1.5%', 0.65, 1, 0.015, None),
    ('K=1 thr=0.65 + SL 1.5% TP 3%', 0.65, 1, 0.015, 0.03),
    ('K=1 thr=0.58 + SL 1.5%', 0.58, 1, 0.015, None),
    ('K=2 thr=0.62 + SL 1.5%', 0.62, 2, 0.015, None),
]

for win_name, dmin, dmax in WINDOWS:
    print('='*130)
    print(f'WINDOW: {win_name}  ({dmin.date() if dmin else "..."} ~ {dmax.date() if dmax else "now"})')
    print('='*130)
    print(f'{"Strategy":<40}  {"n":>4}  {"WR":>6}  {"Avg_W":>7}  {"Avg_L":>7}  {"Payoff":>7}  {"E%":>8}  {"Sharpe":>7}  {"MDD":>7}  {"Total":>7}')
    for name, thr, k, sl, tp in STRATEGIES:
        s = run_strategy(thr, k, sl, tp, dmin, dmax)
        if s is None or s['n'] < 5:
            print(f'{name:<40}  -- insufficient data --')
            continue
        # Mark valid signals per CLAUDE.md
        valid = s['wr'] > 0 and s['payoff'] >= config.MIN_PAYOFF and s['E_pct'] > 0
        marker = "*" if valid else " "
        print(f'{name:<40}{marker} {s["n"]:>4}  {s["wr"]:>6.2%}  '
              f'{s["avg_W_pct"]:>+6.3f}%  {s["avg_L_pct"]:>+6.3f}%  '
              f'{s["payoff"]:>7.2f}  {s["E_pct"]:>+7.4f}%  '
              f'{s["sharpe"]:>+7.3f}  {s["mdd_pct"]:>+6.2f}%  {s["total_pct"]:>+6.2f}%')
    print()

print('* = passes is_valid() (WR>0, payoff>=1.1, E>0)')
