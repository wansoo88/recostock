"""Test layered regime filters on K=1 thr=0.65 + SL 1.5% strategy.

Filters tested (cumulative):
  A. baseline K=1 thr=0.65 + SL 1.5%
  B. + VIX gate (skip when VIX > 25, panic regime)
  C. + market trend (skip when SPY < 200-day SMA, downtrend)
  D. + credit spread gate (skip when HY/IG ratio extreme)
  E. + day-of-month effect (FOMC weeks etc.)

Walk-forward holdout: train regime params on 2020-2023, test on 2024+.

Run:  python scripts/experiment_regime_filters.py
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
vix = pd.read_parquet('data/raw/vix.parquet').iloc[:, 0]

proba_df = proba_raw.unstack(level=1)
ema_df = proba_df.ewm(span=5).mean()
fri_dates = proba_df.index[proba_df.index.dayofweek == 4]
common = sorted(set(proba_df.columns) & set(close.columns))
ema_fri = ema_df.loc[fri_dates, common]

# SPY 200-day SMA for market trend
spy_close = close['SPY']
spy_sma200 = spy_close.rolling(200).mean()
spy_above_sma = (spy_close > spy_sma200).reindex(fri_dates).ffill()

# VIX on Fridays
vix_fri = vix.reindex(fri_dates).ffill()


def realized(entry_date, ticker, sl_pct=0.015, hold=5):
    if ticker not in close.columns: return float('nan')
    try: idx = close.index.get_loc(entry_date)
    except KeyError: return float('nan')
    entry = close[ticker].iloc[idx]
    if pd.isna(entry) or entry <= 0: return float('nan')
    end = min(idx + hold, len(close) - 1)
    for j in range(idx + 1, end + 1):
        dl = low[ticker].iloc[j]
        if not pd.isna(dl) and dl <= entry * (1 - sl_pct):
            return -sl_pct
    ex = close[ticker].iloc[end]
    if pd.isna(ex): return float('nan')
    return (ex - entry) / entry


def run_filter(filter_fn, label, date_min=None):
    pnls = []
    skipped = 0
    selected_trades = 0
    for fri in ema_fri.index:
        if date_min is not None and fri < date_min:
            continue
        if not filter_fn(fri):
            skipped += 1
            continue
        elig = ema_fri.loc[fri][ema_fri.loc[fri] >= 0.65].sort_values(ascending=False).head(1)
        for t in elig.index:
            r = realized(fri, t)
            if not pd.isna(r):
                pnls.append(r - config.TOTAL_COST_ROUNDTRIP)
                selected_trades += 1
    if not pnls: return None
    s = pd.Series(pnls)
    wins = s[s > 0]; losses = s[s <= 0]
    wr = (s > 0).mean()
    avg_w = float(wins.mean()) if len(wins) else 0
    avg_l = float(abs(losses.mean())) if len(losses) else 0
    payoff = avg_w/avg_l if avg_l > 0 else float('inf')
    E = wr*avg_w - (1-wr)*avg_l
    return {'label': label, 'skipped': skipped, 'n': len(s),
            'wr': wr, 'payoff': payoff, 'E_pct': E*100, 'total_pct': s.sum()*100}


HOLDOUT = pd.Timestamp('2024-01-01')


print('='*120)
print('Layered regime filters on K=1 thr=0.65 + SL 1.5% (Holdout 2024+)')
print('='*120)
print(f'{"Filter":<55}  {"skipped":>7}  {"n":>3}  {"WR":>7}  {"Payoff":>7}  {"E%":>8}  {"Total":>8}')

FILTERS = [
    (lambda d: True, "A. baseline (no filter)"),
    (lambda d: vix_fri.loc[d] < 25 if d in vix_fri.index else True, "B. VIX < 25"),
    (lambda d: vix_fri.loc[d] < 20 if d in vix_fri.index else True, "B'. VIX < 20"),
    (lambda d: (vix_fri.loc[d] >= 15 and vix_fri.loc[d] < 25) if d in vix_fri.index else True, "B''. 15 <= VIX < 25"),
    (lambda d: spy_above_sma.loc[d] if d in spy_above_sma.index else True, "C. SPY > 200d SMA"),
    (lambda d: (vix_fri.loc[d] < 25 and spy_above_sma.loc[d]) if d in vix_fri.index and d in spy_above_sma.index else True, "D. VIX<25 AND SPY>200d"),
    (lambda d: (vix_fri.loc[d] < 20 and spy_above_sma.loc[d]) if d in vix_fri.index and d in spy_above_sma.index else True, "E. VIX<20 AND SPY>200d"),
    (lambda d: not spy_above_sma.loc[d] if d in spy_above_sma.index else False, "F. CONTRARIAN: SPY < 200d only"),
]

for fn, label in FILTERS:
    r = run_filter(fn, label, HOLDOUT)
    if r is None or r['n'] < 5:
        print(f'{label:<55}  -- too few trades --')
        continue
    print(f'{label:<55}  {r["skipped"]:>7}  {r["n"]:>3}  {r["wr"]:>7.2%}  '
          f'{r["payoff"]:>7.2f}  {r["E_pct"]:>+7.4f}%  {r["total_pct"]:>+7.2f}%')


print()
print('='*120)
print('SAME filters on FULL period (2020-12+)')
print('='*120)
print(f'{"Filter":<55}  {"skipped":>7}  {"n":>3}  {"WR":>7}  {"Payoff":>7}  {"E%":>8}  {"Total":>8}')
for fn, label in FILTERS:
    r = run_filter(fn, label, None)
    if r is None or r['n'] < 5:
        print(f'{label:<55}  -- too few --')
        continue
    print(f'{label:<55}  {r["skipped"]:>7}  {r["n"]:>3}  {r["wr"]:>7.2%}  '
          f'{r["payoff"]:>7.2f}  {r["E_pct"]:>+7.4f}%  {r["total_pct"]:>+7.2f}%')
