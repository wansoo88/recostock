"""Test universe-restricted strategies to push WR higher.

Hypothesis: inverse and volatility ETFs (SH, PSQ, DOG, VXX) have different
return characteristics and may degrade model performance. Restricting to
long-only sector and core ETFs may improve WR.

Strategies:
1. K=1 thr=0.65 + SL 1.5% over FULL universe (baseline)
2. Same, but exclude inverse + VXX
3. Same, but only CORE (SPY/QQQ/DIA)
4. Same, but only SECTOR (XLK/XLF/etc.)
5. Same, but only the top-WR tickers from per-ticker analysis

Run:  python scripts/experiment_universe_filter.py
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

spy_close = close['SPY']
spy_sma200 = spy_close.rolling(200).mean()
spy_above_sma = (spy_close > spy_sma200).reindex(fri_dates).ffill()
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


def backtest(allowed_tickers, regime_filter=None, date_min=None, thr=0.65, k=1):
    universe = [t for t in allowed_tickers if t in ema_df.columns and t in close.columns]
    ema_sub = ema_df.loc[fri_dates, universe]
    pnls = []
    for fri in ema_sub.index:
        if date_min is not None and fri < date_min: continue
        if regime_filter and not regime_filter(fri): continue
        elig = ema_sub.loc[fri][ema_sub.loc[fri] >= thr].sort_values(ascending=False).head(k)
        for t in elig.index:
            r = realized(fri, t)
            if not pd.isna(r):
                pnls.append(r - config.TOTAL_COST_ROUNDTRIP)
    if not pnls: return None
    s = pd.Series(pnls)
    wins = s[s > 0]; losses = s[s <= 0]
    wr = (s > 0).mean()
    avg_w = float(wins.mean()) if len(wins) else 0
    avg_l = float(abs(losses.mean())) if len(losses) else 0
    payoff = avg_w/avg_l if avg_l > 0 else float('inf')
    E = wr*avg_w - (1-wr)*avg_l
    return {'n': len(s), 'wr': wr, 'payoff': payoff,
            'E_pct': E*100, 'total_pct': s.sum()*100}


ALL = config.CORE_ETFS + config.SECTOR_ETFS + config.INVERSE_ETFS + ["XLB","XLU","XLP","XLC","IBB"] + config.VOLATILITY_ETFS
LONG_ONLY = config.CORE_ETFS + config.SECTOR_ETFS + ["XLB","XLU","XLP","XLC","IBB"]
CORE = config.CORE_ETFS
SECTOR = config.SECTOR_ETFS + ["XLB","XLU","XLP","XLC","IBB"]
# Top per per_ticker_recent12m: XLI, XLE, SPY, XLK, QQQ
TOP5 = ["XLI", "XLE", "SPY", "XLK", "QQQ"]
TOP3 = ["SPY", "QQQ", "XLK"]

vix_lt_20 = lambda d: vix_fri.loc[d] < 20 if d in vix_fri.index else True
spy_uptrend = lambda d: spy_above_sma.loc[d] if d in spy_above_sma.index else True
combined_regime = lambda d: vix_lt_20(d) and spy_uptrend(d)

HOLDOUT = pd.Timestamp('2024-01-01')

print('='*120)
print('Universe restriction on K=1 thr=0.65 + SL 1.5% (Holdout 2024+, no regime filter)')
print('='*120)
print(f'{"Universe":<35}  {"size":>4}  {"n":>3}  {"WR":>7}  {"Payoff":>7}  {"E%":>8}  {"Total":>8}')
for name, u in [("FULL (18)", ALL), ("LONG-ONLY (11)", LONG_ONLY),
                ("CORE (3)", CORE), ("SECTOR (11)", SECTOR),
                ("TOP-5 historical WR", TOP5), ("TOP-3 historical WR", TOP3)]:
    r = backtest(u, None, HOLDOUT)
    if r is None: continue
    print(f'{name:<35}  {len(u):>4}  {r["n"]:>3}  {r["wr"]:>7.2%}  '
          f'{r["payoff"]:>7.2f}  {r["E_pct"]:>+7.4f}%  {r["total_pct"]:>+7.2f}%')


print()
print('='*120)
print('Universe × regime filter (VIX<20 AND SPY>200d), Holdout 2024+')
print('='*120)
print(f'{"Universe":<35}  {"size":>4}  {"n":>3}  {"WR":>7}  {"Payoff":>7}  {"E%":>8}  {"Total":>8}')
for name, u in [("FULL", ALL), ("LONG-ONLY", LONG_ONLY),
                ("TOP-5 historical", TOP5), ("TOP-3 historical", TOP3)]:
    r = backtest(u, combined_regime, HOLDOUT)
    if r is None: continue
    print(f'{name:<35}  {len(u):>4}  {r["n"]:>3}  {r["wr"]:>7.2%}  '
          f'{r["payoff"]:>7.2f}  {r["E_pct"]:>+7.4f}%  {r["total_pct"]:>+7.2f}%')


print()
print('='*120)
print('SAME analyses on Full period (sanity check, not for selection)')
print('='*120)
print(f'{"Strategy":<40}  {"n":>3}  {"WR":>7}  {"Payoff":>7}  {"Total":>8}')
for name, u, rf in [("FULL no regime", ALL, None),
                     ("LONG-ONLY no regime", LONG_ONLY, None),
                     ("LONG-ONLY + VIX<20+SPY>200", LONG_ONLY, combined_regime),
                     ("TOP-5 + VIX<20+SPY>200", TOP5, combined_regime),
                     ("TOP-3 + VIX<20+SPY>200", TOP3, combined_regime)]:
    r = backtest(u, rf, None)
    if r is None: continue
    print(f'{name:<40}  {r["n"]:>3}  {r["wr"]:>7.2%}  {r["payoff"]:>7.2f}  {r["total_pct"]:>+7.2f}%')
