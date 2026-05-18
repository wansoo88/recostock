"""Apply identical conviction_v4 strategy to v3 and v5 OOS proba.

If v5 produces materially higher Holdout WR (with payoff >= 1.1), it
becomes the new production model. If WR is comparable or lower, we
keep v3 to avoid churn.

Compares both Full and Holdout windows for stability.
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

# Load both OOS proba sets
proba_v3 = pd.read_parquet('data/logs/phase3_v3_oos_proba.parquet')['proba']
proba_v5 = pd.read_parquet('data/logs/phase3_v5_oos_proba.parquet')['proba']

ohlcv = pd.read_parquet('data/raw/etf_ohlcv.parquet')
close = ohlcv['Close']
low = ohlcv['Low']
high = ohlcv['High']
vix = pd.read_parquet('data/raw/vix.parquet').iloc[:, 0]
vix9d = pd.read_parquet('data/raw/macro/vix9d.parquet').iloc[:, 0]
skew = pd.read_parquet('data/raw/macro/skew.parquet').iloc[:, 0]
move = pd.read_parquet('data/raw/macro/move.parquet').iloc[:, 0]

LONG_ONLY = config.CORE_ETFS + config.SECTOR_ETFS + ["XLB","XLU","XLP","XLC","IBB"]
spy_close = close['SPY']; spy_sma200 = spy_close.rolling(200).mean()


def zscore(s, w=60):
    return (s - s.rolling(w).mean()) / s.rolling(w).std()

skew_z = zscore(skew)
move_z = zscore(move)


def realized(d, ticker, sl=0.010, tp=0.030, hold=5):
    if ticker not in close.columns: return float('nan')
    try: idx = close.index.get_loc(d)
    except KeyError: return float('nan')
    entry = close[ticker].iloc[idx]
    if pd.isna(entry) or entry <= 0: return float('nan')
    end = min(idx + hold, len(close) - 1)
    for j in range(idx + 1, end + 1):
        dl = low[ticker].iloc[j]; dh = high[ticker].iloc[j]
        if not pd.isna(dl) and dl <= entry * (1 - sl): return -sl
        if not pd.isna(dh) and dh >= entry * (1 + tp): return tp
    ex = close[ticker].iloc[end]
    if pd.isna(ex): return float('nan')
    return (ex - entry) / entry


def at_or_before(s, d):
    s2 = s.loc[:d]
    return s2.iloc[-1] if not s2.empty else None


def regime_v4(d):
    v = at_or_before(vix, d)
    if v is None or v >= 20.0: return False
    sma = at_or_before(spy_sma200, d); sc = at_or_before(spy_close, d)
    if sma is None or sc is None or pd.isna(sma) or sc <= sma: return False
    v9 = at_or_before(vix9d, d)
    if v9 is None or v9 / v >= 1.0: return False
    sz = at_or_before(skew_z, d)
    if sz is None or pd.isna(sz) or sz >= 1.0: return False
    mz = at_or_before(move_z, d)
    if mz is None or pd.isna(mz) or mz >= 1.0: return False
    return True


def conviction_v4_backtest(proba_series, date_min, label):
    proba_df = proba_series.unstack(level=1)
    ema3 = proba_df.ewm(span=3).mean()
    ema5 = proba_df.ewm(span=5).mean()
    ema7 = proba_df.ewm(span=7).mean()
    fri_dates = proba_df.index[proba_df.index.dayofweek == 4]
    common = sorted(set(proba_df.columns) & set(close.columns) & set(LONG_ONLY))

    pnls = []
    for d in fri_dates:
        if d < date_min: continue
        if not regime_v4(d): continue
        r3 = ema3.loc[d, common]; r5 = ema5.loc[d, common]; r7 = ema7.loc[d, common]
        mask = (r3 >= 0.65) & (r5 >= 0.65) & (r7 >= 0.65)
        elig = r5[mask].sort_values(ascending=False).head(1)
        for t in elig.index:
            r = realized(d, t)
            if not pd.isna(r):
                pnls.append(r - config.TOTAL_COST_ROUNDTRIP)
    if not pnls:
        return None
    s = pd.Series(pnls)
    wins = s[s > 0]; losses = s[s <= 0]
    wr = (s > 0).mean()
    aw = float(wins.mean()) if len(wins) else 0
    al = float(abs(losses.mean())) if len(losses) else 0
    payoff = aw/al if al > 0 else float('inf')
    E = wr*aw - (1-wr)*al
    return {
        'label': label, 'n': len(s), 'wr': wr, 'payoff': payoff,
        'E_pct': E*100, 'total_pct': s.sum()*100,
    }


print('='*120)
print('Conviction v4 strategy applied to v3 vs v5 OOS proba')
print('Strategy: LONG-ONLY + VIX<20 + SPY>200d + VIX9D/VIX<1.0 + SKEW z<1.0 + MOVE z<1.0')
print('         + Multi-EMA(3,5,7) ≥ 0.65 + K=1 + SL 1.0% + TP 3.0%')
print('='*120)

for window_name, date_min in [
    ('Holdout (2024+)',  pd.Timestamp('2024-01-01')),
    ('Full (2020-12+)',  pd.Timestamp('2020-01-01')),
    ('Recent 12m',       pd.Timestamp('2025-05-15')),
]:
    print(f'\n--- {window_name} ---')
    print(f'{"Model":<12}  {"n":>4}  {"WR":>7}  {"Payoff":>7}  {"E%":>8}  {"Total":>8}')
    for label, p in [('v3 (current)', proba_v3), ('v5 (new)', proba_v5)]:
        r = conviction_v4_backtest(p, date_min, label)
        if r is None:
            print(f'{label:<12}  -- insufficient --')
            continue
        valid = "✓" if r['payoff'] >= 1.1 and r['E_pct'] > 0 else " "
        print(f'{label:<12} {valid} {r["n"]:>4}  {r["wr"]:>7.2%}  '
              f'{r["payoff"]:>7.2f}  {r["E_pct"]:>+7.4f}%  {r["total_pct"]:>+7.2f}%')

print('\n' + '='*120)
print('VERDICT')
print('='*120)
holdout_v3 = conviction_v4_backtest(proba_v3, pd.Timestamp('2024-01-01'), 'v3')
holdout_v5 = conviction_v4_backtest(proba_v5, pd.Timestamp('2024-01-01'), 'v5')
full_v3 = conviction_v4_backtest(proba_v3, pd.Timestamp('2020-01-01'), 'v3')
full_v5 = conviction_v4_backtest(proba_v5, pd.Timestamp('2020-01-01'), 'v5')

if holdout_v5 and holdout_v3:
    wr_delta = (holdout_v5['wr'] - holdout_v3['wr']) * 100
    payoff_delta = holdout_v5['payoff'] - holdout_v3['payoff']
    full_wr_delta = (full_v5['wr'] - full_v3['wr']) * 100 if full_v5 and full_v3 else None
    valid_v5 = holdout_v5['payoff'] >= 1.1 and holdout_v5['E_pct'] > 0
    print(f'Holdout WR: v3={holdout_v3["wr"]:.2%}  v5={holdout_v5["wr"]:.2%}  Δ={wr_delta:+.2f}pp')
    print(f'Holdout Payoff: v3={holdout_v3["payoff"]:.2f}  v5={holdout_v5["payoff"]:.2f}  Δ={payoff_delta:+.2f}')
    if full_wr_delta is not None:
        print(f'Full WR delta: {full_wr_delta:+.2f}pp')
    print(f'v5 is_valid: {valid_v5}')
    if valid_v5 and wr_delta > 0 and full_wr_delta is not None and full_wr_delta >= -3.0:
        print('\n>>> PROMOTE: v5 beats v3 on Holdout WR, stable on Full. Replace production weights.')
    elif wr_delta < 0:
        print('\n>>> KEEP v3: v5 WR is lower. New features did not help.')
    else:
        print('\n>>> AMBIGUOUS: review per-fold AUC and Sharpe before deciding.')
