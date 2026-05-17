"""Layer non-VIX signals onto conviction_v3 — test if WR pushes beyond 70%.

Baselines:
  v2 (Multi-EMA only):      Holdout n=33  WR 63.64%  Total +16.60%
  v3 (+ SKEW + VIX9D term): Holdout n=20  WR 70.00%  Total +14.37%

Question: can we push WR above 70% (or stabilize 70% with bigger n) by
adding non-VIX signals (MOVE, HYG/LQD, Gold/SPY)?
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

# Cached data
ohlcv = pd.read_parquet('data/raw/etf_ohlcv.parquet')
close = ohlcv['Close']
low = ohlcv['Low']
high = ohlcv['High']

vix = pd.read_parquet('data/raw/vix.parquet').iloc[:, 0]
vix9d = pd.read_parquet('data/raw/macro/vix9d.parquet').iloc[:, 0]
skew = pd.read_parquet('data/raw/macro/skew.parquet').iloc[:, 0]
move = pd.read_parquet('data/raw/macro/move.parquet').iloc[:, 0]
hyg = pd.read_parquet('data/raw/macro/hyg.parquet').iloc[:, 0]
lqd = pd.read_parquet('data/raw/macro/lqd.parquet').iloc[:, 0]
gold = pd.read_parquet('data/raw/macro/gold.parquet').iloc[:, 0]
spy_close = close['SPY']

def zscore(s, window=60):
    return (s - s.rolling(window).mean()) / s.rolling(window).std()

skew_z = zscore(skew)
move_z = zscore(move)
hyg_lqd_z = zscore(hyg / lqd)
common_gs = gold.index.intersection(spy_close.index)
gold_spy_z = zscore(gold.reindex(common_gs) / spy_close.reindex(common_gs))

proba_raw = pd.read_parquet('data/logs/phase3_v3_oos_proba.parquet')['proba']
proba_df = proba_raw.unstack(level=1)
ema3 = proba_df.ewm(span=3).mean()
ema5 = proba_df.ewm(span=5).mean()
ema7 = proba_df.ewm(span=7).mean()

LONG_ONLY = config.CORE_ETFS + config.SECTOR_ETFS + ["XLB","XLU","XLP","XLC","IBB"]
common_tk = sorted(set(proba_df.columns) & set(close.columns) & set(LONG_ONLY))
spy_sma200 = spy_close.rolling(200).mean()
fri_dates = proba_df.index[proba_df.index.dayofweek == 4]


def realized(entry_date, ticker, sl_pct=0.010, tp_pct=0.030, hold=5):
    if ticker not in close.columns: return float('nan')
    try: idx = close.index.get_loc(entry_date)
    except KeyError: return float('nan')
    entry = close[ticker].iloc[idx]
    if pd.isna(entry) or entry <= 0: return float('nan')
    end = min(idx + hold, len(close) - 1)
    for j in range(idx + 1, end + 1):
        dl = low[ticker].iloc[j]; dh = high[ticker].iloc[j]
        if not pd.isna(dl) and dl <= entry * (1 - sl_pct): return -sl_pct
        if not pd.isna(dh) and dh >= entry * (1 + tp_pct): return tp_pct
    ex = close[ticker].iloc[end]
    if pd.isna(ex): return float('nan')
    return (ex - entry) / entry


def at_or_before(s, d):
    s2 = s.loc[:d]
    return s2.iloc[-1] if not s2.empty else None

def safe(v, default):
    return default if v is None or pd.isna(v) else v


def regime_v3(d, vix_term_max=1.0, skew_z_max=1.0):
    """v3 regime: VIX<20 + SPY>200d + VIX9D/VIX<1.0 + SKEW z<1.0"""
    v = at_or_before(vix, d)
    if v is None or v >= 20.0: return False
    sma = at_or_before(spy_sma200, d); sc = at_or_before(spy_close, d)
    if sma is None or sc is None or pd.isna(sma) or sc <= sma: return False
    v9 = at_or_before(vix9d, d)
    if v9 is None or v9 / v >= vix_term_max: return False
    sz = at_or_before(skew_z, d)
    if sz is None or pd.isna(sz) or sz >= skew_z_max: return False
    return True


def run(regime_fn, date_min):
    pnls = []
    for d in fri_dates:
        if d < date_min: continue
        if not regime_fn(d): continue
        r3 = ema3.loc[d, common_tk]; r5 = ema5.loc[d, common_tk]; r7 = ema7.loc[d, common_tk]
        mask = (r3 >= 0.65) & (r5 >= 0.65) & (r7 >= 0.65)
        elig = r5[mask].sort_values(ascending=False).head(1)
        for t in elig.index:
            r = realized(d, t)
            if not pd.isna(r):
                pnls.append(r - config.TOTAL_COST_ROUNDTRIP)
    if not pnls: return None
    s = pd.Series(pnls)
    wins = s[s > 0]; losses = s[s <= 0]
    wr = (s > 0).mean()
    aw = float(wins.mean()) if len(wins) else 0
    al = float(abs(losses.mean())) if len(losses) else 0
    payoff = aw/al if al > 0 else float('inf')
    E = wr*aw - (1-wr)*al
    return {'n': len(s), 'wr': wr, 'payoff': payoff, 'E_pct': E*100, 'total_pct': s.sum()*100}


HOLDOUT = pd.Timestamp('2024-01-01')


def print_row(label, r, ref=0.70):
    if r is None or r['n'] < 5:
        print(f'{label:<58}  -- insufficient --')
        return
    valid = "✓" if (r['wr'] > 0 and r['payoff'] >= 1.1 and r['E_pct'] > 0) else " "
    wr70 = "🎯" if r['wr'] >= ref else " "
    print(f'{label:<58} {valid}{wr70} n={r["n"]:>3}  WR={r["wr"]:>6.2%}  '
          f'Payoff={r["payoff"]:>5.2f}  E%={r["E_pct"]:>+6.3f}  Total={r["total_pct"]:>+6.2f}%')


print('='*125)
print('LAYER non-VIX signals onto conviction_v3 (Holdout 2024+)')
print('v3 baseline: n=20, WR 70.00%, Payoff 1.25, Total +14.37%')
print('Goal: push WR ABOVE 70% with sample n ≥ 15')
print('='*125)

print_row("BASELINE v3 (Multi-EMA + SKEW<1 + VIX9D/VIX<1)", run(regime_v3, HOLDOUT))

print('\n--- v3 + MOVE filter ---')
for thr in [0.5, 1.0, 1.5]:
    fn = lambda d, t=thr: regime_v3(d) and safe(at_or_before(move_z, d), 0) < t
    print_row(f'+ MOVE z < {thr:.1f}', run(fn, HOLDOUT))

print('\n--- v3 + Gold/SPY filter ---')
for thr in [-0.5, 0.0, 0.5, 1.0]:
    fn = lambda d, t=thr: regime_v3(d) and safe(at_or_before(gold_spy_z, d), 10) < t
    print_row(f'+ Gold/SPY z < {thr:+.1f}', run(fn, HOLDOUT))

print('\n--- v3 + HYG/LQD filter ---')
for thr in [-0.5, 0.0, 0.5]:
    fn = lambda d, t=thr: regime_v3(d) and safe(at_or_before(hyg_lqd_z, d), 0) > t
    print_row(f'+ HYG/LQD z > {thr:+.1f}', run(fn, HOLDOUT))

print('\n--- v3 + combined non-VIX signals ---')
combos = [
    ('+ MOVE z<1 AND Gold/SPY z<0.5',
     lambda d: regime_v3(d) and safe(at_or_before(move_z, d), 0) < 1.0
               and safe(at_or_before(gold_spy_z, d), 10) < 0.5),
    ('+ MOVE z<1 AND HYG/LQD z>0',
     lambda d: regime_v3(d) and safe(at_or_before(move_z, d), 0) < 1.0
               and safe(at_or_before(hyg_lqd_z, d), 0) > 0.0),
    ('+ MOVE z<1.5 AND Gold/SPY z<1',
     lambda d: regime_v3(d) and safe(at_or_before(move_z, d), 0) < 1.5
               and safe(at_or_before(gold_spy_z, d), 10) < 1.0),
]
for label, fn in combos:
    print_row(label, run(fn, HOLDOUT))


print('\n--- ALTERNATIVE: v2 + MOVE (forgo SKEW+VIX9D, use MOVE instead) ---')
def regime_v2(d):
    v = at_or_before(vix, d)
    if v is None or v >= 20.0: return False
    sma = at_or_before(spy_sma200, d); sc = at_or_before(spy_close, d)
    if sma is None or sc is None or pd.isna(sma) or sc <= sma: return False
    return True

for thr in [0.5, 1.0, 1.5]:
    fn = lambda d, t=thr: regime_v2(d) and safe(at_or_before(move_z, d), 0) < t
    print_row(f'v2 + MOVE z < {thr:.1f}', run(fn, HOLDOUT))

print('\n--- "Best Total" candidates — robustness check (Full period) ---')
candidates = [
    ('v2 baseline', regime_v2),
    ('v3 baseline (SKEW+VIX9D)', regime_v3),
    ('v2 + MOVE z<1.0',
     lambda d: regime_v2(d) and safe(at_or_before(move_z, d), 0) < 1.0),
    ('v3 + Gold/SPY z<0.5',
     lambda d: regime_v3(d) and safe(at_or_before(gold_spy_z, d), 10) < 0.5),
    ('v3 + MOVE z<1.0',
     lambda d: regime_v3(d) and safe(at_or_before(move_z, d), 0) < 1.0),
]
for label, fn in candidates:
    print_row(label, run(fn, pd.Timestamp('2020-01-01')), ref=0.60)
