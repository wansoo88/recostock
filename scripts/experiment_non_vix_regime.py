"""Non-VIX regime filters — explore cross-asset signals beyond options markets.

Baseline = conviction_v2 (Multi-EMA, no v3 options gates):
    Holdout 2024+: n=33, WR 63.64%, Payoff 1.20, Total +16.60%

Tested indicators (all daily, free):
  ^MOVE       — ICE BofA Bond Volatility (bond stress proxy)
  HYG/LQD     — High-yield / Investment-grade ratio (credit spread)
  10y - 2y    — Yield curve slope (recession indicator)
  Gold/SPY    — Flight-to-quality ratio
  DXY change  — Dollar momentum

Question: do non-VIX/non-SKEW signals add WR improvement?
Run:  python scripts/experiment_non_vix_regime.py
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
import yfinance as yf
from datetime import date, timedelta
import config

CACHE = Path('data/raw/macro')
CACHE.mkdir(parents=True, exist_ok=True)


def fetch(symbol, years=11):
    name = symbol.replace("^","").replace("-","_").lower()
    p = CACHE / f'{name}.parquet'
    if p.exists():
        df = pd.read_parquet(p)
        s = df.iloc[:, 0]
        return s.dropna()
    end = date.today()
    start = end - timedelta(days=years * 365)
    df = yf.download(symbol, start=str(start), end=str(end),
                     auto_adjust=True, progress=False, threads=False)
    close = df['Close'].squeeze().dropna()
    close.name = name
    close.to_frame().to_parquet(p)
    return close


print('Loading market data...')
ohlcv = pd.read_parquet('data/raw/etf_ohlcv.parquet')
close = ohlcv['Close']
low = ohlcv['Low']
high = ohlcv['High']

vix = pd.read_parquet('data/raw/vix.parquet').iloc[:, 0]

# Existing macro
dxy = fetch('DX-Y.NYB')
gold = fetch('GLD')
hyg = fetch('HYG')
lqd = fetch('LQD')
y10 = fetch('^TNX') / 10  # CBOE TNX is *10
y2 = fetch('^IRX') / 10   # IRX is 13-week T-bill, used as 2y proxy

# Fetch NEW indicators
print('Fetching new indicators...')
move = fetch('^MOVE')

# SPY for ratio
spy_close = close['SPY']

# Build derived features (all daily, indexed by dates with all data available)
def common_idx(*series):
    idx = series[0].index
    for s in series[1:]:
        idx = idx.intersection(s.index)
    return idx

# Feature 1: MOVE z-score (60d)
def zscore(s, window=60):
    mu = s.rolling(window).mean()
    sig = s.rolling(window).std()
    return (s - mu) / sig

move_z = zscore(move).rename('move_z')

# Feature 2: HYG/LQD ratio z-score
hyg_lqd = (hyg / lqd)
hyg_lqd_z = zscore(hyg_lqd).rename('hyg_lqd_z')

# Feature 3: Yield curve (10y - 2y)
yield_spread = (y10 - y2).rename('yield_spread')
yield_spread_z = zscore(yield_spread).rename('yield_spread_z')

# Feature 4: Gold / SPY ratio (z-score)
common_gs = common_idx(gold, spy_close)
gold_spy = (gold.reindex(common_gs) / spy_close.reindex(common_gs)).rename('gold_spy')
gold_spy_z = zscore(gold_spy).rename('gold_spy_z')

# Feature 5: DXY change 5d
dxy_chg_5d = dxy.pct_change(5).rename('dxy_chg_5d')

print('\nDistribution stats (full sample):')
for s in [move_z, hyg_lqd_z, yield_spread, gold_spy_z, dxy_chg_5d]:
    s_clean = s.dropna()
    print(f'  {s.name:<18}: mean={s_clean.mean():+.3f}  p10={s_clean.quantile(0.1):+.3f}  '
          f'p50={s_clean.quantile(0.5):+.3f}  p90={s_clean.quantile(0.9):+.3f}')

# Set up backtest
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
        if not pd.isna(dl) and dl <= entry * (1 - sl_pct):
            return -sl_pct
        if not pd.isna(dh) and dh >= entry * (1 + tp_pct):
            return tp_pct
    ex = close[ticker].iloc[end]
    if pd.isna(ex): return float('nan')
    return (ex - entry) / entry


def make_v2_regime(extra_filter=None):
    """conviction_v2 regime (VIX<20 + SPY>200d) + optional extra_filter(d) -> bool"""
    def fn(d):
        v = vix.loc[:d].iloc[-1] if not vix.loc[:d].empty else None
        if v is None or v >= 20.0: return False
        if d in spy_sma200.index:
            sma = spy_sma200.loc[d]
            sc = spy_close.loc[d]
            if not pd.isna(sma) and sc <= sma:
                return False
        if extra_filter is not None and not extra_filter(d):
            return False
        return True
    return fn


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


def at_or_before(s, d):
    return s.loc[:d].iloc[-1] if not s.loc[:d].empty else None


def safe(v, default=None):
    return default if v is None or pd.isna(v) else v


HOLDOUT = pd.Timestamp('2024-01-01')


def print_row(label, r, ref=0.6364):
    if r is None or r['n'] < 5:
        print(f'{label:<58}  -- insufficient --')
        return
    valid = "✓" if (r['wr'] > 0 and r['payoff'] >= 1.1 and r['E_pct'] > 0) else " "
    wr70 = "🎯" if r['wr'] >= 0.70 else " "
    beat = "↑" if r['wr'] > ref else "—"
    print(f'{label:<58} {valid}{wr70}{beat} n={r["n"]:>3}  WR={r["wr"]:>6.2%}  '
          f'Payoff={r["payoff"]:>5.2f}  E%={r["E_pct"]:>+6.3f}  Total={r["total_pct"]:>+6.2f}%')


print('\n' + '='*125)
print('NON-VIX REGIME FILTERS — Holdout 2024+, on top of conviction_v2 (Multi-EMA)')
print('Baseline: n=33, WR 63.64%, Payoff 1.20, Total +16.60%')
print('='*125)

# Baseline
print_row("BASELINE (Multi-EMA only)", run(make_v2_regime(), HOLDOUT))

print('\n--- MOVE z-score filter (bond volatility) ---')
for thr in [0.0, 0.5, 1.0, 1.5]:
    f = lambda d, t=thr: safe(at_or_before(move_z, d), 0) < t
    print_row(f'+ MOVE z < {thr:.1f}', run(make_v2_regime(f), HOLDOUT))

print('\n--- HYG/LQD credit ratio z-score (credit health) ---')
for thr in [-0.5, 0.0, 0.5, 1.0]:
    # Higher = credit healthy; we want positive credit health → require HYG/LQD z > thr
    f = lambda d, t=thr: safe(at_or_before(hyg_lqd_z, d), 0) > t
    print_row(f'+ HYG/LQD z > {thr:+.1f} (credit healthy)', run(make_v2_regime(f), HOLDOUT))

print('\n--- Yield curve absolute (10y - 2y) ---')
for thr in [-0.5, 0.0, 0.5, 1.0]:
    f = lambda d, t=thr: safe(at_or_before(yield_spread, d), -10) > t
    print_row(f'+ (10y - 2y) > {thr:+.1f}% (curve not too inverted)', run(make_v2_regime(f), HOLDOUT))

print('\n--- Gold/SPY ratio z-score (low = risk-on) ---')
for thr in [-0.5, 0.0, 0.5, 1.0]:
    f = lambda d, t=thr: safe(at_or_before(gold_spy_z, d), 10) < t
    print_row(f'+ Gold/SPY z < {thr:+.1f} (no flight to safety)', run(make_v2_regime(f), HOLDOUT))

print('\n--- DXY 5-day change (dollar weakness = risk-on) ---')
for thr in [0.005, 0.01, 0.015, 0.02]:
    f = lambda d, t=thr: safe(at_or_before(dxy_chg_5d, d), 1) < t
    print_row(f'+ DXY chg 5d < {thr*100:.1f}% (dollar not surging)', run(make_v2_regime(f), HOLDOUT))

print('\n--- Combinations of best from each category ---')
combos = [
    ('+ MOVE z < 1.0',
     lambda d: safe(at_or_before(move_z, d), 0) < 1.0),
    ('+ HYG/LQD z > 0',
     lambda d: safe(at_or_before(hyg_lqd_z, d), 0) > 0.0),
    ('+ MOVE z < 1.0 AND HYG/LQD z > 0',
     lambda d: safe(at_or_before(move_z, d), 0) < 1.0 and
               safe(at_or_before(hyg_lqd_z, d), 0) > 0.0),
    ('+ MOVE z < 0.5 AND HYG/LQD z > 0',
     lambda d: safe(at_or_before(move_z, d), 0) < 0.5 and
               safe(at_or_before(hyg_lqd_z, d), 0) > 0.0),
    ('+ MOVE z < 1.0 AND Gold/SPY z < 0.5',
     lambda d: safe(at_or_before(move_z, d), 0) < 1.0 and
               safe(at_or_before(gold_spy_z, d), 10) < 0.5),
    ('+ MOVE z < 1.0 AND yield_spread > 0',
     lambda d: safe(at_or_before(move_z, d), 0) < 1.0 and
               safe(at_or_before(yield_spread, d), -10) > 0.0),
]
for label, f in combos:
    print_row(label, run(make_v2_regime(f), HOLDOUT))

print('\n--- Full period sanity check (best candidates only) ---')
best = [
    ('BASELINE Multi-EMA only', None),
    ('+ MOVE z < 1.0',
     lambda d: safe(at_or_before(move_z, d), 0) < 1.0),
    ('+ HYG/LQD z > 0',
     lambda d: safe(at_or_before(hyg_lqd_z, d), 0) > 0.0),
    ('+ MOVE z < 1.0 AND HYG/LQD z > 0',
     lambda d: safe(at_or_before(move_z, d), 0) < 1.0 and
               safe(at_or_before(hyg_lqd_z, d), 0) > 0.0),
]
for label, f in best:
    print_row(label, run(make_v2_regime(f), pd.Timestamp('2020-01-01')), ref=0.5246)
