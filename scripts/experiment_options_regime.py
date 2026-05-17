"""Options market data as regime filter — boost conviction_v2 WR.

Data sources (yfinance, free, no API key):
  ^VIX9D  — 9-day VIX (short-term)
  ^VIX    — 30-day VIX (current)
  ^VIX3M  — 3-month VIX (medium)
  ^SKEW   — CBOE SKEW Index (tail risk pricing)

Derived features (all daily):
  vix_9d_30d_ratio  = VIX9D / VIX   (>1 = backwardation = short-term stress)
  vix_30d_3m_ratio  = VIX / VIX3M   (>1 = inverted = severe stress)
  skew_z_60         = SKEW z-score over 60d (high z = unusual tail risk pricing)

Tests as ADDITIONAL regime filters on top of conviction_v2 baseline:
  baseline: LONG-ONLY + VIX<20 + SPY>200d + K=1 thr=0.65 + Multi-EMA + SL 1%/TP 3%
  Holdout 2024+: WR 63.64%, n=33, Payoff 1.20, Total +16.60%

Hypothesis: adding "VIX term in contango" or "SKEW not extreme" should
filter out hidden-risk days that the simple VIX<20 gate misses.

Run:  python scripts/experiment_options_regime.py
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

CACHE_DIR = Path('data/raw/options')
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_with_cache(symbol: str, years: int = 11) -> pd.Series:
    cache_path = CACHE_DIR / f'{symbol.replace("^","").lower()}.parquet'
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        return df['Close']
    end = date.today()
    start = end - timedelta(days=years * 365)
    df = yf.download(symbol, start=str(start), end=str(end),
                     auto_adjust=True, progress=False, threads=False)
    close = df['Close'].squeeze().dropna()
    close.name = 'Close'
    close.to_frame().to_parquet(cache_path)
    return close


print('Fetching options indices...')
vix = fetch_with_cache('^VIX')
vix9d = fetch_with_cache('^VIX9D')
vix3m = fetch_with_cache('^VIX3M')
skew = fetch_with_cache('^SKEW')
print(f'  VIX:    {len(vix)} bars  latest={vix.iloc[-1]:.2f}')
print(f'  VIX9D:  {len(vix9d)} bars  latest={vix9d.iloc[-1]:.2f}')
print(f'  VIX3M:  {len(vix3m)} bars  latest={vix3m.iloc[-1]:.2f}')
print(f'  SKEW:   {len(skew)} bars  latest={skew.iloc[-1]:.2f}')

# Align all on common index
common_idx = vix.index.intersection(vix9d.index).intersection(vix3m.index).intersection(skew.index)
vix = vix.reindex(common_idx); vix9d = vix9d.reindex(common_idx)
vix3m = vix3m.reindex(common_idx); skew = skew.reindex(common_idx)

# Derived features
vix_9d_30d = (vix9d / vix).rename('vix_9d_30d_ratio')
vix_30d_3m = (vix / vix3m).rename('vix_30d_3m_ratio')
skew_mu60 = skew.rolling(60).mean()
skew_sig60 = skew.rolling(60).std()
skew_z = ((skew - skew_mu60) / skew_sig60).rename('skew_z_60')

print('\nDistribution stats (full sample):')
for s in [vix_9d_30d, vix_30d_3m, skew_z]:
    print(f'  {s.name:<22}: mean={s.mean():.3f}  std={s.std():.3f}  '
          f'p10={s.quantile(0.1):.3f}  p90={s.quantile(0.9):.3f}')


# Load proba + close data
proba_raw = pd.read_parquet('data/logs/phase3_v3_oos_proba.parquet')['proba']
ohlcv = pd.read_parquet('data/raw/etf_ohlcv.parquet')
close = ohlcv['Close']
low = ohlcv['Low']
high = ohlcv['High']

proba_df = proba_raw.unstack(level=1)
ema3 = proba_df.ewm(span=3).mean()
ema5 = proba_df.ewm(span=5).mean()
ema7 = proba_df.ewm(span=7).mean()

LONG_ONLY = config.CORE_ETFS + config.SECTOR_ETFS + ["XLB", "XLU", "XLP", "XLC", "IBB"]
common_tk = sorted(set(proba_df.columns) & set(close.columns) & set(LONG_ONLY))

spy_close = close['SPY']; spy_sma200 = spy_close.rolling(200).mean()
fri_dates = proba_df.index[proba_df.index.dayofweek == 4]


def realized(entry_date, ticker, sl_pct=0.010, tp_pct=0.030, hold=5):
    try: idx = close.index.get_loc(entry_date)
    except KeyError: return float('nan')
    if ticker not in close.columns: return float('nan')
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


def make_regime(vix_max, vix_term_max=None, skew_z_max=None):
    """Return regime_ok(date) lambda."""
    def fn(d):
        # base: VIX < threshold AND SPY > 200d SMA
        v = vix.loc[:d].iloc[-1] if d in vix.index else (vix.loc[:d].iloc[-1] if not vix.loc[:d].empty else None)
        if v is None or v >= vix_max:
            return False
        if d in spy_sma200.index:
            sma = spy_sma200.loc[d]
            sc = spy_close.loc[d]
            if not pd.isna(sma) and sc <= sma:
                return False
        # Optional: VIX term structure (9D/VIX ratio) — backwardation filter
        if vix_term_max is not None:
            r = vix_9d_30d.loc[:d].iloc[-1] if not vix_9d_30d.loc[:d].empty else None
            if r is None or r >= vix_term_max:
                return False
        # Optional: SKEW z-score < threshold (tail risk pricing not extreme)
        if skew_z_max is not None:
            z = skew_z.loc[:d].iloc[-1] if not skew_z.loc[:d].empty else None
            if z is None or pd.isna(z) or z >= skew_z_max:
                return False
        return True
    return fn


def run_v2(regime_fn, date_min):
    """Conviction v2: Multi-EMA + K=1 + given regime."""
    pnls = []; skipped_regime = 0
    for d in fri_dates:
        if d < date_min: continue
        if not regime_fn(d):
            skipped_regime += 1
            continue
        # Multi-EMA confirm
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
    avg_w = float(wins.mean()) if len(wins) else 0
    avg_l = float(abs(losses.mean())) if len(losses) else 0
    payoff = avg_w/avg_l if avg_l > 0 else float('inf')
    E = wr*avg_w - (1-wr)*avg_l
    return {'n': len(s), 'wr': wr, 'avg_w_pct': avg_w*100, 'avg_l_pct': avg_l*100,
            'payoff': payoff, 'E_pct': E*100, 'total_pct': s.sum()*100,
            'skipped_by_regime': skipped_regime}


HOLDOUT = pd.Timestamp('2024-01-01')


def print_row(label, r, ref_wr=0.6364):
    if r is None or r['n'] < 5:
        print(f'{label:<55}  -- insufficient --')
        return
    valid = "✓" if (r['wr'] > 0 and r['payoff'] >= 1.1 and r['E_pct'] > 0) else " "
    wr70 = "🎯" if r['wr'] >= 0.70 else " "
    beat = "↑" if r['wr'] > ref_wr else "—"
    print(f'{label:<55} {valid}{wr70}{beat} n={r["n"]:>3}  WR={r["wr"]:>6.2%}  '
          f'Payoff={r["payoff"]:>5.2f}  E%={r["E_pct"]:>+6.3f}  Total={r["total_pct"]:>+6.2f}%')


print('\n' + '='*125)
print('OPTIONS-DATA REGIME FILTERS on top of conviction_v2 (Multi-EMA, K=1, SL 1%/TP 3%)')
print('Holdout 2024+. ✓ = is_valid passes. 🎯 = WR ≥ 70%. ↑ = WR > baseline 63.64%')
print('='*125)

# Baseline (conviction_v2)
r = run_v2(make_regime(20.0), HOLDOUT)
print_row("BASELINE conviction_v2 (VIX<20 + SPY>200d)", r)

print()
print('--- VIX term structure filter (VIX9D / VIX) ---')
print('(< 1.0 = contango = normal market. ≥ 1.0 = backwardation = stress)')
for thr in [0.95, 0.98, 1.00, 1.05]:
    r = run_v2(make_regime(20.0, vix_term_max=thr), HOLDOUT)
    print_row(f'+ VIX9D/VIX < {thr:.2f}', r)

print()
print('--- SKEW z-score filter ---')
print('(z >= 1 means SKEW notably above 60d average = elevated tail risk)')
for thr in [0.5, 1.0, 1.5, 2.0]:
    r = run_v2(make_regime(20.0, skew_z_max=thr), HOLDOUT)
    print_row(f'+ SKEW z < {thr:.1f}', r)

print()
print('--- Combo: VIX term + SKEW ---')
for vt, sk in [(1.0, 1.5), (0.98, 1.0), (1.0, 1.0), (0.95, 0.5)]:
    r = run_v2(make_regime(20.0, vix_term_max=vt, skew_z_max=sk), HOLDOUT)
    print_row(f'+ VIX9D/VIX<{vt} AND SKEW_z<{sk}', r)

print()
print('--- Full period sanity check (2020-12+, for stability) ---')
for label, vt, sk in [
    ('BASELINE conviction_v2', None, None),
    ('+ VIX9D/VIX < 1.00', 1.00, None),
    ('+ VIX9D/VIX < 0.98', 0.98, None),
    ('+ SKEW z < 1.0', None, 1.0),
    ('+ SKEW z < 1.5', None, 1.5),
    ('+ VIX9D/VIX<1.0 AND SKEW z<1.5', 1.00, 1.5),
]:
    r = run_v2(make_regime(20.0, vix_term_max=vt, skew_z_max=sk), pd.Timestamp('2020-01-01'))
    print_row(label, r)
