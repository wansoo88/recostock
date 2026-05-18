"""Test v5 with NO regime gates — let the model's probability do all filtering.

Hypothesis: if v5 internalized the regime via features, removing the
external gates may unlock better WR (less double-filtering).
"""
from __future__ import annotations
import io, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import config

proba_v3 = pd.read_parquet('data/logs/phase3_v3_oos_proba.parquet')['proba']
proba_v5 = pd.read_parquet('data/logs/phase3_v5_oos_proba.parquet')['proba']

ohlcv = pd.read_parquet('data/raw/etf_ohlcv.parquet')
close, low, high = ohlcv['Close'], ohlcv['Low'], ohlcv['High']
LONG_ONLY = config.CORE_ETFS + config.SECTOR_ETFS + ["XLB","XLU","XLP","XLC","IBB"]
spy_close = close['SPY']; spy_sma200 = spy_close.rolling(200).mean()


def realized(d, ticker, sl=0.010, tp=0.030, hold=5):
    if ticker not in close.columns: return float('nan')
    try: idx = close.index.get_loc(d)
    except KeyError: return float('nan')
    entry = close[ticker].iloc[idx]
    if pd.isna(entry) or entry <= 0: return float('nan')
    end = min(idx + hold, len(close) - 1)
    for j in range(idx + 1, end + 1):
        dl, dh = low[ticker].iloc[j], high[ticker].iloc[j]
        if not pd.isna(dl) and dl <= entry * (1 - sl): return -sl
        if not pd.isna(dh) and dh >= entry * (1 + tp): return tp
    ex = close[ticker].iloc[end]
    if pd.isna(ex): return float('nan')
    return (ex - entry) / entry


def run(proba_series, thr, multi_ema, regime_fn, date_min, k=1):
    proba_df = proba_series.unstack(level=1)
    ema3 = proba_df.ewm(span=3).mean()
    ema5 = proba_df.ewm(span=5).mean()
    ema7 = proba_df.ewm(span=7).mean()
    fri = proba_df.index[proba_df.index.dayofweek == 4]
    common = sorted(set(proba_df.columns) & set(close.columns) & set(LONG_ONLY))
    pnls = []
    for d in fri:
        if d < date_min: continue
        if regime_fn and not regime_fn(d): continue
        if multi_ema:
            r3 = ema3.loc[d, common]; r5 = ema5.loc[d, common]; r7 = ema7.loc[d, common]
            mask = (r3 >= thr) & (r5 >= thr) & (r7 >= thr)
            elig = r5[mask].sort_values(ascending=False).head(k)
        else:
            r5 = ema5.loc[d, common]
            elig = r5[r5 >= thr].sort_values(ascending=False).head(k)
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
    return {'n': len(s), 'wr': wr, 'payoff': payoff, 'E_pct': E*100, 'total': s.sum()*100}


def regime_v2_only(d):
    """VIX<20 + SPY>200d (no SKEW/MOVE/VIX9D — those are now model features)."""
    v = config.CONVICTION_VIX_MAX  # 20
    try:
        vix = pd.read_parquet('data/raw/vix.parquet').iloc[:,0]
    except: return True
    s = vix.loc[:d]
    if s.empty or s.iloc[-1] >= v: return False
    if d in spy_sma200.index:
        sma = spy_sma200.loc[d]; sc = spy_close.loc[d]
        if not pd.isna(sma) and sc <= sma: return False
    return True


HOLDOUT = pd.Timestamp('2024-01-01')
FULL = pd.Timestamp('2020-01-01')

print('='*110)
print('v5 model with REDUCED gates — let model probability filter regime')
print('='*110)
print(f'{"Config":<48}  {"n":>4}  {"WR":>7}  {"Payoff":>6}  {"E%":>7}  {"Total":>8}')

for label, proba, multi_ema, regime, thr in [
    ('v3 + conviction_v4 (CURRENT BASELINE)',         proba_v3, True, 'full', 0.65),
    ('v5 + conviction_v4 (all gates)',                proba_v5, True, 'full', 0.65),
    ('v5 + Multi-EMA + VIX<20 only',                  proba_v5, True, 'v2only', 0.65),
    ('v5 + Multi-EMA + NO gates',                     proba_v5, True, None, 0.65),
    ('v5 + Multi-EMA + NO gates + thr=0.70',          proba_v5, True, None, 0.70),
    ('v5 + Multi-EMA + NO gates + thr=0.75',          proba_v5, True, None, 0.75),
    ('v5 + single EMA + NO gates + thr=0.65',         proba_v5, False, None, 0.65),
    ('v5 + single EMA + NO gates + thr=0.70',         proba_v5, False, None, 0.70),
    ('v5 + single EMA + NO gates + thr=0.75',         proba_v5, False, None, 0.75),
]:
    if regime == 'full':
        # Approximate v4 regime via the already-stored function in compare script
        from scripts.compare_v5_vs_v4 import regime_v4 as regime_fn
    elif regime == 'v2only':
        regime_fn = regime_v2_only
    else:
        regime_fn = None
    r = run(proba, thr, multi_ema, regime_fn, HOLDOUT)
    if r is None or r['n'] < 5:
        print(f'{label:<48}  -- few --')
        continue
    valid = "✓" if r['payoff'] >= 1.1 and r['E_pct'] > 0 else " "
    wr70 = "🎯" if r['wr'] >= 0.70 else " "
    print(f'{label:<48} {valid}{wr70} {r["n"]:>4}  {r["wr"]:>7.2%}  '
          f'{r["payoff"]:>6.2f}  {r["E_pct"]:>+6.3f}%  {r["total"]:>+7.2f}%')
