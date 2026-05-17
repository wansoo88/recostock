"""TP (Take Profit) sweep on the best clean strategy.

Best strategy so far (from REVIEW §9):
  universe: LONG-ONLY (CORE + SECTOR + extras, 14 tickers)
  regime: VIX < 20 AND SPY > 200d SMA
  selection: K=1 by ema_proba, threshold 0.65
  exit: SL 1.5%, hold up to 5 trading days
  Holdout WR 66.67% but Payoff 0.77 (below MIN_PAYOFF 1.1)

Question: does adding TP at various levels lift Payoff above 1.1 while
preserving WR?

Trade-off intuition:
  - Low TP (e.g. +1%): high hit rate but small wins → WR up, Payoff down
  - High TP (e.g. +5%): rare hits but big wins → WR down, Payoff up
  - Sweet spot depends on hold period (5 days) and intraday volatility

Run:  python scripts/experiment_tp_sweep.py
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

# Regime indicators
spy_close = close['SPY']
spy_sma200 = spy_close.rolling(200).mean()
spy_above_sma = (spy_close > spy_sma200).reindex(fri_dates).ffill()
vix_fri = vix.reindex(fri_dates).ffill()

# Universes
LONG_ONLY = config.CORE_ETFS + config.SECTOR_ETFS + ["XLB", "XLU", "XLP", "XLC", "IBB"]


def realized(entry_date, ticker, sl_pct, tp_pct, hold=5):
    """Realized return with SL/TP. Pessimistic SL (uses daily Low),
    pessimistic TP (uses daily High but assumes fill at TP price not better)."""
    if ticker not in close.columns: return float('nan')
    try: idx = close.index.get_loc(entry_date)
    except KeyError: return float('nan')
    entry = close[ticker].iloc[idx]
    if pd.isna(entry) or entry <= 0: return float('nan')
    end = min(idx + hold, len(close) - 1)
    for j in range(idx + 1, end + 1):
        # Check both SL and TP each day in order: SL first (more pessimistic)
        # If both hit on same day, we assume SL (worst case for trader)
        dl = low[ticker].iloc[j]
        dh = high[ticker].iloc[j]
        if sl_pct is not None and not pd.isna(dl) and dl <= entry * (1 - sl_pct):
            return -sl_pct
        if tp_pct is not None and not pd.isna(dh) and dh >= entry * (1 + tp_pct):
            return tp_pct
    ex = close[ticker].iloc[end]
    if pd.isna(ex): return float('nan')
    return (ex - entry) / entry


def backtest(allowed_tickers, regime_filter, thr, k, sl, tp, date_min=None):
    universe = [t for t in allowed_tickers if t in ema_df.columns and t in close.columns]
    ema_sub = ema_df.loc[fri_dates, universe]
    pnls = []
    weekly_pnls = []
    for fri in ema_sub.index:
        if date_min is not None and fri < date_min: continue
        if regime_filter and not regime_filter(fri):
            weekly_pnls.append(0.0)
            continue
        elig = ema_sub.loc[fri][ema_sub.loc[fri] >= thr].sort_values(ascending=False).head(k)
        wkly_trade = []
        for t in elig.index:
            r = realized(fri, t, sl, tp)
            if not pd.isna(r):
                net = r - config.TOTAL_COST_ROUNDTRIP
                pnls.append(net)
                wkly_trade.append(net)
        weekly_pnls.append(np.mean(wkly_trade) if wkly_trade else 0.0)
    if not pnls: return None
    s = pd.Series(pnls)
    wins = s[s > 0]; losses = s[s <= 0]
    wr = (s > 0).mean()
    avg_w = float(wins.mean()) if len(wins) else 0
    avg_l = float(abs(losses.mean())) if len(losses) else 0
    payoff = avg_w/avg_l if avg_l > 0 else float('inf')
    E = wr*avg_w - (1-wr)*avg_l
    # Portfolio (weekly) Sharpe / MDD
    wkly = pd.Series(weekly_pnls)
    active = wkly[wkly != 0]
    sharpe = active.mean()/active.std()*np.sqrt(52) if len(active) > 1 and active.std() > 0 else 0
    eq = (1 + wkly).cumprod()
    mdd = float(((eq - eq.cummax()) / eq.cummax()).min()) if len(eq) else 0
    return {'n': len(s), 'wr': wr, 'avg_w_pct': avg_w*100, 'avg_l_pct': avg_l*100,
            'payoff': payoff, 'E_pct': E*100,
            'sharpe': float(sharpe), 'mdd_pct': mdd*100,
            'total_pct': float(eq.iloc[-1] - 1)*100}


vix_lt20 = lambda d: vix_fri.loc[d] < 20 if d in vix_fri.index else True
spy_up = lambda d: spy_above_sma.loc[d] if d in spy_above_sma.index else True
regime = lambda d: vix_lt20(d) and spy_up(d)

HOLDOUT = pd.Timestamp('2024-01-01')


def print_row(label, r):
    if r is None or r['n'] < 5:
        print(f'{label:<35}  -- insufficient --')
        return
    valid_mark = "✓" if (r['wr'] > 0 and r['payoff'] >= 1.1 and r['E_pct'] > 0) else " "
    print(f'{label:<35} {valid_mark} {r["n"]:>3}  {r["wr"]:>7.2%}  '
          f'{r["avg_w_pct"]:>+6.2f}%  {r["avg_l_pct"]:>+6.2f}%  '
          f'{r["payoff"]:>6.2f}  {r["E_pct"]:>+7.4f}%  '
          f'{r["sharpe"]:>+6.3f}  {r["mdd_pct"]:>+6.2f}%  {r["total_pct"]:>+7.2f}%')


print('='*128)
print('TP sweep on LONG-ONLY + regime filter + K=1 thr=0.65 + SL 1.5%  (HOLDOUT 2024+)')
print('  ✓ = passes is_valid() (WR>0, payoff>=1.1, E>0)')
print('='*128)
print(f'{"Strategy":<35}    {"n":>3}  {"WR":>7}  {"Avg_W":>7}  {"Avg_L":>7}  '
      f'{"Payoff":>6}  {"E%":>8}  {"Sharpe":>6}  {"MDD":>7}  {"Total":>7}')

for tp in [None, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
    label = f'  SL 1.5% / TP {tp*100:.1f}%' if tp else '  SL 1.5% / no TP (baseline)'
    r = backtest(LONG_ONLY, regime, 0.65, 1, 0.015, tp, HOLDOUT)
    print_row(label, r)


print()
print('='*128)
print('Same sweep, FULL period (2020-12+) — for stability check')
print('='*128)
print(f'{"Strategy":<35}    {"n":>3}  {"WR":>7}  {"Avg_W":>7}  {"Avg_L":>7}  '
      f'{"Payoff":>6}  {"E%":>8}  {"Sharpe":>6}  {"MDD":>7}  {"Total":>7}')

for tp in [None, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
    label = f'  SL 1.5% / TP {tp*100:.1f}%' if tp else '  SL 1.5% / no TP (baseline)'
    r = backtest(LONG_ONLY, regime, 0.65, 1, 0.015, tp, None)
    print_row(label, r)


# Sensitivity: what if SL is varied alongside TP?
print()
print('='*128)
print('SL × TP grid (HOLDOUT 2024+, LONG-ONLY + regime + K=1 thr=0.65)')
print('  (showing only configs that pass is_valid)')
print('='*128)
print(f'{"SL":>5}  {"TP":>5}  {"n":>3}  {"WR":>7}  {"Payoff":>6}  {"E%":>7}  {"Sharpe":>6}  {"MDD":>7}  {"Total":>7}')
for sl in [0.010, 0.015, 0.020, 0.025]:
    for tp in [0.02, 0.025, 0.03, 0.04, 0.05, 0.06]:
        r = backtest(LONG_ONLY, regime, 0.65, 1, sl, tp, HOLDOUT)
        if r is None or r['n'] < 10: continue
        valid = r['wr'] > 0 and r['payoff'] >= 1.1 and r['E_pct'] > 0
        if not valid: continue
        print(f'{sl*100:>4.1f}%  {tp*100:>4.1f}%  {r["n"]:>3}  {r["wr"]:>7.2%}  '
              f'{r["payoff"]:>6.2f}  {r["E_pct"]:>+6.4f}%  '
              f'{r["sharpe"]:>+6.3f}  {r["mdd_pct"]:>+6.2f}%  {r["total_pct"]:>+7.2f}%')
