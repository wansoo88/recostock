"""Round 4 experiments: 4 untried levers vs conviction_v1 baseline.

All compared on identical walk-forward holdout (2024-01 ~ 2026-05).
Baseline: conviction_v1 = LONG-ONLY + VIX<20 + SPY>200d + K=1 thr=0.65 +
                          fixed SL 1.0% / TP 3.0%
Holdout: WR 58.33%, Payoff 1.20, Sharpe 1.67, Total +13.22%

Experiments
-----------
A. Multi-EMA confirmation
   Require proba_ema3 >= 0.65 AND proba_ema5 >= 0.65 AND proba_ema7 >= 0.65.
   Hypothesis: cross-timeframe agreement filters noisy signals → higher WR.

B. ATR-based dynamic SL/TP
   Replace fixed 1%/3% with 1×ATR / 3×ATR (21-day rolling ATR).
   Hypothesis: adapts to volatility regime; calm days avoid premature stops.

C. Entry day-of-week variation
   Current: Friday entry. Test Monday and Wednesday entry on same proba data.
   Hypothesis: weekly seasonality may favor certain entry days.

D. Tighter VIX band (15-20)
   Current: VIX < 20. Test 15 <= VIX < 20 sweet spot.
   Hypothesis: extreme low-vol regime (<15) is mean-reversion territory,
   may degrade trend-following signal.

Run:  python scripts/experiment_round4.py
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

# Data
proba_raw = pd.read_parquet('data/logs/phase3_v3_oos_proba.parquet')['proba']
ohlcv = pd.read_parquet('data/raw/etf_ohlcv.parquet')
close = ohlcv['Close']
low = ohlcv['Low']
high = ohlcv['High']
vix = pd.read_parquet('data/raw/vix.parquet').iloc[:, 0]

proba_df = proba_raw.unstack(level=1)
ema3 = proba_df.ewm(span=3).mean()
ema5 = proba_df.ewm(span=5).mean()
ema7 = proba_df.ewm(span=7).mean()

LONG_ONLY = config.CORE_ETFS + config.SECTOR_ETFS + ["XLB", "XLU", "XLP", "XLC", "IBB"]
common = sorted(set(proba_df.columns) & set(close.columns) & set(LONG_ONLY))

# Regime indicators
spy_close = close['SPY']
spy_sma200 = spy_close.rolling(200).mean()

# Per-ticker ATR(21)
def compute_atr(t, period=21):
    """ATR using True Range = max(H-L, |H-C_prev|, |L-C_prev|)."""
    if t not in close.columns or t not in high.columns or t not in low.columns:
        return None
    c = close[t]
    h = high[t]
    l = low[t]
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean() / c  # ATR as % of close

atr_pct = {t: compute_atr(t) for t in common}


def realized_fixed(entry_date, ticker, sl_pct, tp_pct, hold=5):
    """Fixed-SL/TP realization."""
    if ticker not in close.columns: return float('nan')
    try: idx = close.index.get_loc(entry_date)
    except KeyError: return float('nan')
    entry = close[ticker].iloc[idx]
    if pd.isna(entry) or entry <= 0: return float('nan')
    end = min(idx + hold, len(close) - 1)
    for j in range(idx + 1, end + 1):
        dl = low[ticker].iloc[j]
        dh = high[ticker].iloc[j]
        if sl_pct and not pd.isna(dl) and dl <= entry * (1 - sl_pct):
            return -sl_pct
        if tp_pct and not pd.isna(dh) and dh >= entry * (1 + tp_pct):
            return tp_pct
    ex = close[ticker].iloc[end]
    if pd.isna(ex): return float('nan')
    return (ex - entry) / entry


def realized_atr(entry_date, ticker, sl_mult, tp_mult, hold=5):
    """ATR-based SL/TP. sl_mult and tp_mult are multipliers on entry-day ATR."""
    if ticker not in close.columns: return float('nan')
    if atr_pct.get(ticker) is None: return float('nan')
    try: idx = close.index.get_loc(entry_date)
    except KeyError: return float('nan')
    entry = close[ticker].iloc[idx]
    if pd.isna(entry) or entry <= 0: return float('nan')
    a = atr_pct[ticker].iloc[idx]
    if pd.isna(a) or a <= 0: return float('nan')
    sl_pct = sl_mult * a
    tp_pct = tp_mult * a
    end = min(idx + hold, len(close) - 1)
    for j in range(idx + 1, end + 1):
        dl = low[ticker].iloc[j]
        dh = high[ticker].iloc[j]
        if not pd.isna(dl) and dl <= entry * (1 - sl_pct):
            return -sl_pct
        if not pd.isna(dh) and dh >= entry * (1 + tp_pct):
            return tp_pct
    ex = close[ticker].iloc[end]
    if pd.isna(ex): return float('nan')
    return (ex - entry) / entry


def run_strategy(ema_for_selection, entry_dayofweek, sl_pct, tp_pct,
                 regime_fn, mode='fixed', sl_mult=None, tp_mult=None,
                 thr=0.65, k=1, multi_ema_confirm=False,
                 date_min=None):
    """Generic backtest. ema_for_selection: DataFrame indexed (date, columns=ticker).
    multi_ema_confirm: if True, also require ema3, ema5, ema7 all >= thr."""
    entry_dates = ema_for_selection.index[ema_for_selection.index.dayofweek == entry_dayofweek]
    if date_min is not None:
        entry_dates = entry_dates[entry_dates >= date_min]

    pnls = []; weeks = 0
    for d in entry_dates:
        if regime_fn and not regime_fn(d):
            continue
        # Selection
        row = ema_for_selection.loc[d, common]
        if multi_ema_confirm:
            r3 = ema3.loc[d, common]; r5 = ema5.loc[d, common]; r7 = ema7.loc[d, common]
            eligible_mask = (r3 >= thr) & (r5 >= thr) & (r7 >= thr)
            eligible = row[eligible_mask].sort_values(ascending=False).head(k)
        else:
            eligible = row[row >= thr].sort_values(ascending=False).head(k)
        if eligible.empty: continue
        weeks += 1
        for t in eligible.index:
            if mode == 'fixed':
                r = realized_fixed(d, t, sl_pct, tp_pct)
            else:
                r = realized_atr(d, t, sl_mult, tp_mult)
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
            'n_weeks_traded': weeks}


def vix_under_20(d):
    return (vix.loc[:d].iloc[-1] if d not in vix.index else vix.loc[d]) < 20

def vix_15_20(d):
    v = vix.loc[:d].iloc[-1] if d not in vix.index else vix.loc[d]
    return 15 <= v < 20

def spy_uptrend(d):
    if d not in spy_sma200.index: return True
    sma = spy_sma200.loc[d]
    sc = spy_close.loc[d] if d in spy_close.index else spy_close.loc[:d].iloc[-1]
    if pd.isna(sma): return True
    return sc > sma

def regime_default(d):
    return vix_under_20(d) and spy_uptrend(d)

def regime_tight_vix(d):
    return vix_15_20(d) and spy_uptrend(d)


HOLDOUT = pd.Timestamp('2024-01-01')


def print_row(label, r):
    if r is None or r['n'] < 5:
        print(f'{label:<55}  -- insufficient --')
        return
    valid = "✓" if (r['wr'] > 0 and r['payoff'] >= 1.1 and r['E_pct'] > 0) else " "
    wr70 = "🎯" if r['wr'] >= 0.70 else " "
    print(f'{label:<55} {valid}{wr70} n={r["n"]:>3}  WR={r["wr"]:>6.2%}  '
          f'W={r["avg_w_pct"]:>+5.2f}%  L={r["avg_l_pct"]:>+5.2f}%  '
          f'Payoff={r["payoff"]:>5.2f}  E%={r["E_pct"]:>+6.3f}  Total={r["total_pct"]:>+6.2f}%')


print('='*120)
print('ROUND 4 EXPERIMENTS — Holdout 2024-01 ~ 2026-05')
print('Baseline = conviction_v1: LONG-ONLY + VIX<20 + SPY>200d + K=1 thr=0.65 + SL 1.0% / TP 3.0%')
print('Legend: ✓ = passes is_valid() (Payoff≥1.1, E>0). 🎯 = WR ≥ 70%')
print('='*120)

# Baseline
r = run_strategy(ema5, 4, 0.010, 0.030, regime_default, mode='fixed', date_min=HOLDOUT)
print_row("BASELINE conviction_v1 (Friday)", r)

print()
print('--- Experiment A: Multi-EMA confirmation (require EMA-3, 5, 7 all ≥ 0.65) ---')
r = run_strategy(ema5, 4, 0.010, 0.030, regime_default, mode='fixed',
                 multi_ema_confirm=True, date_min=HOLDOUT)
print_row("A1. Multi-EMA + same SL/TP", r)
r = run_strategy(ema5, 4, 0.015, 0.030, regime_default, mode='fixed',
                 multi_ema_confirm=True, date_min=HOLDOUT)
print_row("A2. Multi-EMA + SL 1.5% / TP 3.0%", r)
r = run_strategy(ema5, 4, 0.010, 0.050, regime_default, mode='fixed',
                 multi_ema_confirm=True, date_min=HOLDOUT)
print_row("A3. Multi-EMA + SL 1.0% / TP 5.0%", r)

print()
print('--- Experiment B: ATR-based dynamic SL/TP (1×ATR / 3×ATR) ---')
for sl_m, tp_m in [(1.0, 3.0), (1.5, 3.0), (1.0, 4.0), (0.8, 3.0)]:
    r = run_strategy(ema5, 4, None, None, regime_default, mode='atr',
                     sl_mult=sl_m, tp_mult=tp_m, date_min=HOLDOUT)
    print_row(f"B. SL {sl_m}×ATR / TP {tp_m}×ATR", r)

print()
print('--- Experiment C: Entry day-of-week variation ---')
# 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday
days = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
for dow, dname in days.items():
    r = run_strategy(ema5, dow, 0.010, 0.030, regime_default, mode='fixed', date_min=HOLDOUT)
    print_row(f"C. Entry={dname} + SL 1.0% / TP 3.0%", r)

print()
print('--- Experiment D: Tighter VIX band (15 ≤ VIX < 20) ---')
r = run_strategy(ema5, 4, 0.010, 0.030, regime_tight_vix, mode='fixed', date_min=HOLDOUT)
print_row("D. VIX 15-20 + SL 1.0% / TP 3.0%", r)
r = run_strategy(ema5, 4, 0.015, 0.030, regime_tight_vix, mode='fixed', date_min=HOLDOUT)
print_row("D2. VIX 15-20 + SL 1.5% / TP 3.0%", r)
r = run_strategy(ema5, 4, 0.010, 0.050, regime_tight_vix, mode='fixed', date_min=HOLDOUT)
print_row("D3. VIX 15-20 + SL 1.0% / TP 5.0%", r)

print()
print('--- Combo: best ingredients ---')
# Try Multi-EMA + tight VIX
r = run_strategy(ema5, 4, 0.010, 0.030, regime_tight_vix, mode='fixed',
                 multi_ema_confirm=True, date_min=HOLDOUT)
print_row("Combo1. Multi-EMA + VIX 15-20", r)
# Multi-EMA + Wed entry
r = run_strategy(ema5, 2, 0.010, 0.030, regime_default, mode='fixed',
                 multi_ema_confirm=True, date_min=HOLDOUT)
print_row("Combo2. Multi-EMA + Wed entry", r)
# Multi-EMA + ATR
r = run_strategy(ema5, 4, None, None, regime_default, mode='atr',
                 sl_mult=1.0, tp_mult=3.0, multi_ema_confirm=True, date_min=HOLDOUT)
print_row("Combo3. Multi-EMA + ATR 1×/3×", r)
