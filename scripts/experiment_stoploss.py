"""Experiment: cap losses with intra-week stop-loss to fix payoff.

5d forward return assumes hold-to-Friday. But the production system has
TP/SL in signal/generator.py — losses could be cut early. We simulate
the effect of a tight stop using daily OHLC data: if at any point during
the 5-day hold the position is down more than SL, exit at SL.

Caveats:
- Uses 5d Close-only return is_a worst case; in reality close-based daily
  monitoring may miss intra-day spikes. For an OHLC-based simulation we
  use intraday Low for stop checks.
- This is still daily, not intraday — slippage on stop fills underestimated.
- The 0.25% cost is added at entry; if stopped out, full cost still charged.

Run:  python scripts/experiment_stoploss.py
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

# Need both Close and Low for stop-loss
is_multi = isinstance(ohlcv.columns, pd.MultiIndex)
close = ohlcv['Close'] if is_multi else ohlcv
low = ohlcv['Low'] if is_multi else None
high = ohlcv['High'] if is_multi else None

if low is None:
    print("FATAL: OHLCV doesn't have Low/High — single-frame parquet.")
    sys.exit(1)

proba_df = proba_raw.unstack(level=1)
ema_df = proba_df.ewm(span=5).mean()
fri_dates = proba_df.index[proba_df.index.dayofweek == 4]
common = sorted(set(proba_df.columns) & set(close.columns))


def realized_return(entry_date, ticker, sl_pct: float | None, tp_pct: float | None,
                    hold_days: int = 5) -> float:
    """Compute realized return for a single trade, respecting SL/TP.

    entry: close on entry_date.
    Each subsequent day: check Low (for SL) and High (for TP). If hit, exit
    at SL/TP price (NOT the open or true fill — pessimistic for SL since it
    assumes worst close, but slippage isn't modeled).

    Returns gross fractional return (cost added later).
    """
    if ticker not in close.columns:
        return float('nan')
    try:
        idx_pos = close.index.get_loc(entry_date)
    except KeyError:
        return float('nan')
    entry_price = close[ticker].iloc[idx_pos]
    if pd.isna(entry_price) or entry_price <= 0:
        return float('nan')

    # Future window
    end_pos = min(idx_pos + hold_days, len(close) - 1)
    for j in range(idx_pos + 1, end_pos + 1):
        day_low = low[ticker].iloc[j]
        day_high = high[ticker].iloc[j]
        if sl_pct is not None and not pd.isna(day_low):
            sl_price = entry_price * (1 - sl_pct)
            if day_low <= sl_price:
                return (sl_price - entry_price) / entry_price
        if tp_pct is not None and not pd.isna(day_high):
            tp_price = entry_price * (1 + tp_pct)
            if day_high >= tp_price:
                return (tp_price - entry_price) / entry_price
    # No stop hit: exit at hold_days close
    exit_price = close[ticker].iloc[end_pos]
    if pd.isna(exit_price):
        return float('nan')
    return (exit_price - entry_price) / entry_price


def strategy_stats(thr: float, top_k: int, sl_pct: float | None, tp_pct: float | None,
                   date_min=None):
    """Run K-selection on Fridays, with SL/TP, compute trade-level stats."""
    p = ema_df.loc[fri_dates, common]
    if date_min is not None:
        p = p[p.index >= date_min]

    pnls = []
    for fri in p.index:
        row = p.loc[fri]
        elig = row[row >= thr].sort_values(ascending=False).head(top_k)
        for t in elig.index:
            r = realized_return(fri, t, sl_pct, tp_pct, hold_days=5)
            if not pd.isna(r):
                pnls.append(r - config.TOTAL_COST_ROUNDTRIP)

    if not pnls:
        return None
    pnl = pd.Series(pnls)
    wins = pnl[pnl > 0]; losses = pnl[pnl <= 0]
    wr = len(wins)/len(pnl)
    avg_win = float(wins.mean()) if len(wins) else 0
    avg_loss = float(abs(losses.mean())) if len(losses) else 0
    payoff = avg_win/avg_loss if avg_loss > 0 else float('inf')
    expectancy = wr*avg_win - (1-wr)*avg_loss
    return {
        'n': len(pnl), 'wr': round(wr,4),
        'avg_win_pct': round(avg_win*100,3),
        'avg_loss_pct': round(avg_loss*100,3),
        'payoff': round(payoff,2),
        'E_pct': round(expectancy*100,4),
        'total_pnl_pct': round(pnl.sum()*100, 2),
    }


HOLDOUT_MIN = pd.Timestamp('2024-01-01')

print('='*120)
print('Stop-Loss + Take-Profit sweep on K=1 thr=0.65 strategy (Holdout 2024+)')
print('='*120)
print(f'{"SL":>5}  {"TP":>5}  {"n":>3}  {"WR":>7}  {"Avg_W":>7}  {"Avg_L":>7}  {"Payoff":>7}  {"E%":>8}  {"TotalRet":>9}')
configs = [
    (None, None, "baseline (no SL/TP)"),
    (0.015, None, "SL 1.5%"),
    (0.02, None, "SL 2.0%"),
    (0.025, None, "SL 2.5%"),
    (0.03, None, "SL 3.0%"),
    (0.02, 0.02, "SL 2% / TP 2%"),
    (0.02, 0.03, "SL 2% / TP 3%"),
    (0.025, 0.05, "SL 2.5% / TP 5%"),
    (0.03, 0.06, "SL 3% / TP 6%"),
    (0.015, 0.03, "SL 1.5% / TP 3%"),
]
for sl, tp, label in configs:
    s = strategy_stats(0.65, 1, sl, tp, HOLDOUT_MIN)
    if s is None: continue
    sl_str = f'{sl*100:.1f}%' if sl else '—'
    tp_str = f'{tp*100:.1f}%' if tp else '—'
    print(f'{sl_str:>5}  {tp_str:>5}  {s["n"]:>3}  {s["wr"]:>7.2%}  '
          f'{s["avg_win_pct"]:>+6.3f}%  {s["avg_loss_pct"]:>+6.3f}%  '
          f'{s["payoff"]:>7.2f}  {s["E_pct"]:>+7.4f}%  {s["total_pnl_pct"]:>+8.2f}%  | {label}')

print()
print('='*120)
print('Same sweep on production top-5 thr=0.58 (Holdout 2024+)')
print('='*120)
print(f'{"SL":>5}  {"TP":>5}  {"n":>3}  {"WR":>7}  {"Avg_W":>7}  {"Avg_L":>7}  {"Payoff":>7}  {"E%":>8}  {"TotalRet":>9}')
for sl, tp, label in configs:
    s = strategy_stats(0.58, 5, sl, tp, HOLDOUT_MIN)
    if s is None: continue
    sl_str = f'{sl*100:.1f}%' if sl else '—'
    tp_str = f'{tp*100:.1f}%' if tp else '—'
    print(f'{sl_str:>5}  {tp_str:>5}  {s["n"]:>3}  {s["wr"]:>7.2%}  '
          f'{s["avg_win_pct"]:>+6.3f}%  {s["avg_loss_pct"]:>+6.3f}%  '
          f'{s["payoff"]:>7.2f}  {s["E_pct"]:>+7.4f}%  {s["total_pnl_pct"]:>+8.2f}%  | {label}')
