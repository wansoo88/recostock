"""Experiment: high-conviction concentration vs production baseline.

Walk-forward holdout split (train: 2020-2023, holdout: 2024+).
Tests K=1 single-best strategy with fractional allocation to manage MDD,
and confidence-weighted execution.

Run:  python scripts/experiment_conviction.py
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
close = ohlcv['Close'] if isinstance(ohlcv.columns, pd.MultiIndex) else ohlcv

proba_df = proba_raw.unstack(level=1)
ema_df = proba_df.ewm(span=5).mean()
fwd5 = close.pct_change(5).shift(-5)
fri_dates = proba_df.index[proba_df.index.dayofweek == 4]
common = sorted(set(proba_df.columns) & set(close.columns))
ema_fri = ema_df.loc[fri_dates, common]
ret_fri = fwd5.reindex(fri_dates)[common]


def detailed_stats(p_mat, r_mat, thr, top_k, date_min=None, date_max=None,
                   alloc_per_trade=None):
    p = p_mat.copy(); r = r_mat.copy()
    if date_min is not None: p = p[p.index >= date_min]; r = r[r.index >= date_min]
    if date_max is not None: p = p[p.index < date_max]; r = r[r.index < date_max]
    elig = p.where(p >= thr)
    ranks = elig.rank(axis=1, ascending=False, method='first')
    chosen = ranks <= top_k
    pnl = r.where(chosen).stack(future_stack=True).dropna() - config.TOTAL_COST_ROUNDTRIP
    n = len(pnl)
    if n == 0: return None
    wins = pnl[pnl > 0]; losses = pnl[pnl < 0]
    wr = len(wins)/n
    avg_win = float(wins.mean()) if len(wins) else 0
    avg_loss = float(abs(losses.mean())) if len(losses) else 0
    payoff = avg_win/avg_loss if avg_loss > 0 else float('inf')
    expectancy = wr*avg_win - (1-wr)*avg_loss
    if alloc_per_trade is None:
        w = chosen.astype(float).div(chosen.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    else:
        w = chosen.astype(float) * alloc_per_trade
    weekly = (w * r).sum(axis=1)
    weekly_active = weekly[weekly != 0]
    sharpe = weekly_active.mean()/weekly_active.std()*np.sqrt(52) if weekly_active.std() > 0 else 0
    eq = (1 + weekly).cumprod()
    mdd = float(((eq - eq.cummax()) / eq.cummax()).min())
    return {'n': n, 'wr': round(wr,4), 'avg_win_pct': round(avg_win*100,3),
            'avg_loss_pct': round(avg_loss*100,3), 'payoff': round(payoff,2),
            'E_pct': round(expectancy*100,4), 'sharpe': round(float(sharpe),3),
            'mdd_pct': round(mdd*100,2)}


HOLDOUT_MIN = pd.Timestamp('2024-01-01')

print('='*110)
print('K=1 single-best, Holdout 2024+, detailed payoff structure')
print('='*110)
print(f'{"thr":>5}  {"alloc":>6}  {"n":>3}  {"WR":>6}  {"Avg_W":>7}  {"Avg_L":>7}  {"Payoff":>7}  {"E%":>8}  {"Sharpe":>7}  {"MDD":>7}')
for thr in [0.55, 0.58, 0.60, 0.62, 0.65, 0.68]:
    for alloc in [None, 0.5, 0.3, 0.2]:
        s = detailed_stats(ema_fri, ret_fri, thr, 1, HOLDOUT_MIN, None, alloc)
        if s is None or s['n'] < 15: continue
        alloc_str = '100%' if alloc is None else f'{int(alloc*100)}%'
        print(f'{thr:>5.2f}  {alloc_str:>6}  {s["n"]:>3}  {s["wr"]:>6.2%}  '
              f'{s["avg_win_pct"]:>+6.3f}%  {s["avg_loss_pct"]:>+6.3f}%  '
              f'{s["payoff"]:>7.2f}  {s["E_pct"]:>+7.4f}%  {s["sharpe"]:>+7.3f}  {s["mdd_pct"]:>+6.2f}%')

print()
print('Production baseline (top-5 thr=0.58):')
s = detailed_stats(ema_fri, ret_fri, 0.58, 5, HOLDOUT_MIN, None, None)
print(f'  n={s["n"]}  WR={s["wr"]:.2%}  Payoff={s["payoff"]:.2f}  Sharpe={s["sharpe"]:+.3f}  MDD={s["mdd_pct"]:+.2f}%')

print()
print('Top-3 thr=0.58 alternative:')
s = detailed_stats(ema_fri, ret_fri, 0.58, 3, HOLDOUT_MIN, None, None)
print(f'  n={s["n"]}  WR={s["wr"]:.2%}  Payoff={s["payoff"]:.2f}  Sharpe={s["sharpe"]:+.3f}  MDD={s["mdd_pct"]:+.2f}%')
