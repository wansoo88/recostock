"""Train v3 with recency weighting + apply best discovered strategy.

Reuses nightly_retrain's universe + feature build, but flips
use_recency_weight=True to test if concept-drift correction helps the
2024-2026 underperformance.

Outputs:
- data/logs/phase3_v3_weighted_oos_proba.parquet
- Comparison vs v3_uniform on K=1 thr=0.65 + SL 1.5% strategy

Run:  python scripts/experiment_weighted_v3.py
"""
from __future__ import annotations
import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import logging
import numpy as np
import pandas as pd

import config
from data.collector import load_parquet
from models.train_lgbm_v2 import (
    build_feature_matrix_v2, build_target_v2, walk_forward_lgbm_v2,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

EXPANDED_TICKERS = (
    config.CORE_ETFS + config.SECTOR_ETFS + config.INVERSE_ETFS +
    ["XLB", "XLU", "XLP", "XLC", "IBB"] + config.VOLATILITY_ETFS
)

WEIGHTED_PROBA_PATH = Path("data/logs/phase3_v3_weighted_oos_proba.parquet")


def train_v3_weighted():
    """Train v3 LightGBM with recency sample_weight (half-life 252 days)."""
    ohlcv = load_parquet("etf_ohlcv")
    close_full = ohlcv["Close"] if isinstance(ohlcv.columns, pd.MultiIndex) else ohlcv
    vix = load_parquet("vix")

    tickers = [t for t in EXPANDED_TICKERS if t in close_full.columns]
    close_df = close_full[tickers].dropna(how="all")
    log.info("Universe: %d tickers", len(tickers))
    log.info("Date range: %s ~ %s", close_df.index.min().date(), close_df.index.max().date())

    log.info("Building v3 feature matrix...")
    X = build_feature_matrix_v2(close_df, vix, macro=None)
    y = build_target_v2(close_df, horizon=5)
    log.info("Features: %d rows × %d cols", *X.shape)

    log.info("Walk-forward training (5-split, recency-weighted half-life=252d)...")
    proba, wf = walk_forward_lgbm_v2(
        X, y, n_splits=5, use_recency_weight=True,
        save_dir=None,  # do NOT promote to production weights
        save_suffix="v3_experimental",
    )
    log.info("WF summary:\n%s", wf.to_string(index=False))
    return proba, wf


def apply_strategy(proba_series, strategy_name):
    """Apply K=1 thr=0.65 + SL 1.5% to a given OOS proba series."""
    ohlcv = load_parquet("etf_ohlcv")
    close = ohlcv["Close"]
    low = ohlcv["Low"]
    high = ohlcv["High"]

    proba_df = proba_series.unstack(level=1)
    ema_df = proba_df.ewm(span=5).mean()
    fri_dates = proba_df.index[proba_df.index.dayofweek == 4]
    common = sorted(set(proba_df.columns) & set(close.columns))
    ema_fri = ema_df.loc[fri_dates, common]

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

    def run_window(thr, k, sl, date_min=None):
        p = ema_fri
        if date_min is not None:
            p = p[p.index >= date_min]
        pnls = []
        for fri in p.index:
            elig = p.loc[fri][p.loc[fri] >= thr].sort_values(ascending=False).head(k)
            for t in elig.index:
                r = realized(fri, t, sl)
                if not pd.isna(r):
                    pnls.append(r - config.TOTAL_COST_ROUNDTRIP)
        if not pnls: return None
        s = pd.Series(pnls)
        wins = s[s > 0]; losses = s[s <= 0]
        wr = len(wins)/len(s)
        avg_w = float(wins.mean()) if len(wins) else 0
        avg_l = float(abs(losses.mean())) if len(losses) else 0
        payoff = avg_w/avg_l if avg_l > 0 else float('inf')
        E = wr*avg_w - (1-wr)*avg_l
        return {'n': len(s), 'wr': wr, 'payoff': payoff, 'E_pct': E*100,
                'total_pct': s.sum()*100}

    print(f'\n=== {strategy_name}: K=1 thr=0.65 + SL 1.5% ===')
    for win_name, dmin in [('FULL', None),
                            ('Pre-2024', None),
                            ('2024+', pd.Timestamp('2024-01-01')),
                            ('Last 12m', pd.Timestamp('2025-05-15'))]:
        if win_name == 'Pre-2024':
            # Need date_max — handle separately
            p2 = ema_fri[ema_fri.index < pd.Timestamp('2024-01-01')]
            pnls = []
            for fri in p2.index:
                elig = p2.loc[fri][p2.loc[fri] >= 0.65].sort_values(ascending=False).head(1)
                for t in elig.index:
                    r = realized(fri, t, 0.015)
                    if not pd.isna(r):
                        pnls.append(r - config.TOTAL_COST_ROUNDTRIP)
            if pnls:
                s = pd.Series(pnls)
                wr = (s > 0).mean()
                wins = s[s > 0]; losses = s[s <= 0]
                payoff = (wins.mean()/abs(losses.mean())) if len(losses) else float('inf')
                E = wr*float(wins.mean() if len(wins) else 0) - (1-wr)*abs(float(losses.mean() if len(losses) else 0))
                total = s.sum()*100
                print(f'  {win_name:>10}  n={len(s):>3}  WR={wr:.2%}  Payoff={payoff:.2f}  E%={E*100:+.4f}%  Total={total:+.2f}%')
        else:
            r = run_window(0.65, 1, 0.015, dmin)
            if r:
                print(f'  {win_name:>10}  n={r["n"]:>3}  WR={r["wr"]:.2%}  '
                      f'Payoff={r["payoff"]:.2f}  E%={r["E_pct"]:+.4f}%  Total={r["total_pct"]:+.2f}%')


def main():
    # Train new weighted model
    if WEIGHTED_PROBA_PATH.exists():
        log.info("Loading cached weighted OOS proba: %s", WEIGHTED_PROBA_PATH)
        proba_w = pd.read_parquet(WEIGHTED_PROBA_PATH)['proba']
    else:
        log.info("No cached weighted model — training fresh")
        proba_w, _ = train_v3_weighted()
        WEIGHTED_PROBA_PATH.parent.mkdir(parents=True, exist_ok=True)
        proba_w.to_frame(name='proba').to_parquet(WEIGHTED_PROBA_PATH)
        log.info("Saved weighted OOS proba")

    # Load uniform baseline
    proba_u = pd.read_parquet('data/logs/phase3_v3_oos_proba.parquet')['proba']

    # Compare
    print('\n' + '='*100)
    print('UNIFORM vs RECENCY-WEIGHTED v3 — same strategy applied')
    print('='*100)
    apply_strategy(proba_u, "UNIFORM (current production)")
    apply_strategy(proba_w, "RECENCY-WEIGHTED (half-life 252d)")


if __name__ == "__main__":
    main()
