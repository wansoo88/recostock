"""Apply top-K selection to BOTH v2 (9 ETF) and v3 (17 ETF) — find true best.

Hypothesis: top-K filtering is the key uplift, universe size is secondary.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from pathlib import Path

import config
from data.collector import load_parquet

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def apply_ema_weekly_smooth(proba_s: pd.Series, ema_span: int = 5):
    prob_df = proba_s.unstack(level=1)
    smoothed = prob_df.ewm(span=ema_span).mean()
    fri = smoothed.index.dayofweek == 4
    return smoothed[fri]


def weights_top_k(smooth_fri: pd.DataFrame, k: int, threshold: float = 0.50) -> pd.DataFrame:
    masked = smooth_fri.where(smooth_fri >= threshold)
    ranks = masked.rank(axis=1, ascending=False, method="first")
    chosen = (ranks <= k).astype(float)
    return chosen.div(chosen.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


def weights_equal(smooth_fri: pd.DataFrame, threshold: float = 0.53) -> pd.DataFrame:
    sig = (smooth_fri >= threshold).astype(float)
    return sig.div(sig.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


def backtest_weighted(weights_fri: pd.DataFrame, close_df: pd.DataFrame) -> pd.Series:
    common_t = [t for t in weights_fri.columns if t in close_df.columns]
    weights_fri = weights_fri[common_t]
    ret_5d = close_df.pct_change(5).shift(-5).reindex(weights_fri.index)[common_t]
    dw = weights_fri.diff().abs().fillna(weights_fri.abs())
    cost_per_week = dw.sum(axis=1) * config.TOTAL_COST_ROUNDTRIP / 2.0
    gross = (weights_fri * ret_5d).sum(axis=1)
    return (gross - cost_per_week).fillna(0)


def sharpe(r): return float(r.mean()/r.std()*np.sqrt(52)) if r.std() > 1e-12 else 0.0
def mdd(r):
    eq = (1+r).cumprod()
    return float(((eq - eq.cummax())/eq.cummax()).min()) * 100
def wr_active(r):
    a = r[r != 0]
    return float((a > 0).mean()) if len(a) else 0.0


ohlcv = load_parquet("etf_ohlcv")
close_full = ohlcv["Close"]

# Load both v2 and v3 OOS probas
proba_v2 = pd.read_parquet("data/logs/phase3_v2_uniform_oos_proba.parquet")["proba"]
proba_v3 = pd.read_parquet("data/logs/phase3_v3_oos_proba.parquet")["proba"]


def metrics_for(r, label, window=None):
    if window:
        cutoff = r.index.max() - pd.Timedelta(days=int(window))
        r = r[r.index >= cutoff]
    return {
        "model": label,
        "n_weeks": int((r != 0).sum()),
        "wr": round(wr_active(r), 4),
        "avg_week_pct": round(r.mean()*100, 4),
        "avg_day_pct": round(r.mean()*100/5, 4),
        "sharpe": round(sharpe(r), 3),
        "mdd_pct": round(mdd(r), 2),
        "total_ret_pct": round(((1+r).cumprod().iloc[-1] - 1)*100, 2),
    }


# Build scenarios for both universes
v2_smooth = apply_ema_weekly_smooth(proba_v2)
v3_smooth = apply_ema_weekly_smooth(proba_v3)

v2_tickers = v2_smooth.columns.tolist()
v3_tickers = v3_smooth.columns.tolist()
close_v2 = close_full[[t for t in v2_tickers if t in close_full.columns]]
close_v3 = close_full[[t for t in v3_tickers if t in close_full.columns]]

scenarios = {
    "v2 equal (9 ETF, all signals)":   backtest_weighted(weights_equal(v2_smooth), close_v2),
    "v2 top-3 (9 ETF)":                backtest_weighted(weights_top_k(v2_smooth, 3), close_v2),
    "v2 top-5 (9 ETF)":                backtest_weighted(weights_top_k(v2_smooth, 5), close_v2),
    "v2 top-7 (9 ETF)":                backtest_weighted(weights_top_k(v2_smooth, 7), close_v2),
    "v3 equal (17 ETF, all signals)":  backtest_weighted(weights_equal(v3_smooth), close_v3),
    "v3 top-3 (17 ETF)":               backtest_weighted(weights_top_k(v3_smooth, 3), close_v3),
    "v3 top-5 (17 ETF)":               backtest_weighted(weights_top_k(v3_smooth, 5), close_v3),
    "v3 top-7 (17 ETF)":               backtest_weighted(weights_top_k(v3_smooth, 7), close_v3),
}

for label_window, days in [("FULL OOS", None),
                            ("RECENT 12 MONTHS", 365),
                            ("RECENT 6 MONTHS", 183),
                            ("RECENT 3 MONTHS", 92)]:
    print("\n" + "=" * 100)
    print(label_window)
    print("=" * 100)
    rows = [metrics_for(r, lbl, window=days) for lbl, r in scenarios.items()]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

# Save best scenario equity
out = Path("data/logs")
pd.DataFrame({k: v for k, v in scenarios.items()}).to_csv(out / "final_topk_scenarios.csv")
log.info("Saved final scenarios to %s", out)
