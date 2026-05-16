"""Aggressive WR-maximisation simulation.

User demand: "WR still too low — push it higher".
This script explores trade-off WR vs trade-count vs Sharpe.

Scenarios:
  - v3 top-K with raised threshold (0.55, 0.58, 0.60, 0.62)
  - v2+v3 ensemble: enter only when BOTH models agree
  - Triple gate: ensemble + threshold + top-K
  - Volatility regime gate (VIX bands)
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


def smooth_proba(proba_s: pd.Series, ema_span: int = 5):
    prob_df = proba_s.unstack(level=1)
    sm = prob_df.ewm(span=ema_span).mean()
    return sm[sm.index.dayofweek == 4]


def weights_topk_threshold(smooth_fri: pd.DataFrame, k: int, thr: float) -> pd.DataFrame:
    masked = smooth_fri.where(smooth_fri >= thr)
    ranks = masked.rank(axis=1, ascending=False, method="first")
    chosen = (ranks <= k).astype(float)
    return chosen.div(chosen.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


def weights_ensemble(s_v2: pd.DataFrame, s_v3: pd.DataFrame,
                     k: int, thr: float) -> pd.DataFrame:
    """Both v2 AND v3 must be ≥ thr. Then top-K by min(proba_v2, proba_v3)."""
    # Align columns (v2 may have fewer tickers than v3)
    common = sorted(set(s_v2.columns) & set(s_v3.columns))
    s_v2 = s_v2[common].reindex(s_v3.index, method="ffill")
    s_v3 = s_v3[common]
    eligible = (s_v2 >= thr) & (s_v3 >= thr)
    # Use min of two probas as conservative ranking signal
    min_proba = pd.DataFrame(np.minimum(s_v2.values, s_v3.values),
                             index=s_v3.index, columns=common)
    min_proba = min_proba.where(eligible)
    ranks = min_proba.rank(axis=1, ascending=False, method="first")
    chosen = (ranks <= k).astype(float)
    return chosen.div(chosen.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


def backtest_weighted(weights_fri: pd.DataFrame, close_df: pd.DataFrame) -> pd.Series:
    common_t = [t for t in weights_fri.columns if t in close_df.columns]
    weights_fri = weights_fri[common_t]
    ret_5d = close_df.pct_change(5).shift(-5).reindex(weights_fri.index)[common_t]
    dw = weights_fri.diff().abs().fillna(weights_fri.abs())
    cost = dw.sum(axis=1) * config.TOTAL_COST_ROUNDTRIP / 2.0
    gross = (weights_fri * ret_5d).sum(axis=1)
    return (gross - cost).fillna(0)


def stats(r: pd.Series, label: str, window_days: int | None = None):
    if window_days:
        cutoff = r.index.max() - pd.Timedelta(days=window_days)
        r = r[r.index >= cutoff]
    active = r[r != 0]
    if len(active) == 0:
        return {"label": label, "n": 0, "wr": 0, "avg_day_pct": 0, "sharpe": 0,
                "mdd_pct": 0, "total_ret_pct": 0}
    sh = active.mean() / active.std() * np.sqrt(52) if active.std() > 0 else 0
    eq = (1 + r).cumprod()
    mdd = float(((eq - eq.cummax()) / eq.cummax()).min()) * 100
    return {
        "label": label,
        "n": int(len(active)),
        "wr": round((active > 0).mean(), 4),
        "avg_day_pct": round(r.mean()*100/5, 4),
        "sharpe": round(float(sh), 2),
        "mdd_pct": round(mdd, 2),
        "total_ret_pct": round((eq.iloc[-1] - 1)*100, 2),
    }


# ── Load
ohlcv = load_parquet("etf_ohlcv")
close_full = ohlcv["Close"]
vix = load_parquet("vix").iloc[:, 0]

proba_v2 = pd.read_parquet("data/logs/phase3_v2_uniform_oos_proba.parquet")["proba"]
proba_v3 = pd.read_parquet("data/logs/phase3_v3_oos_proba.parquet")["proba"]

s_v2 = smooth_proba(proba_v2)
s_v3 = smooth_proba(proba_v3)
close_v3 = close_full[[t for t in s_v3.columns if t in close_full.columns]]


# ── Build all scenarios
scenarios = {}

# v3 top-K threshold sweep
for k in [3, 5, 7]:
    for thr in [0.53, 0.55, 0.58, 0.60, 0.62]:
        label = f"v3 top-{k} thr={thr}"
        w = weights_topk_threshold(s_v3, k, thr)
        scenarios[label] = backtest_weighted(w, close_v3)

# Ensemble (v2 AND v3 agree)
for k in [3, 5, 7]:
    for thr in [0.53, 0.55, 0.58, 0.60]:
        label = f"ens v2+v3 top-{k} thr={thr}"
        w = weights_ensemble(s_v2, s_v3, k, thr)
        scenarios[label] = backtest_weighted(w, close_v3)

# ── VIX-regime gate variant: only when VIX>=18 AND v3 top-K
vix_fri = vix.reindex(s_v3.index, method="ffill")
for k in [3, 5, 7]:
    for vix_min in [16, 18, 20]:
        label = f"v3 top-{k} thr=0.55 vix>={vix_min}"
        w = weights_topk_threshold(s_v3, k, 0.55)
        mask = (vix_fri >= vix_min).values
        w = pd.DataFrame(np.where(mask[:, None], w.values, 0.0),
                         index=w.index, columns=w.columns)
        scenarios[label] = backtest_weighted(w, close_v3)


# ── Report — best by recent 12m and recent 6m
def summarize_all(window_days, header):
    print("\n" + "=" * 110)
    print(header)
    print("=" * 110)
    rows = [stats(r, lbl, window_days=window_days) for lbl, r in scenarios.items()]
    df = pd.DataFrame(rows)
    # Filter: must have at least 5 trades
    df_ok = df[df["n"] >= 5].copy()
    df_ok = df_ok.sort_values("wr", ascending=False)
    print(df_ok.head(20).to_string(index=False))
    print(f"\n(Filtered to n_active_weeks >= 5; top-20 by WR shown)")


summarize_all(None, "FULL OOS — ranked by WR (n≥5)")
summarize_all(365, "RECENT 12 MONTHS — ranked by WR (n≥5)")
summarize_all(183, "RECENT 6 MONTHS — ranked by WR (n≥5)")
summarize_all(92,  "RECENT 3 MONTHS — ranked by WR (n≥5)")


# ── Top-WR scenarios — also show Sharpe (to expose trade-off)
print("\n" + "=" * 110)
print("WR-OPTIMIZED but Sharpe-AWARE — recent 12m, n≥10 (more reliable)")
print("=" * 110)
rows = [stats(r, lbl, window_days=365) for lbl, r in scenarios.items()]
df = pd.DataFrame(rows)
df_ok = df[df["n"] >= 10].sort_values(["wr", "sharpe"], ascending=[False, False])
print(df_ok.head(15).to_string(index=False))


# Save full result
out = Path("data/logs")
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame([stats(r, lbl) for lbl, r in scenarios.items()]).to_csv(
    out / "push_wr_scenarios_full.csv", index=False)
pd.DataFrame([stats(r, lbl, window_days=365) for lbl, r in scenarios.items()]).to_csv(
    out / "push_wr_scenarios_recent12m.csv", index=False)
log.info("Saved push_wr_scenarios_*.csv")
