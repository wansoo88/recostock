"""Train and evaluate LightGBM v2 (technical + macro + recency weight).

Side-by-side comparison: v1 (original, 1d target) vs v2 weighted vs v2 uniform.
Backtests use Friday-rebalance long-only with EMA-5 smoothing.
Saves results to data/logs/phase3_v2_*.csv for further analysis.
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from data.collector import load_parquet
from models.train_lgbm_v2 import (
    build_feature_matrix_v2,
    build_target_v2,
    walk_forward_lgbm_v2,
    MACRO_KEEP_FEATURES,
)
from models.train_lgbm import (
    apply_ema_weekly,
)

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


# ── Load data ──────────────────────────────────────────────────────────────────

ohlcv = load_parquet("etf_ohlcv")
close_full = ohlcv["Close"]
vix = load_parquet("vix")

# Phase 4 universe (core + sector, no leverage)
tickers = [t for t in config.CORE_ETFS + config.SECTOR_ETFS + config.INVERSE_ETFS
           if t in close_full.columns]
close_df = close_full[tickers].dropna(how="all")

log.info("Universe: %s", tickers)
log.info("Date range: %s → %s (n=%d)", close_df.index.min().date(),
         close_df.index.max().date(), len(close_df))


# ── Build features + target ────────────────────────────────────────────────────

log.info("Building feature matrix (v2 with macro)...")
X = build_feature_matrix_v2(close_df, vix, macro=None)
log.info("Feature matrix: %d rows × %d cols", *X.shape)
log.info("Feature columns: %s", list(X.columns))

y = build_target_v2(close_df, horizon=5)
log.info("Target: %d rows (5d forward)", len(y))


# ── Train two variants: weighted vs uniform ────────────────────────────────────

WEIGHTS_DIR = Path("models/weights")

log.info("\n" + "=" * 70)
log.info("V2 WEIGHTED (recency half-life=252d)")
log.info("=" * 70)
proba_w, wf_w = walk_forward_lgbm_v2(
    X, y, n_splits=5, use_recency_weight=True,
    save_dir=WEIGHTS_DIR, save_suffix="v2",
)
log.info("WF summary (weighted):\n%s", wf_w.to_string(index=False))

log.info("\n" + "=" * 70)
log.info("V2 UNIFORM (no recency weight) — baseline for comparison")
log.info("=" * 70)
proba_u, wf_u = walk_forward_lgbm_v2(
    X, y, n_splits=5, use_recency_weight=False,
    save_dir=WEIGHTS_DIR, save_suffix="v2",
)
log.info("WF summary (uniform):\n%s", wf_u.to_string(index=False))


# ── Backtest each: EMA-5 weekly Friday rebalance ───────────────────────────────

def backtest(proba: pd.Series, label: str, threshold: float = 0.53) -> dict:
    sig = apply_ema_weekly(proba, ema_span=5, threshold=threshold)
    # Forward 5d return for each (date, ticker), shifted so we know it at sig date
    ret_5d = close_df.pct_change(5).shift(-5)
    # Align
    common_idx = sig.index.intersection(ret_5d.index)
    sig = sig.reindex(common_idx)
    ret_5d = ret_5d.reindex(common_idx)
    # Apply cost only on position TURN-ONs (carry-over: no cost when signal stays)
    on_turns = sig.diff().fillna(sig).clip(lower=0)  # 1 only on 0→1
    off_turns = sig.diff().fillna(0).clip(upper=0).abs()  # 1 only on 1→0
    cost_today = (on_turns + off_turns) * (config.TOTAL_COST_ROUNDTRIP / 2.0)  # half each leg
    # Portfolio return: avg of active positions × forward 5d return - cost
    common_tickers = [t for t in sig.columns if t in ret_5d.columns]
    sig_c = sig[common_tickers]
    ret_c = ret_5d[common_tickers]
    cost_c = cost_today[common_tickers]
    # Weekly: take every Friday's signal row
    fri_mask = sig_c.index.dayofweek == 4
    sig_fri = sig_c[fri_mask]
    ret_fri = ret_c.reindex(sig_fri.index)
    cost_fri = cost_c.reindex(sig_fri.index)
    n_active = sig_fri.sum(axis=1).replace(0, np.nan)
    # Equal-weight across active tickers
    weighted_pnl = (sig_fri * ret_fri).sum(axis=1) / n_active
    portfolio_cost = (cost_fri.sum(axis=1) / n_active.replace(0, 1))
    net_weekly = (weighted_pnl - portfolio_cost).fillna(0)
    # Metrics
    equity = (1 + net_weekly).cumprod()
    sharpe = (net_weekly.mean() / net_weekly.std() * np.sqrt(52)) if net_weekly.std() > 0 else 0.0
    peak = equity.cummax()
    mdd = float(((equity - peak) / peak).min())
    total_ret = float(equity.iloc[-1] - 1)
    n_active_avg = float(n_active.mean()) if not n_active.dropna().empty else 0.0
    # Win-rate at portfolio level (weekly positive)
    pos_weeks = (net_weekly > 0).sum()
    wr = pos_weeks / max(1, (net_weekly != 0).sum())
    return {
        "label": label,
        "n_weeks": int(len(net_weekly)),
        "n_active_weeks": int((net_weekly != 0).sum()),
        "wr_weeks": round(wr, 4),
        "n_active_avg": round(n_active_avg, 2),
        "sharpe": round(sharpe, 4),
        "mdd_pct": round(mdd * 100, 2),
        "total_ret_pct": round(total_ret * 100, 2),
    }


# Also load v1 model and produce its OOS proba for comparison
def v1_compare() -> pd.Series | None:
    """Run walk_forward with v1 features so we have apples-to-apples OOS proba.
    Re-trains v1 here (no save) to get same WF splits as v2.
    """
    from models.train_lgbm import build_feature_matrix, build_target, walk_forward_lgbm
    log.info("Re-running v1 WF for comparison...")
    X_v1 = build_feature_matrix(close_df, vix)
    y_v1 = build_target(close_df, horizon=1)  # v1 uses 1d
    proba_v1, _ = walk_forward_lgbm(X_v1, y_v1, n_splits=5, save_dir=None)
    return proba_v1


try:
    proba_v1 = v1_compare()
except Exception as exc:
    log.warning("v1 comparison skipped: %s", exc)
    proba_v1 = None


# ── Backtest ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 90)
print("BACKTEST — full WF OOS period (EMA-5 weekly Friday rebalance)")
print("=" * 90)
results = []
if proba_v1 is not None:
    results.append(backtest(proba_v1, "v1 baseline (1d target)"))
results.append(backtest(proba_u, "v2 uniform (5d target, no weight)"))
results.append(backtest(proba_w, "v2 weighted (5d target, recency)"))
df_res = pd.DataFrame(results)
print(df_res.to_string(index=False))


# ── Recent 12m focus (per user requirement) ────────────────────────────────────

def backtest_recent(proba: pd.Series, label: str, months: int = 12) -> dict:
    """Same backtest, restricted to last N months."""
    sig = apply_ema_weekly(proba, ema_span=5, threshold=0.53)
    cutoff = sig.index.max() - pd.Timedelta(days=int(months * 30.5))
    sig = sig[sig.index >= cutoff]
    ret_5d = close_df.pct_change(5).shift(-5).reindex(sig.index)
    common_tickers = [t for t in sig.columns if t in ret_5d.columns]
    sig = sig[common_tickers]
    ret_5d = ret_5d[common_tickers]
    on_t = sig.diff().fillna(sig).clip(lower=0)
    off_t = sig.diff().fillna(0).clip(upper=0).abs()
    cost = (on_t + off_t) * (config.TOTAL_COST_ROUNDTRIP / 2.0)
    fri = sig.index.dayofweek == 4
    sig_f = sig[fri]
    ret_f = ret_5d.reindex(sig_f.index)
    cost_f = cost.reindex(sig_f.index)
    n_act = sig_f.sum(axis=1).replace(0, np.nan)
    weekly = (sig_f * ret_f).sum(axis=1) / n_act
    weekly = (weekly - cost_f.sum(axis=1) / n_act.replace(0, 1)).fillna(0)
    if weekly.std() == 0:
        return {"label": label, "n_weeks": len(weekly), "sharpe": 0, "wr": 0, "ret": 0, "mdd": 0}
    sharpe = weekly.mean() / weekly.std() * np.sqrt(52)
    eq = (1 + weekly).cumprod()
    mdd = float(((eq - eq.cummax()) / eq.cummax()).min())
    return {
        "label": label,
        "n_weeks": len(weekly),
        "wr": round((weekly > 0).sum() / max(1, (weekly != 0).sum()), 4),
        "sharpe": round(sharpe, 4),
        "mdd_pct": round(mdd * 100, 2),
        "total_ret_pct": round((eq.iloc[-1] - 1) * 100, 2),
    }


print("\n" + "=" * 90)
print("RECENT 12 MONTHS — main user concern")
print("=" * 90)
recent_results = []
if proba_v1 is not None:
    recent_results.append(backtest_recent(proba_v1, "v1 baseline (1d target)"))
recent_results.append(backtest_recent(proba_u, "v2 uniform"))
recent_results.append(backtest_recent(proba_w, "v2 weighted"))
print(pd.DataFrame(recent_results).to_string(index=False))


# ── Feature importance comparison ──────────────────────────────────────────────

print("\n" + "=" * 90)
print("FEATURE IMPORTANCE — v2 weighted (final model)")
print("=" * 90)
imp_path = WEIGHTS_DIR / "lgbm_phase3_v2_weighted_importance.csv"
if imp_path.exists():
    imp = pd.read_csv(imp_path)
    print(imp.head(20).to_string(index=False))


# ── Save full results ─────────────────────────────────────────────────────────

out_dir = Path("data/logs")
out_dir.mkdir(parents=True, exist_ok=True)
wf_w.to_csv(out_dir / "phase3_v2_weighted_wf.csv", index=False)
wf_u.to_csv(out_dir / "phase3_v2_uniform_wf.csv", index=False)
df_res.to_csv(out_dir / "phase3_v2_backtest_full.csv", index=False)
pd.DataFrame(recent_results).to_csv(out_dir / "phase3_v2_backtest_recent12m.csv", index=False)

# Save OOS probas for downstream analysis
if proba_v1 is not None:
    proba_v1.to_frame("proba").to_parquet(out_dir / "phase3_v1_oos_proba.parquet")
proba_u.to_frame("proba").to_parquet(out_dir / "phase3_v2_uniform_oos_proba.parquet")
proba_w.to_frame("proba").to_parquet(out_dir / "phase3_v2_weighted_oos_proba.parquet")
log.info("Saved all results to %s", out_dir)
