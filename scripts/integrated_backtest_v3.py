"""V3 integrated backtest: v2 model + expanded universe + position sizing + dynamic threshold.

Steps:
  1. Re-train v2 with expanded universe (14 sector ETFs)
  2. Apply position sizing options: equal / proba-weighted / inverse-vol / top-K
  3. Compare against v1 baseline + v2 single-improvement
  4. Recent 12m / 6m / 3m / 30d performance focus per user demand
"""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config
from data.collector import load_parquet
from features.factors import compute_factors
from features.macro_factors import build_global_macro
from data.universe import UNIVERSE_BY_TICKER
from data.macro_collector import load_macro_cache
from models.train_lgbm_v2 import (
    walk_forward_lgbm_v2, build_target_v2, MACRO_KEEP_FEATURES,
    FACTOR_COLS, RANK_COLS, _CATEGORY_MAP,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── Expanded universe (14 sector + 3 core + 3 inverse) ────────────────────────
NEW_SECTORS = ["XLB", "XLU", "XLP", "XLC", "IBB"]
EXPANDED_TICKERS = (config.CORE_ETFS + config.SECTOR_ETFS + NEW_SECTORS
                    + config.INVERSE_ETFS)


def build_feature_matrix_expanded(close_df: pd.DataFrame, vix_df: pd.DataFrame,
                                  macro: dict) -> pd.DataFrame:
    """Feature matrix with new tickers — treat new sector ETFs same as existing sectors."""
    ticker_factors: dict[str, pd.DataFrame] = {}
    for t in close_df.columns:
        close = close_df[t].dropna()
        if len(close) < 100:
            continue
        ticker_factors[t] = compute_factors(close)
    if not ticker_factors:
        return pd.DataFrame()

    common_idx = None
    for f in ticker_factors.values():
        common_idx = f.index if common_idx is None else common_idx.intersection(f.index)

    rank_mats = {}
    for col in RANK_COLS:
        mat = pd.DataFrame({
            t: ticker_factors[t][col].reindex(common_idx)
            for t in ticker_factors if col in ticker_factors[t].columns
        })
        rank_mats[col] = mat.rank(axis=1, pct=True)

    global_mac = build_global_macro(common_idx, macro)
    macro_cols = [c for c in MACRO_KEEP_FEATURES if c in global_mac.columns]
    macro_aligned = global_mac[macro_cols].reindex(common_idx).ffill()

    parts = []
    for t, f in ticker_factors.items():
        # For NEW sector ETFs not in UNIVERSE_BY_TICKER, treat as sector(1)
        meta = UNIVERSE_BY_TICKER.get(t)
        is_inverse = int(meta.inverse) if meta else 0
        category_code = _CATEGORY_MAP.get(meta.category, 1) if meta else 1

        sub = f.reindex(common_idx)[FACTOR_COLS].copy()
        sub["is_inverse"] = is_inverse
        sub["category_code"] = category_code
        for col in RANK_COLS:
            if col in rank_mats and t in rank_mats[col].columns:
                sub[f"{col}_rank"] = rank_mats[col][t]
        for col in macro_cols:
            sub[col] = macro_aligned[col].values
        sub.index = pd.MultiIndex.from_arrays(
            [sub.index, [t] * len(sub)], names=["date", "ticker"]
        )
        parts.append(sub)
    features = pd.concat(parts).sort_index()

    if vix_df is not None:
        vix = vix_df.iloc[:, 0].clip(lower=1)
        date_idx = features.index.get_level_values("date")
        features["vix_log"] = date_idx.map(np.log(vix).to_dict())
        features["vix_chg_1d"] = date_idx.map(vix.pct_change().to_dict())

    return features.dropna()


# ── Load data ──────────────────────────────────────────────────────────────────
ohlcv = load_parquet("etf_ohlcv")
close_full = ohlcv["Close"]
vix = load_parquet("vix")
macro = load_macro_cache()

tickers = [t for t in EXPANDED_TICKERS if t in close_full.columns]
close_df = close_full[tickers].dropna(how="all")
log.info("Expanded universe (%d): %s", len(tickers), tickers)

X = build_feature_matrix_expanded(close_df, vix, macro)
y = build_target_v2(close_df, horizon=5)
log.info("Feature matrix: %s, target: %d", X.shape, len(y))


# ── Train expanded v3 (= v2 uniform on expanded universe) ──────────────────────
WEIGHTS = Path("models/weights")
log.info("\n=== Training v3 (v2 spec + expanded universe) ===")
proba_v3, wf_v3 = walk_forward_lgbm_v2(
    X, y, n_splits=5, use_recency_weight=False,
    save_dir=WEIGHTS, save_suffix="v3",
)
log.info("WF summary:\n%s", wf_v3.to_string(index=False))


# ── Position sizing methods ───────────────────────────────────────────────────

def apply_ema_weekly_proba(proba_s: pd.Series, ema_span: int = 5):
    """Returns smoothed-proba DataFrame (date × ticker) on Fridays only."""
    prob_df = proba_s.unstack(level=1)
    smoothed = prob_df.ewm(span=ema_span).mean()
    fri = smoothed.index.dayofweek == 4
    return smoothed[fri]


def weights_equal(smooth_fri: pd.DataFrame, threshold: float = 0.53) -> pd.DataFrame:
    sig = (smooth_fri >= threshold).astype(float)
    return sig.div(sig.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


def weights_proba(smooth_fri: pd.DataFrame, threshold: float = 0.53) -> pd.DataFrame:
    """Weight ∝ (proba - 0.5) above threshold. More confidence → bigger position."""
    eligible = smooth_fri >= threshold
    edge = (smooth_fri - 0.5).clip(lower=0) * eligible.astype(float)
    return edge.div(edge.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


def weights_inverse_vol(smooth_fri: pd.DataFrame, close_df: pd.DataFrame,
                        threshold: float = 0.53, vol_window: int = 21) -> pd.DataFrame:
    """Equal weight among eligible, scaled by 1/vol so low-vol ETFs get more capital."""
    eligible = (smooth_fri >= threshold).astype(float)
    ret = close_df.pct_change()
    vol = ret.rolling(vol_window).std() * np.sqrt(252)
    vol_fri = vol.reindex(smooth_fri.index).fillna(method="ffill")
    # only over eligible
    common_t = [t for t in smooth_fri.columns if t in vol_fri.columns]
    eligible = eligible[common_t]
    vol_fri = vol_fri[common_t].replace(0, np.nan)
    inv_vol = (1.0 / vol_fri) * eligible
    return inv_vol.div(inv_vol.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


def weights_top_k(smooth_fri: pd.DataFrame, k: int = 3, threshold: float = 0.50) -> pd.DataFrame:
    """Top-K probabilities each Friday (subject to min threshold)."""
    masked = smooth_fri.where(smooth_fri >= threshold)
    # rank descending per row, keep top k
    ranks = masked.rank(axis=1, ascending=False, method="first")
    chosen = (ranks <= k).astype(float)
    return chosen.div(chosen.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


# ── Backtest with weights and carry-over cost ─────────────────────────────────

def backtest_weighted(weights_fri: pd.DataFrame, close_df: pd.DataFrame) -> pd.Series:
    """5-day forward return per Friday rebalance.

    Cost is charged on weight CHANGES — carry-over has no cost.
    Net weekly return = Σ_t (w_t × r_5d_t) - 0.5 * Σ_t |Δw_t| * cost
    (Half cost on entry, half on exit → |Δw| * cost / 2 each leg = full cost roundtrip)
    """
    common_t = [t for t in weights_fri.columns if t in close_df.columns]
    weights_fri = weights_fri[common_t]
    ret_5d = close_df.pct_change(5).shift(-5).reindex(weights_fri.index)[common_t]
    # weight change
    dw = weights_fri.diff().abs().fillna(weights_fri.abs())
    cost_per_week = dw.sum(axis=1) * config.TOTAL_COST_ROUNDTRIP / 2.0
    gross_pnl = (weights_fri * ret_5d).sum(axis=1)
    return (gross_pnl - cost_per_week).fillna(0)


def sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(52)) if r.std() > 1e-12 else 0.0


def mdd(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float(((eq - eq.cummax()) / eq.cummax()).min()) * 100


def wr_active(r: pd.Series) -> float:
    a = r[r != 0]
    return float((a > 0).mean()) if len(a) else 0.0


# ── Run scenarios ──────────────────────────────────────────────────────────────

smoothed_fri = apply_ema_weekly_proba(proba_v3)
log.info("\nSmoothed Friday signals: %s", smoothed_fri.shape)

w_eq = weights_equal(smoothed_fri)
w_pr = weights_proba(smoothed_fri)
w_iv = weights_inverse_vol(smoothed_fri, close_df)
w_top3 = weights_top_k(smoothed_fri, k=3)
w_top5 = weights_top_k(smoothed_fri, k=5)
w_top7 = weights_top_k(smoothed_fri, k=7)

scenarios = {
    "v3 equal-weight (baseline)": backtest_weighted(w_eq, close_df),
    "v3 proba-weighted":          backtest_weighted(w_pr, close_df),
    "v3 inverse-vol":             backtest_weighted(w_iv, close_df),
    "v3 top-3":                   backtest_weighted(w_top3, close_df),
    "v3 top-5":                   backtest_weighted(w_top5, close_df),
    "v3 top-7":                   backtest_weighted(w_top7, close_df),
}


def metrics_for(r: pd.Series, label: str, window: int = None) -> dict:
    if window:
        cutoff = r.index.max() - pd.Timedelta(days=int(window))
        r = r[r.index >= cutoff]
    return {
        "label": label,
        "n": int((r != 0).sum()),
        "wr": round(wr_active(r), 4),
        "avg_week_pct": round(r.mean() * 100, 4),
        "avg_day_pct": round(r.mean() * 100 / 5, 4),
        "sharpe": round(sharpe(r), 3),
        "mdd_pct": round(mdd(r), 2),
        "total_ret_pct": round(((1+r).cumprod().iloc[-1] - 1) * 100, 2),
    }


# Full OOS
print("\n" + "=" * 100)
print("FULL OOS — scenario comparison")
print("=" * 100)
full = pd.DataFrame([metrics_for(r, label) for label, r in scenarios.items()])
print(full.to_string(index=False))

# Recent windows
for window_days, label in [(365, "RECENT 12 MONTHS"),
                            (183, "RECENT 6 MONTHS"),
                            (92, "RECENT 3 MONTHS"),
                            (30, "RECENT 30 DAYS")]:
    print("\n" + "=" * 100)
    print(f"{label}")
    print("=" * 100)
    rows = [metrics_for(r, lbl, window=window_days) for lbl, r in scenarios.items()]
    rec = pd.DataFrame(rows)
    print(rec.to_string(index=False))


# Save best scenario equity curve
out_dir = Path("data/logs")
out_dir.mkdir(parents=True, exist_ok=True)
results_df = pd.DataFrame({k: v for k, v in scenarios.items()})
results_df.to_csv(out_dir / "v3_scenarios_weekly.csv")
proba_v3.to_frame("proba").to_parquet(out_dir / "phase3_v3_oos_proba.parquet")
log.info("Saved v3 scenarios + OOS proba to %s", out_dir)
