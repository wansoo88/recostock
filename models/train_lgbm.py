"""Phase 3: LightGBM walk-forward training and signal generation.

Pooled model across all ETFs — one model learns general factor patterns.
Features: 11 time-series factors + 5 cross-sectional ranks + VIX + ETF metadata.
Target: binary — 1 if next-day return > 0, else 0.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from features.factors import compute_factors
from data.universe import UNIVERSE_BY_TICKER

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

LGBM_PARAMS: dict = dict(
    n_estimators=300,
    max_depth=4,
    num_leaves=16,
    learning_rate=0.05,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    min_child_samples=50,
    reg_alpha=0.1,
    reg_lambda=1.0,
    is_unbalance=True,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)

FACTOR_COLS = [
    "mom_1d", "mom_5d", "mom_10d", "mom_21d", "mom_63d",
    "rsi_14", "zscore_20",
    "vol_5d", "vol_21d", "vol_ratio",
    "trend_strength",
]

RANK_COLS = ["mom_63d", "mom_21d", "zscore_20", "rsi_14", "vol_5d"]

_CATEGORY_MAP = {
    "core": 0, "sector": 1, "inverse": 2,
    "volatility": 3, "leverage_long": 4, "leverage_inverse": 5,
}


# ── Feature engineering ───────────────────────────────────────────────────────

def build_feature_matrix(
    close_df: pd.DataFrame,
    vix_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build (date, ticker) MultiIndex feature DataFrame.

    All features are strictly causal: features at T use only data up to T.
    """
    # Per-ticker factor computation
    ticker_factors: dict[str, pd.DataFrame] = {}
    for ticker in close_df.columns:
        close = close_df[ticker].dropna()
        if len(close) < 100:
            continue
        f = compute_factors(close)
        ticker_factors[ticker] = f

    if not ticker_factors:
        return pd.DataFrame()

    # Common date index (all tickers must have factor values)
    common_idx: pd.Index = None  # type: ignore[assignment]
    for f in ticker_factors.values():
        common_idx = f.index if common_idx is None else common_idx.intersection(f.index)

    # Cross-sectional pct-rank per factor per date (causal: uses all tickers at date T)
    rank_mats: dict[str, pd.DataFrame] = {}
    for col in RANK_COLS:
        mat = pd.DataFrame(
            {t: ticker_factors[t][col].reindex(common_idx) for t in ticker_factors if col in ticker_factors[t].columns}
        )
        rank_mats[col] = mat.rank(axis=1, pct=True)  # date × ticker

    # Assemble per-ticker rows
    parts: list[pd.DataFrame] = []
    for ticker, f in ticker_factors.items():
        meta = UNIVERSE_BY_TICKER.get(ticker)
        sub = f.reindex(common_idx)[FACTOR_COLS].copy()
        sub["is_inverse"] = int(meta.inverse) if meta else 0
        sub["category_code"] = _CATEGORY_MAP.get(meta.category, 0) if meta else 0
        for col in RANK_COLS:
            if col in rank_mats and ticker in rank_mats[col].columns:
                sub[f"{col}_rank"] = rank_mats[col][ticker]
        sub.index = pd.MultiIndex.from_arrays(
            [sub.index, [ticker] * len(sub)], names=["date", "ticker"]
        )
        parts.append(sub)

    features = pd.concat(parts).sort_index()

    # VIX features (macro context)
    if vix_df is not None:
        vix = vix_df.iloc[:, 0].clip(lower=1)
        vix_log = np.log(vix)
        vix_chg = vix.pct_change()
        date_idx = features.index.get_level_values("date")
        features["vix_log"] = date_idx.map(vix_log.to_dict())
        features["vix_chg_1d"] = date_idx.map(vix_chg.to_dict())

    return features.dropna()


def build_target(close_df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Binary target: 1 if forward return T→T+horizon > 0, else 0.

    Indexed by (date, ticker) MultiIndex.
    """
    parts: list[pd.Series] = []
    for ticker in close_df.columns:
        close = close_df[ticker].dropna()
        fwd = close.pct_change(horizon).shift(-horizon)
        y = (fwd > 0).astype(np.int8)
        y.index = pd.MultiIndex.from_arrays(
            [y.index, [ticker] * len(y)], names=["date", "ticker"]
        )
        parts.append(y.rename("target"))
    return pd.concat(parts).sort_index()


# ── Walk-forward training ─────────────────────────────────────────────────────

def walk_forward_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    save_dir: Path | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Expanding-window walk-forward: train on IS, predict OOS, repeat.

    Returns
    -------
    proba : OOS probabilities, (date, ticker) MultiIndex, values in [0, 1]
    metrics : DataFrame with per-window IS/OOS AUC, sample counts, dates
    """
    dates = sorted(X.index.get_level_values("date").unique())
    n = len(dates)
    min_is = min(504, n // 3)
    oos_size = (n - min_is) // n_splits
    if oos_size < 20:
        raise ValueError(f"Insufficient dates for {n_splits}-split walk-forward: n={n}")

    proba_parts: list[pd.Series] = []
    window_rows: list[dict] = []
    final_model: lgb.LGBMClassifier | None = None

    for k in range(n_splits):
        is_end = min_is + k * oos_size
        oos_start = is_end
        oos_end = oos_start + oos_size if k < n_splits - 1 else n

        is_set = set(dates[:is_end])
        oos_set = set(dates[oos_start:oos_end])

        X_is = X[X.index.get_level_values("date").isin(is_set)]
        y_is = y[y.index.get_level_values("date").isin(is_set)]
        X_oos = X[X.index.get_level_values("date").isin(oos_set)]
        y_oos = y[y.index.get_level_values("date").isin(oos_set)]

        # Align to common index per split
        ci_is = X_is.index.intersection(y_is.index)
        ci_oos = X_oos.index.intersection(y_oos.index)
        X_is, y_is = X_is.loc[ci_is], y_is.loc[ci_is]
        X_oos, y_oos = X_oos.loc[ci_oos], y_oos.loc[ci_oos]

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(X_is, y_is.values,
                  feature_name=list(X.columns),
                  categorical_feature=["is_inverse", "category_code"])

        is_proba_arr = model.predict_proba(X_is)[:, 1]
        oos_proba_arr = model.predict_proba(X_oos)[:, 1]

        is_auc = roc_auc_score(y_is.values, is_proba_arr)
        oos_auc = roc_auc_score(y_oos.values, oos_proba_arr)

        proba_parts.append(pd.Series(oos_proba_arr, index=ci_oos, name="proba"))

        window_rows.append({
            "window": k + 1,
            "is_end": str(dates[is_end - 1].date()),
            "oos_start": str(dates[oos_start].date()),
            "oos_end": str(dates[min(oos_end - 1, n - 1)].date()),
            "n_train": len(X_is),
            "n_test": len(X_oos),
            "is_auc": round(is_auc, 4),
            "oos_auc": round(oos_auc, 4),
        })
        log.info("WF %d/%d  IS AUC=%.4f  OOS AUC=%.4f  n_train=%d  n_test=%d",
                 k + 1, n_splits, is_auc, oos_auc, len(X_is), len(X_oos))
        final_model = model

    if save_dir is not None and final_model is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "lgbm_phase3.pkl", "wb") as fh:
            pickle.dump(final_model, fh)
        pd.DataFrame({
            "feature": list(X.columns),
            "importance_gain": final_model.booster_.feature_importance(importance_type="gain"),
        }).sort_values("importance_gain", ascending=False).to_csv(
            save_dir / "lgbm_phase3_feature_importance.csv", index=False
        )
        log.info("Model + importances saved to %s", save_dir)

    all_proba = pd.concat(proba_parts).sort_index() if proba_parts else pd.Series(dtype=float)
    return all_proba, pd.DataFrame(window_rows)


# ── Signal conversion ─────────────────────────────────────────────────────────

def proba_to_signals(
    proba: pd.Series,
    threshold_long: float = 0.53,
    threshold_short: float = 0.47,
    long_flat: bool = True,
) -> pd.DataFrame:
    """Convert OOS probabilities to date×ticker signal DataFrame.

    long_flat=True : {0, +1}  — no shorting (Phase 2 finding: lower MDD)
    long_flat=False: {-1, 0, +1} — full long/short
    """
    sig = proba.unstack(level=1)  # date × ticker
    vals = np.zeros(sig.shape, dtype=np.int8)
    vals[sig.values >= threshold_long] = 1
    if not long_flat:
        vals[sig.values <= threshold_short] = -1
    return pd.DataFrame(vals, index=sig.index, columns=sig.columns)


def apply_ema_weekly(
    proba: pd.Series,
    ema_span: int = 5,
    threshold: float = 0.53,
) -> pd.DataFrame:
    """Best Phase 3 signal: EMA-smoothed probability + Friday-only rebalancing.

    Reduces daily noise (EMA) and transaction costs (weekly hold).
    Returns {0, +1} DataFrame (long/flat, no shorting).
    """
    prob_df = proba.unstack(level=1)  # date × ticker
    smoothed = prob_df.ewm(span=ema_span).mean()
    daily_sig = (smoothed >= threshold).astype(float)
    # Forward-fill non-Friday days with last Friday's signal
    weekly = daily_sig.copy()
    weekly[weekly.index.dayofweek != 4] = np.nan
    return weekly.ffill().fillna(0)
