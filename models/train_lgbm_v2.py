"""Phase 3.5: LightGBM v2 — adds macro features + sample_weight for recency.

Differences vs train_lgbm.py:
1. Adds 11 KEEP macro features (from analyze_macro_ic.py results)
2. sample_weight = exponential decay (half-life = 1 year of trading days)
3. Uses 5-day forward target (not 1d) — matches Friday rebalance directly
4. Saves to lgbm_phase3_v2.pkl (does NOT overwrite production model)
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

import config
from features.factors import compute_factors
from features.macro_factors import build_global_macro
from data.universe import UNIVERSE_BY_TICKER
from data.macro_collector import load_macro_cache

log = logging.getLogger(__name__)

# ── KEEP macro features (from analyze_macro_ic.py pooled h=5) ─────────────────

MACRO_KEEP_FEATURES = [
    "oil_chg_5d",
    "hy_ig_z",
    "vvix_z",
    "y10_z",
    "oil_z",
    "tlt_z",
    "y10_chg_5d",
    "term_spread",
    "gold_chg_21d",
    "hy_ig_logratio",
    "dxy_z",
]

LGBM_PARAMS_V2: dict = dict(
    n_estimators=400,        # slightly more (more features)
    max_depth=4,
    num_leaves=16,
    learning_rate=0.04,      # slightly lower for more features
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,    # more feature subsampling (regularize)
    min_child_samples=80,    # stronger min — more features need more samples per leaf
    reg_alpha=0.2,
    reg_lambda=1.5,
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

# Forward horizon — 5d matches Friday-to-Friday rebalance directly
TARGET_HORIZON = 5

# Sample-weight recency decay (252 trading days = 1 year half-life)
RECENCY_HALF_LIFE_DAYS = 252


# ── Feature engineering ───────────────────────────────────────────────────────

def build_feature_matrix_v2(close_df: pd.DataFrame,
                            vix_df: pd.DataFrame | None = None,
                            macro: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """(date, ticker) MultiIndex with technical + macro features. Strictly causal."""

    if macro is None:
        macro = load_macro_cache()

    # Per-ticker technical factors
    ticker_factors: dict[str, pd.DataFrame] = {}
    for ticker in close_df.columns:
        close = close_df[ticker].dropna()
        if len(close) < 100:
            continue
        ticker_factors[ticker] = compute_factors(close)

    if not ticker_factors:
        return pd.DataFrame()

    common_idx: pd.Index | None = None
    for f in ticker_factors.values():
        common_idx = f.index if common_idx is None else common_idx.intersection(f.index)

    # Cross-section ranks
    rank_mats: dict[str, pd.DataFrame] = {}
    for col in RANK_COLS:
        mat = pd.DataFrame({
            t: ticker_factors[t][col].reindex(common_idx)
            for t in ticker_factors if col in ticker_factors[t].columns
        })
        rank_mats[col] = mat.rank(axis=1, pct=True)

    # Global macro (same for each ticker on a date)
    global_mac = build_global_macro(common_idx, macro)
    macro_cols = [c for c in MACRO_KEEP_FEATURES if c in global_mac.columns]
    macro_aligned = global_mac[macro_cols].reindex(common_idx).ffill()

    parts: list[pd.DataFrame] = []
    for ticker, f in ticker_factors.items():
        meta = UNIVERSE_BY_TICKER.get(ticker)
        sub = f.reindex(common_idx)[FACTOR_COLS].copy()
        sub["is_inverse"] = int(meta.inverse) if meta else 0
        sub["category_code"] = _CATEGORY_MAP.get(meta.category, 0) if meta else 0
        for col in RANK_COLS:
            if col in rank_mats and ticker in rank_mats[col].columns:
                sub[f"{col}_rank"] = rank_mats[col][ticker]
        # Broadcast macro into ticker rows
        for col in macro_cols:
            sub[col] = macro_aligned[col].values

        sub.index = pd.MultiIndex.from_arrays(
            [sub.index, [ticker] * len(sub)], names=["date", "ticker"]
        )
        parts.append(sub)

    features = pd.concat(parts).sort_index()

    if vix_df is not None:
        vix = vix_df.iloc[:, 0].clip(lower=1)
        vix_log = np.log(vix)
        vix_chg = vix.pct_change()
        date_idx = features.index.get_level_values("date")
        features["vix_log"] = date_idx.map(vix_log.to_dict())
        features["vix_chg_1d"] = date_idx.map(vix_chg.to_dict())

    return features.dropna()


def build_target_v2(close_df: pd.DataFrame, horizon: int = TARGET_HORIZON) -> pd.Series:
    """Binary: 1 if forward log-return T→T+horizon (gross of cost) > 0."""
    parts: list[pd.Series] = []
    for ticker in close_df.columns:
        close = close_df[ticker].dropna()
        fwd = np.log(close / close.shift(1)).rolling(horizon).sum().shift(-horizon)
        y = (fwd > 0).astype(np.int8)
        y.index = pd.MultiIndex.from_arrays(
            [y.index, [ticker] * len(y)], names=["date", "ticker"]
        )
        parts.append(y.rename("target"))
    return pd.concat(parts).sort_index()


def recency_weight(dates: pd.Index, half_life: int = RECENCY_HALF_LIFE_DAYS) -> np.ndarray:
    """Exponential decay: w_T = 2 ** -((T_max - T) / half_life). w_T_max = 1."""
    if len(dates) == 0:
        return np.array([], dtype=np.float32)
    dt = pd.DatetimeIndex(dates)
    t_max = dt.max()
    days_back = np.asarray((t_max - dt).total_seconds() / 86400.0, dtype=np.float64)
    w = np.power(2.0, -days_back / half_life).astype(np.float32)
    return w


# ── Walk-forward training (v2) ────────────────────────────────────────────────

def walk_forward_lgbm_v2(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    use_recency_weight: bool = True,
    save_dir: Path | None = None,
    save_suffix: str = "v2",
) -> tuple[pd.Series, pd.DataFrame]:
    """Same WF logic as v1, with optional recency sample_weight."""
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

        ci_is = X_is.index.intersection(y_is.index)
        ci_oos = X_oos.index.intersection(y_oos.index)
        X_is, y_is = X_is.loc[ci_is], y_is.loc[ci_is]
        X_oos, y_oos = X_oos.loc[ci_oos], y_oos.loc[ci_oos]

        if use_recency_weight:
            is_dates = X_is.index.get_level_values("date")
            w_is = recency_weight(pd.DatetimeIndex(is_dates))
        else:
            w_is = None

        model = lgb.LGBMClassifier(**LGBM_PARAMS_V2)
        model.fit(
            X_is, y_is.values,
            sample_weight=w_is,
            feature_name=list(X.columns),
            categorical_feature=["is_inverse", "category_code"],
        )

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
        weight_name = "weighted" if use_recency_weight else "uniform"
        with open(save_dir / f"lgbm_phase3_{save_suffix}_{weight_name}.pkl", "wb") as fh:
            pickle.dump(final_model, fh)
        pd.DataFrame({
            "feature": list(X.columns),
            "importance_gain": final_model.booster_.feature_importance(importance_type="gain"),
        }).sort_values("importance_gain", ascending=False).to_csv(
            save_dir / f"lgbm_phase3_{save_suffix}_{weight_name}_importance.csv",
            index=False,
        )
        log.info("Saved model + importances to %s", save_dir)

    all_proba = pd.concat(proba_parts).sort_index() if proba_parts else pd.Series(dtype=float)
    return all_proba, pd.DataFrame(window_rows)
