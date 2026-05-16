"""Production-ready inference module for v3 model (macro + expanded universe + top-K).

Drop-in replacement for models/inference.py — same return shape, but with:
  - Loads lgbm_phase3_v3_uniform.pkl (macro features, 5-day target)
  - Includes macro_factors features in build_feature_matrix
  - Optional TOP_K selection (default 7) — only top-N highest-proba tickers signal LONG
  - Same rolling stats interface for compatibility

To switch production from v1 → v3:
  scripts/run_daily.py needs to import from this module instead of models.inference
  OR replace inference.py content with this once approved.

Until then, this file is dormant — daily_signal.yml still uses inference.py (v1).
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import config
from features.factors import compute_factors
from features.macro_factors import build_global_macro
from data.universe import UNIVERSE_BY_TICKER
from data.macro_collector import load_macro_cache

log = logging.getLogger(__name__)

WEIGHTS_DIR = Path("models/weights")
PROBA_HISTORY_PATH = Path("data/logs/proba_history_v3.parquet")
PROBA_HISTORY_DAYS = 20
EMA_SPAN = 5
ROLLING_WINDOW = 60
HOLD_DAYS = 5
MIN_ACTIVE_FRIDAYS = 8
TOP_K = 7   # Production setting — verified in final_topk_comparison.py

# Same KEEP macro features as v2/v3 training
MACRO_KEEP_FEATURES = [
    "oil_chg_5d", "hy_ig_z", "vvix_z", "y10_z", "oil_z", "tlt_z",
    "y10_chg_5d", "term_spread", "gold_chg_21d", "hy_ig_logratio", "dxy_z",
]
FACTOR_COLS = [
    "mom_1d", "mom_5d", "mom_10d", "mom_21d", "mom_63d",
    "rsi_14", "zscore_20", "vol_5d", "vol_21d", "vol_ratio", "trend_strength",
]
RANK_COLS = ["mom_63d", "mom_21d", "zscore_20", "rsi_14", "vol_5d"]
_CATEGORY_MAP = {
    "core": 0, "sector": 1, "inverse": 2,
    "volatility": 3, "leverage_long": 4, "leverage_inverse": 5,
}


_cached_model = None
_cached_macro = None


def _load_model():
    global _cached_model
    if _cached_model is None:
        model_path = WEIGHTS_DIR / "lgbm_phase3_v3_uniform.pkl"
        if not model_path.exists():
            # Fallback to v2 if v3 not yet trained on production data
            model_path = WEIGHTS_DIR / "lgbm_phase3_v2_uniform.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"v3/v2 model not found in {WEIGHTS_DIR}")
        with open(model_path, "rb") as f:
            _cached_model = pickle.load(f)
        log.info("LightGBM v3 model loaded from %s", model_path)
    return _cached_model


def _load_macro():
    global _cached_macro
    if _cached_macro is None:
        _cached_macro = load_macro_cache()
    return _cached_macro


def build_feature_matrix_v3(close_df: pd.DataFrame, vix_df: pd.DataFrame | None = None,
                            macro: dict | None = None) -> pd.DataFrame:
    if macro is None:
        macro = _load_macro()

    ticker_factors = {}
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


def load_proba_history() -> pd.DataFrame:
    if PROBA_HISTORY_PATH.exists():
        return pd.read_parquet(PROBA_HISTORY_PATH)
    return pd.DataFrame()


def save_proba_history(history: pd.DataFrame) -> None:
    PROBA_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history.tail(PROBA_HISTORY_DAYS).to_parquet(PROBA_HISTORY_PATH)


def append_today_proba(history: pd.DataFrame, today_proba: pd.Series) -> pd.DataFrame:
    today_df = today_proba.to_frame().T
    today_df.index = pd.DatetimeIndex([
        today_proba.name if today_proba.name else pd.Timestamp.today().normalize()
    ])
    return pd.concat([history, today_df]).drop_duplicates().tail(PROBA_HISTORY_DAYS)


def score_today(close_df: pd.DataFrame, vix_df: pd.DataFrame | None,
                proba_history: pd.DataFrame, top_k: int = TOP_K):
    """Same return shape as inference.score_today — adds top-K selection.

    Returns (results dict, raw_proba Series).
    results[ticker] = {"raw_proba", "ema_proba", "signal", "rank"}
    signal: +1 only for top_k tickers above SIGNAL_THRESHOLD, else 0.
    """
    model = _load_model()
    X = build_feature_matrix_v3(close_df, vix_df)
    if X.empty:
        return {}, pd.Series(dtype=float)

    latest_date = X.index.get_level_values("date").max()
    X_today = X.xs(latest_date, level="date")
    if X_today.empty:
        return {}, pd.Series(dtype=float)

    raw_proba_arr = model.predict_proba(X_today)[:, 1]
    raw_proba = pd.Series(raw_proba_arr, index=X_today.index, name=latest_date)

    hist_with_today = append_today_proba(proba_history, raw_proba)
    ema_proba = hist_with_today.ewm(span=EMA_SPAN).mean().iloc[-1]

    threshold = config.SIGNAL_THRESHOLD
    # Top-K selection: only the top-K highest ema_probas (and above threshold) get signal=1
    eligible = ema_proba[ema_proba >= threshold]
    if len(eligible) == 0:
        chosen = set()
    else:
        chosen = set(eligible.sort_values(ascending=False).head(top_k).index)

    results = {}
    for ticker in raw_proba.index:
        raw = float(raw_proba[ticker])
        ema = float(ema_proba.get(ticker, raw))
        rank = ema_proba.rank(ascending=False).get(ticker, np.nan)
        results[ticker] = {
            "raw_proba": round(raw, 4),
            "ema_proba": round(ema, 4),
            "signal": 1 if ticker in chosen else 0,
            "rank": int(rank) if not pd.isna(rank) else 999,
        }
    return results, raw_proba


# compute_rolling_stats is identical to inference.py's version, since
# the same Friday-rebalance methodology applies. Production switch can use
# either module's rolling_stats interchangeably.
def compute_rolling_stats(close_df, vix_df, proba_history, window=ROLLING_WINDOW):
    """Per-ETF rolling stats — same algorithm as inference.compute_rolling_stats."""
    model = _load_model()
    X = build_feature_matrix_v3(close_df, vix_df)
    if X.empty:
        return pd.DataFrame()

    dates = sorted(X.index.get_level_values("date").unique())
    needed = window + HOLD_DAYS + 1
    hist_dates = dates[-needed:-HOLD_DAYS] if len(dates) > needed else dates[:-HOLD_DAYS]
    if not hist_dates:
        return pd.DataFrame()

    warmup_start = max(0, len(dates) - needed - EMA_SPAN * 5)
    warmup_dates = set(dates[warmup_start:-HOLD_DAYS])
    X_hist = X[X.index.get_level_values("date").isin(warmup_dates)]
    proba_arr = model.predict_proba(X_hist)[:, 1]
    proba_s = pd.Series(proba_arr, index=X_hist.index)
    proba_df = proba_s.unstack(level=1)
    smooth = proba_df.ewm(span=EMA_SPAN).mean()

    friday_dates = [d for d in hist_dates if pd.Timestamp(d).dayofweek == 4]
    if not friday_dates:
        return pd.DataFrame()

    fwd_ret_5d = close_df.pct_change(HOLD_DAYS).shift(-HOLD_DAYS)
    rows = []
    for ticker in smooth.columns:
        if ticker not in fwd_ret_5d.columns:
            continue
        sig = smooth[ticker].reindex(friday_dates).dropna()
        ret = fwd_ret_5d[ticker].reindex(friday_dates).dropna()
        common = sig.index.intersection(ret.index)
        if common.empty:
            continue
        active_idx = common[sig.reindex(common) >= config.SIGNAL_THRESHOLD]
        if len(active_idx) < MIN_ACTIVE_FRIDAYS:
            continue
        r = ret.loc[active_idx]
        wins = r[r > 0]; losses = r[r <= 0]
        winrate = len(wins) / len(r)
        avg_win = float(wins.mean()) if len(wins) else 0.008
        avg_loss = float(abs(losses.mean())) if len(losses) else 0.006
        avg_win = max(avg_win, 0.004); avg_loss = max(avg_loss, 0.003)
        payoff = avg_win / avg_loss
        rows.append({
            "ticker": ticker, "winrate": round(winrate, 4),
            "avg_win": round(avg_win, 5), "avg_loss": round(avg_loss, 5),
            "payoff": round(payoff, 3), "sample_n": len(r),
        })
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()
