"""Phase 3: LightGBM daily inference for production pipeline.

Loads walk-forward trained model, builds today's features, and returns
per-ticker probabilities and rolling backtest statistics.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import config
from models.train_lgbm import build_feature_matrix

log = logging.getLogger(__name__)

WEIGHTS_DIR = Path("models/weights")
PROBA_HISTORY_PATH = Path("data/logs/proba_history.parquet")
PROBA_HISTORY_DAYS = 20   # keep last N days of raw probabilities for EMA
EMA_SPAN = 5
ROLLING_WINDOW = 60       # trading days for per-ETF rolling stats (~12 Fridays, responsive)
HOLD_DAYS = 5             # weekly hold: Friday-to-Friday
MIN_ACTIVE_FRIDAYS = 8    # minimum Friday signal activations for reliable stats (4 was too few)


# ── Model loading ─────────────────────────────────────────────────────────────

_cached_model = None


def _load_model():
    global _cached_model
    if _cached_model is None:
        model_path = WEIGHTS_DIR / "lgbm_phase3.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"모델 없음: {model_path}. Phase 3 학습 먼저 실행.")
        with open(model_path, "rb") as f:
            _cached_model = pickle.load(f)
        log.info("LightGBM model loaded from %s", model_path)
    return _cached_model


# ── Probability history ───────────────────────────────────────────────────────

def load_proba_history() -> pd.DataFrame:
    """Load rolling probability history. Empty DataFrame if not exists."""
    if PROBA_HISTORY_PATH.exists():
        return pd.read_parquet(PROBA_HISTORY_PATH)
    return pd.DataFrame()


def save_proba_history(history: pd.DataFrame) -> None:
    """Persist probability history (keep last PROBA_HISTORY_DAYS rows)."""
    PROBA_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    trimmed = history.tail(PROBA_HISTORY_DAYS)
    trimmed.to_parquet(PROBA_HISTORY_PATH)


def append_today_proba(history: pd.DataFrame, today_proba: pd.Series) -> pd.DataFrame:
    """Append today's raw probability row to history and return updated DataFrame."""
    today_df = today_proba.to_frame().T
    today_df.index = pd.DatetimeIndex([today_proba.name if today_proba.name else pd.Timestamp.today().normalize()])
    return pd.concat([history, today_df]).drop_duplicates().tail(PROBA_HISTORY_DAYS)


# ── Daily inference ───────────────────────────────────────────────────────────

def score_today(
    close_df: pd.DataFrame,
    vix_df: pd.DataFrame | None,
    proba_history: pd.DataFrame,
) -> dict[str, dict]:
    """Compute per-ticker model output for today.

    Returns dict: {ticker: {"raw_proba": float, "ema_proba": float, "signal": int}}
    signal: +1 (long) if ema_proba >= 0.53, 0 (flat) otherwise.
    """
    model = _load_model()
    X = build_feature_matrix(close_df, vix_df)
    if X.empty:
        return {}

    latest_date = X.index.get_level_values("date").max()
    X_today = X.xs(latest_date, level="date")
    if X_today.empty:
        return {}

    raw_proba_arr = model.predict_proba(X_today)[:, 1]
    raw_proba = pd.Series(raw_proba_arr, index=X_today.index, name=latest_date)

    # Compute EMA-5 using history
    history_with_today = append_today_proba(proba_history, raw_proba)
    ema_proba = history_with_today.ewm(span=EMA_SPAN).mean().iloc[-1]

    threshold = config.SIGNAL_THRESHOLD
    results: dict[str, dict] = {}
    for ticker in raw_proba.index:
        raw = float(raw_proba[ticker])
        ema = float(ema_proba.get(ticker, raw))
        results[ticker] = {
            "raw_proba": round(raw, 4),
            "ema_proba": round(ema, 4),
            "signal": 1 if ema >= threshold else 0,
        }
    return results, raw_proba


def compute_rolling_stats(
    close_df: pd.DataFrame,
    vix_df: pd.DataFrame | None,
    proba_history: pd.DataFrame,
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Per-ETF rolling stats: Friday-signal → 5-day (weekly) forward returns.

    Measures what actually happens in paper trading: enter at Friday close,
    exit next Friday close. Uses last `window` trading days (~window/5 Fridays).
    Returns DataFrame indexed by ticker with: winrate, avg_win, avg_loss, payoff, sample_n
    """
    model = _load_model()
    X = build_feature_matrix(close_df, vix_df)
    if X.empty:
        return pd.DataFrame()

    dates = sorted(X.index.get_level_values("date").unique())
    # Need window + HOLD_DAYS extra days so the last Friday has a realized exit
    needed = window + HOLD_DAYS + 1
    hist_dates = dates[-needed:-HOLD_DAYS] if len(dates) > needed else dates[:-HOLD_DAYS]
    if not hist_dates:
        return pd.DataFrame()

    # Compute EMA-smoothed signal over hist window (need some warm-up, use full range)
    warmup_start = max(0, len(dates) - needed - EMA_SPAN * 5)
    warmup_dates = set(dates[warmup_start:-HOLD_DAYS])
    X_hist = X[X.index.get_level_values("date").isin(warmup_dates)]
    proba_arr = model.predict_proba(X_hist)[:, 1]
    proba_s = pd.Series(proba_arr, index=X_hist.index)

    proba_df = proba_s.unstack(level=1)
    smooth = proba_df.ewm(span=EMA_SPAN).mean()

    # Only evaluate on Fridays in the actual hist window
    hist_set = set(hist_dates)
    friday_dates = [d for d in hist_dates if pd.Timestamp(d).dayofweek == 4]
    if not friday_dates:
        return pd.DataFrame()

    # 5-day forward return: enter Friday close, exit next Friday close
    fwd_ret_5d = close_df.pct_change(HOLD_DAYS).shift(-HOLD_DAYS)

    rows: list[dict] = []
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
        wins = r[r > 0]
        losses = r[r <= 0]

        winrate = len(wins) / len(r)
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.008
        avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.006
        avg_win = max(avg_win, 0.004)
        avg_loss = max(avg_loss, 0.003)
        payoff = avg_win / avg_loss

        rows.append({
            "ticker": ticker,
            "winrate": round(winrate, 4),
            "avg_win": round(avg_win, 5),
            "avg_loss": round(avg_loss, 5),
            "payoff": round(payoff, 3),
            "sample_n": len(r),
        })

    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


# ── Legacy stubs (Phase 0) kept for backward compat ──────────────────────────

def score_direction(features: pd.DataFrame) -> pd.Series:
    """Deprecated stub — use score_today() for Phase 3+ inference."""
    return pd.Series(0.0, index=features.columns if hasattr(features, "columns") else [])


def score_confidence(features: pd.DataFrame) -> pd.Series:
    """Deprecated stub — use score_today() for Phase 3+ inference."""
    return pd.Series(0.5, index=features.columns if hasattr(features, "columns") else [])
