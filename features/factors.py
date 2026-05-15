"""Technical factor computation. All factors are strictly causal — no look-ahead.

Adding a new factor: run pytest tests/data/test_lookahead.py after every change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_factors(close: pd.Series, volume: pd.Series | None = None) -> pd.DataFrame:
    """Compute technical factors for a single ETF close-price series.

    Returns a DataFrame indexed on the same dates, NaN rows dropped.
    Only closed bars are used; no forward-looking operations.
    """
    df = pd.DataFrame(index=close.index)

    # Multi-period momentum (pure look-back, safe)
    for period in [1, 5, 10, 21, 63]:
        df[f"mom_{period}d"] = close.pct_change(period)

    # Short-term mean reversion
    df["rsi_14"] = _rsi(close, 14)
    df["zscore_20"] = _zscore(close, 20)

    # Volatility regime
    ret = close.pct_change()
    df["vol_5d"] = ret.rolling(5).std() * np.sqrt(252)
    df["vol_21d"] = ret.rolling(21).std() * np.sqrt(252)
    df["vol_ratio"] = df["vol_5d"] / df["vol_21d"].replace(0, np.nan)

    # Trend strength (close-only ADX proxy)
    df["trend_strength"] = _adx_proxy(close, 14)

    if volume is not None:
        df["vol_anomaly"] = _volume_anomaly(volume, 20)

    return df.dropna()


def compute_cross_section(close_df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional relative-strength ranks across ETFs.

    close_df: DataFrame with ETF tickers as columns, dates as index.
    Returns same shape DataFrame with rank-normalized values in [0, 1].
    """
    ret_21d = close_df.pct_change(21)
    # Rank within each row (date) — higher return = higher rank
    return ret_21d.rank(axis=1, pct=True)


# ── Private helpers ───────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _zscore(close: pd.Series, window: int) -> pd.Series:
    mu = close.rolling(window).mean()
    sigma = close.rolling(window).std()
    return (close - mu) / sigma.replace(0, np.nan)


def _adx_proxy(close: pd.Series, period: int) -> pd.Series:
    """Simplified trend-strength proxy using close price only."""
    momentum = close.diff(period).abs()
    total_range = close.rolling(period).apply(lambda x: x.max() - x.min(), raw=True)
    return (momentum / total_range.replace(0, np.nan)).rolling(period).mean()


def _volume_anomaly(volume: pd.Series, window: int) -> pd.Series:
    """Volume z-score relative to rolling mean."""
    mu = volume.rolling(window).mean()
    sigma = volume.rolling(window).std()
    return (volume - mu) / sigma.replace(0, np.nan)
