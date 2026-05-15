"""Intraday technical factors computed on 5-minute OHLCV bars."""
from __future__ import annotations

import pandas as pd


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Daily VWAP — resets at market open each day."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].replace(0, 1)
    cumtp_vol = (typical * vol).cumsum()
    cumvol = vol.cumsum()
    return cumtp_vol / cumvol


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)


def compute_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA5, EMA20, VWAP, RSI14, vol_ratio to an OHLCV DataFrame."""
    out = df.copy()
    out["ema5"] = out["Close"].ewm(span=5, adjust=False).mean()
    out["ema20"] = out["Close"].ewm(span=20, adjust=False).mean()
    out["vwap"] = compute_vwap(out)
    out["rsi14"] = compute_rsi(out["Close"], 14)
    vol_avg = out["Volume"].rolling(20, min_periods=5).mean()
    out["vol_ratio"] = out["Volume"] / vol_avg.replace(0, float("nan"))
    return out
