"""Intraday technical factors computed on 5-minute OHLCV bars.

Indicators implemented (no external TA library dependency — pure pandas/numpy):
  EMA5, EMA20     — trend direction
  VWAP + SD bands — institutional price benchmark + normalized deviation
  RSI(14)         — momentum (standard)
  StochRSI %K     — faster momentum confirmation (14/14/3)
  ADX(14)         — trend strength gate; filters ranging-market whipsaws
  OBV slope       — volume-confirmed direction (3-bar rate of change)
  vol_ratio       — current bar vs 20-bar avg volume
  minutes_from_open — time-of-day: suppresses Opening Range (first 30 min)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── VWAP ──────────────────────────────────────────────────────────────────────

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Daily VWAP — resets at session open each day."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].replace(0, 1)
    return (typical * vol).cumsum() / vol.cumsum()


def compute_vwap_sd(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """VWAP + 1-SD band.

    Returns (vwap, vwap_sd) where vwap_sd is the running 1-standard-deviation
    of typical price around VWAP, volume-weighted.  Used to normalize VWAP
    deviation: vwap_dev_sd = (close - vwap) / vwap_sd.
    """
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].replace(0, 1)
    cum_vol = vol.cumsum()
    vwap = (typical * vol).cumsum() / cum_vol
    # Running volume-weighted variance of typical price vs VWAP
    variance = ((typical - vwap) ** 2 * vol).cumsum() / cum_vol
    sd = variance.apply(lambda v: max(v, 1e-8) ** 0.5)
    return vwap, sd


# ── RSI ───────────────────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Standard Wilder RSI using EWM (alpha = 1/period).

    When avg_loss == 0 (all bars were up), RSI = 100 by definition.
    """
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.where(loss > 0, float("nan"))
    rsi = 100 - 100 / (1 + rs)
    return rsi.where(loss > 0, 100.0)  # loss=0 → RSI=100 (fully bullish)


def compute_stochrsi(
    series: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_smooth: int = 3,
) -> pd.Series:
    """Stochastic RSI %K (smoothed).

    RSI(14) on 5-min bars = 70-minute lookback, which can lag momentum shifts.
    StochRSI normalises RSI within its own 14-bar range and then smooths %K
    with a 3-bar SMA — giving a faster but less noisy reading.

    Threshold for trend confirmation (not mean-reversion):
      > 50 = RSI in upper half of recent range → momentum up → LONG
      < 50 = RSI in lower half              → momentum down → SHORT
    """
    rsi = compute_rsi(series, rsi_period)
    rsi_min = rsi.rolling(stoch_period, min_periods=max(stoch_period // 2, 1)).min()
    rsi_max = rsi.rolling(stoch_period, min_periods=max(stoch_period // 2, 1)).max()
    rsi_range = (rsi_max - rsi_min).replace(0, float("nan"))
    raw_k = 100 * (rsi - rsi_min) / rsi_range
    raw_k = raw_k.fillna(50.0)  # flat RSI range → neutral (50), not a signal
    return raw_k.rolling(k_smooth, min_periods=1).mean()


# ── ADX ───────────────────────────────────────────────────────────────────────

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ADX(14) — trend-strength filter.

    ADX < 20: market is ranging → EMA crossovers are noise, do not trade.
    ADX > 25: trending → crossover signals are reliable.

    Uses Wilder smoothing (EWM alpha = 1/period) matching the original formula.
    """
    hi, lo, cl = df["High"], df["Low"], df["Close"]
    prev_cl = cl.shift(1)

    # True Range
    tr = pd.concat([hi - lo, (hi - prev_cl).abs(), (lo - prev_cl).abs()], axis=1).max(axis=1)

    # Directional Movement
    plus_dm = (hi - hi.shift(1)).clip(lower=0)
    minus_dm = (lo.shift(1) - lo).clip(lower=0)
    # Zero out whichever DM is not dominant on each bar
    both_pos = (plus_dm > 0) & (minus_dm > 0)
    use_plus = plus_dm >= minus_dm
    plus_dm = plus_dm.where(~both_pos | use_plus, 0.0)
    minus_dm = minus_dm.where(~both_pos | ~use_plus, 0.0)

    alpha = 1.0 / period
    atr_s = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s.replace(0, float("nan"))
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s.replace(0, float("nan"))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.ewm(alpha=alpha, adjust=False).mean()


# ── OBV ───────────────────────────────────────────────────────────────────────

def compute_obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume — cumulative signed volume."""
    direction = np.sign(df["Close"].diff().fillna(0))
    return (direction * df["Volume"]).cumsum()


# ── Time-of-day ───────────────────────────────────────────────────────────────

def compute_minutes_from_open(df: pd.DataFrame) -> pd.Series:
    """Minutes elapsed since 9:30 AM ET.  Requires ET-timezone-aware index.

    First 30 minutes (0–29 min) = Opening Range; signals there are suppressed
    in the signal generator because bid-ask spreads are widest and institutional
    order flow is most unpredictable.
    """
    idx = df.index
    try:
        et_idx = idx.tz_convert("America/New_York")
        minutes = (et_idx.hour - 9) * 60 + et_idx.minute - 30
        return pd.Series(minutes.values, index=df.index)
    except Exception:
        # Fallback: no suppression if timezone conversion fails
        return pd.Series(60, index=df.index)  # 60 min → safely past ORB


# ── Master feature builder ────────────────────────────────────────────────────

def compute_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all intraday features to an OHLCV DataFrame.

    New columns added:
      ema5, ema20, vwap, vwap_sd, vwap_dev_sd,
      rsi14, stochrsi_k,
      adx14,
      obv, obv_slope,
      vol_ratio,
      minutes_from_open
    """
    out = df.copy()

    # Trend EMAs
    out["ema5"] = out["Close"].ewm(span=5, adjust=False).mean()
    out["ema20"] = out["Close"].ewm(span=20, adjust=False).mean()

    # VWAP + SD bands
    out["vwap"], vwap_sd = compute_vwap_sd(out)
    out["vwap_sd"] = vwap_sd
    out["vwap_dev_sd"] = (out["Close"] - out["vwap"]) / out["vwap_sd"]

    # RSI + StochRSI
    out["rsi14"] = compute_rsi(out["Close"], 14)
    out["stochrsi_k"] = compute_stochrsi(out["Close"], rsi_period=14, stoch_period=14, k_smooth=3)

    # ADX trend strength
    out["adx14"] = compute_adx(out, period=14)

    # OBV + 3-bar slope (positive = accumulation, negative = distribution)
    out["obv"] = compute_obv(out)
    out["obv_slope"] = out["obv"].diff(3)

    # Volume ratio vs 20-bar average
    vol_avg = out["Volume"].rolling(20, min_periods=5).mean()
    out["vol_ratio"] = out["Volume"] / vol_avg.replace(0, float("nan"))

    # Time-of-day
    out["minutes_from_open"] = compute_minutes_from_open(out)

    return out
