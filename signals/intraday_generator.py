"""Intraday signal generator (rule-based, 5-minute bars).

Signal logic per ticker — ALL conditions must hold simultaneously:
  LONG  (+1):  EMA5 > EMA20          (uptrend)
               AND price > VWAP      (above institutional benchmark)
               AND RSI > 55          (momentum, tighter than 50 to reduce noise)
               AND StochRSI_K > 50   (RSI in upper half of recent range)
               AND ADX > 25          (trending, not ranging — prevents crossover whipsaws)
               AND OBV slope > 0     (accumulation — volume confirms direction)
               AND vol_ratio > 1.0   (above-average volume on this bar)
               AND minutes_from_open >= 30 (past Opening Range — avoid first-30-min chaos)

  SHORT (-1):  inverse of all LONG conditions (core ETFs only via inverse ETFs)

  FLAT  ( 0):  any condition fails

TP/SL sizing:
  ATR(14)-based: TP = 2×ATR, SL = 1×ATR
  Floor: TP ≥ 1.0%, SL ≥ 0.4% (ensures positive expected value after 0.25% roundtrip cost)
  High-vol regime (VIX ≥ 20): widen multipliers to TP=2.5×, SL=1.5×

Strength score for top-5 ranking:
  Composite of VWAP deviation (normalized by SD), EMA spread, and RSI momentum.
  All components are dimensionally consistent (unit-free or %-based).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.intraday import CORE_TICKERS, INVERSE_PAIR, SECTOR_TICKERS
from features.intraday_factors import compute_intraday_features
import config

# ── Thresholds ────────────────────────────────────────────────────────────────

RSI_LONG = 55           # tighter than 50-boundary; reduces false positives
RSI_SHORT = 45
STOCHRSI_LONG = 50      # RSI in upper half of recent range = momentum up
STOCHRSI_SHORT = 50
ADX_TREND_MIN = 25      # below this ADX = ranging market; EMA signals unreliable
OBV_SLOPE_BARS = 3      # bars for OBV slope (3-bar = 15-min rate of change)
ORB_MINUTES = 30        # Opening Range: first 30 min (9:30-10:00 ET) is suppressed

MIN_BARS = 40           # minimum bars before any signal is evaluated
ATR_PERIOD = 14

# TP/SL ATR multipliers — base regime (VIX < 20)
TP_ATR_MULT_BASE = 2.0
SL_ATR_MULT_BASE = 1.0

# TP/SL ATR multipliers — caution/fear regime (VIX ≥ 20)
# Widen stops to avoid premature stop-out during high-volatility swings
TP_ATR_MULT_HIGHVOL = 2.5
SL_ATR_MULT_HIGHVOL = 1.5

# Minimum floors regardless of ATR (positive EV math):
# 0.54×1.0% - 0.46×0.4% - 0.25% = +0.106%/trade
MIN_TP_PCT = 0.010
MIN_SL_PCT = 0.004

# Conservative unvalidated winrate estimate (will be replaced by 60-day backtest)
INTRADAY_WINRATE_EST = 0.54
INTRADAY_AVG_WIN_EST = 0.010
INTRADAY_AVG_LOSS_EST = 0.004


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class IntraSignal:
    ticker: str
    direction: int          # +1 = long, -1 = short, 0 = flat
    price: float
    vwap: float
    ema5: float
    ema20: float
    rsi: float
    stochrsi_k: float
    adx: float
    atr: float
    score: float            # directional strength for ranking
    tp: float
    sl: float
    winrate: float
    exp_return: float       # net expected return (after roundtrip cost)
    vix: float | None

    @property
    def action_ticker(self) -> str:
        if self.direction == -1 and self.ticker in INVERSE_PAIR:
            return INVERSE_PAIR[self.ticker]
        return self.ticker

    @property
    def action_label(self) -> str:
        if self.direction == 1:
            return "LONG"
        if self.direction == -1:
            return "SHORT"
        return "FLAT"

    @property
    def regime(self) -> str:
        if self.vix is None:
            return "normal"
        if self.vix >= 30:
            return "fear"
        if self.vix >= 20:
            return "caution"
        return "normal"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    hi, lo, cl = df["High"], df["Low"], df["Close"]
    prev_cl = cl.shift(1)
    tr = pd.concat([
        hi - lo,
        (hi - prev_cl).abs(),
        (lo - prev_cl).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])


def _tp_sl(
    price: float,
    direction: int,
    atr: float,
    vix: float | None,
) -> tuple[float, float]:
    """ATR-based TP/SL.  Multipliers widen in high-vol regime (VIX ≥ 20)."""
    high_vol = vix is not None and vix >= 20
    tp_mult = TP_ATR_MULT_HIGHVOL if high_vol else TP_ATR_MULT_BASE
    sl_mult = SL_ATR_MULT_HIGHVOL if high_vol else SL_ATR_MULT_BASE
    tp_gap = max(atr * tp_mult, price * MIN_TP_PCT)
    sl_gap = max(atr * sl_mult, price * MIN_SL_PCT)
    if direction == 1:
        return round(price + tp_gap, 4), round(price - sl_gap, 4)
    else:
        return round(price - tp_gap, 4), round(price + sl_gap, 4)


def _strength_score(last: pd.Series, direction: int) -> float:
    """Composite directional strength in comparable units — used for ranking only.

    All components are dimensionally consistent:
      vwap_dev_sd : (close - VWAP) / VWAP_SD → z-score units [-∞, +∞]
      ema_dev_pct : (EMA5 - EMA20) / EMA20    → fraction
      rsi_center  : (RSI - 50) / 50           → [-1, +1]
      stochrsi_c  : (StochRSI_K - 50) / 50   → [-1, +1]
    """
    vwap_dev_sd = float(last.get("vwap_dev_sd", 0.0))
    ema_dev_pct = (last["ema5"] - last["ema20"]) / (last["ema20"] + 1e-10)
    rsi_center = (last["rsi14"] - 50) / 50
    stochrsi_c = (last.get("stochrsi_k", 50) - 50) / 50
    sign = 1 if direction == 1 else -1
    # Weights: VWAP deviation 35%, EMA spread 30%, RSI 20%, StochRSI 15%
    raw = (
        vwap_dev_sd * 0.35
        + ema_dev_pct * 100 * 0.30   # scale ema_dev_pct to ~same range as vwap_dev_sd
        + rsi_center * 0.20
        + stochrsi_c * 0.15
    )
    return float(sign * raw)


def _expected_return(direction: int, tp_pct: float, sl_pct: float, wr: float) -> float:
    return wr * tp_pct - (1 - wr) * sl_pct - config.TOTAL_COST_ROUNDTRIP


def _classify(last: pd.Series, allow_short: bool) -> int:
    """Multi-condition classifier — ALL gates must pass to generate a signal."""

    # ── Gate 1: Opening Range suppression (first 30 min is noise) ─────────────
    minutes = float(last.get("minutes_from_open", 60))
    if minutes < ORB_MINUTES:
        return 0

    # ── Gate 2: Trend strength — not in a ranging market ──────────────────────
    adx = float(last.get("adx14", 0.0))
    if adx < ADX_TREND_MIN:
        return 0

    # ── Gate 3: Volume confirmation ────────────────────────────────────────────
    vol_ratio = float(last.get("vol_ratio", 0.0))
    if vol_ratio <= 1.0:
        return 0

    obv_slope = float(last.get("obv_slope", 0.0))

    # ── Gate 4–7: LONG conditions ──────────────────────────────────────────────
    ema_bull = last["ema5"] > last["ema20"]
    price_bull = last["Close"] > last["vwap"]
    rsi_bull = last["rsi14"] > RSI_LONG
    stochrsi_bull = float(last.get("stochrsi_k", 50)) > STOCHRSI_LONG
    obv_bull = obv_slope > 0

    if ema_bull and price_bull and rsi_bull and stochrsi_bull and obv_bull:
        return 1

    # ── Gate 4–7: SHORT conditions ─────────────────────────────────────────────
    if allow_short:
        ema_bear = last["ema5"] < last["ema20"]
        price_bear = last["Close"] < last["vwap"]
        rsi_bear = last["rsi14"] < RSI_SHORT
        stochrsi_bear = float(last.get("stochrsi_k", 50)) < STOCHRSI_SHORT
        obv_bear = obv_slope < 0

        if ema_bear and price_bear and rsi_bear and stochrsi_bear and obv_bear:
            return -1

    return 0


# ── Public API ────────────────────────────────────────────────────────────────

def generate_signals(
    ohlcv: dict[str, pd.DataFrame],
    vix_level: float | None,
) -> dict[str, IntraSignal]:
    """Compute current signal for each ticker. Returns {ticker: IntraSignal}."""
    fear = vix_level is not None and vix_level >= 30
    tickers_to_eval = CORE_TICKERS + ([] if fear else SECTOR_TICKERS)

    signals: dict[str, IntraSignal] = {}
    for ticker in tickers_to_eval:
        df = ohlcv.get(ticker)
        if df is None or len(df) < MIN_BARS:
            continue

        feat = compute_intraday_features(df)
        last = feat.iloc[-1]
        allow_short = ticker in CORE_TICKERS
        direction = _classify(last, allow_short)
        price = float(last["Close"])
        atr = _compute_atr(df)

        tp, sl = _tp_sl(price, direction, atr, vix_level) if direction != 0 else (0.0, 0.0)
        tp_pct = abs(tp - price) / price if price > 0 else MIN_TP_PCT
        sl_pct = abs(sl - price) / price if price > 0 else MIN_SL_PCT
        exp_ret = _expected_return(direction, tp_pct, sl_pct, INTRADAY_WINRATE_EST)
        score = _strength_score(last, direction) if direction != 0 else 0.0

        signals[ticker] = IntraSignal(
            ticker=ticker,
            direction=direction,
            price=price,
            vwap=round(float(last["vwap"]), 4),
            ema5=round(float(last["ema5"]), 4),
            ema20=round(float(last["ema20"]), 4),
            rsi=round(float(last["rsi14"]), 2),
            stochrsi_k=round(float(last.get("stochrsi_k", float("nan"))), 2),
            adx=round(float(last.get("adx14", float("nan"))), 2),
            atr=round(atr, 4),
            score=round(score, 5),
            tp=tp,
            sl=sl,
            winrate=INTRADAY_WINRATE_EST,
            exp_return=round(exp_ret, 5),
            vix=vix_level,
        )

    return signals


def top_signals(
    signals: dict[str, IntraSignal],
    n: int = 5,
) -> tuple[list[IntraSignal], list[IntraSignal]]:
    """Return (top_n_longs, top_n_shorts) sorted by strength score descending."""
    longs = sorted(
        [s for s in signals.values() if s.direction == 1],
        key=lambda s: s.score, reverse=True,
    )[:n]
    shorts = sorted(
        [s for s in signals.values() if s.direction == -1],
        key=lambda s: s.score, reverse=True,
    )[:n]
    return longs, shorts


def diff_signals(
    prev: dict[str, IntraSignal],
    curr: dict[str, IntraSignal],
) -> list[tuple[IntraSignal | None, IntraSignal]]:
    """Return (prev_sig, curr_sig) pairs where direction changed."""
    changed = []
    all_tickers = set(prev) | set(curr)
    for t in all_tickers:
        p = prev.get(t)
        c = curr.get(t)
        if c is None:
            continue
        prev_dir = p.direction if p else None
        if prev_dir != c.direction:
            changed.append((p, c))
    return changed
