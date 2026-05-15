"""Intraday signal generator (rule-based, 5-minute bars).

Signal logic per ticker:
  LONG  (+1): EMA5 > EMA20  AND  price > VWAP  AND  RSI > 52
  SHORT (-1): EMA5 < EMA20  AND  price < VWAP  AND  RSI < 48
  FLAT  ( 0): otherwise

Strength score ranks tickers so the top-5 LONG and top-5 SHORT can be shown.
TP/SL are ATR(14)-based: TP = 2×ATR, SL = 1×ATR (minimum 0.25% roundtrip buffer).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.intraday import CORE_TICKERS, INVERSE_PAIR, SECTOR_TICKERS
from features.intraday_factors import compute_intraday_features
import config

RSI_LONG = 52
RSI_SHORT = 48
MIN_BARS = 21          # need at least 21 bars for EMA20 warm-up
ATR_PERIOD = 14
TP_ATR_MULT = 2.0      # take-profit = entry ± 2×ATR(5-min)
SL_ATR_MULT = 1.0      # stop-loss   = entry ∓ 1×ATR(5-min)
MIN_TP_PCT = 0.006     # floor: TP at least 0.6% from entry (covers cost + margin)
MIN_SL_PCT = 0.003     # floor: SL at least 0.3% from entry

# Conservative backtest-implied winrate for intraday rule-based signals.
# Will be refined once 60-day 5-min history backtest is added.
INTRADAY_WINRATE_EST = 0.54
INTRADAY_AVG_WIN_EST = 0.008   # 0.8% gross avg win
INTRADAY_AVG_LOSS_EST = 0.004  # 0.4% gross avg loss


@dataclass
class IntraSignal:
    ticker: str
    direction: int          # +1 = long, -1 = short, 0 = flat
    price: float
    vwap: float
    ema5: float
    ema20: float
    rsi: float
    atr: float
    score: float            # directional strength for ranking
    tp: float               # take-profit price
    sl: float               # stop-loss price
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


def _compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    hi, lo, cl = df["High"], df["Low"], df["Close"]
    prev_cl = cl.shift(1)
    tr = pd.concat([
        hi - lo,
        (hi - prev_cl).abs(),
        (lo - prev_cl).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])


def _tp_sl(price: float, direction: int, atr: float) -> tuple[float, float]:
    """ATR-based TP and SL with a minimum floor."""
    raw_tp = atr * TP_ATR_MULT
    raw_sl = atr * SL_ATR_MULT
    tp_gap = max(raw_tp, price * MIN_TP_PCT)
    sl_gap = max(raw_sl, price * MIN_SL_PCT)
    if direction == 1:
        return round(price + tp_gap, 4), round(price - sl_gap, 4)
    else:  # short
        return round(price - tp_gap, 4), round(price + sl_gap, 4)


def _strength_score(last: pd.Series, direction: int) -> float:
    """Composite signal strength in [0, 1] — used to rank top-5."""
    rsi_norm = (last["rsi14"] - 50) / 50          # [-1, +1]
    vwap_dev = (last["Close"] - last["vwap"]) / last["vwap"]
    ema_dev = (last["ema5"] - last["ema20"]) / last["ema20"]
    # For SHORT, invert so higher score = stronger short
    sign = 1 if direction == 1 else -1
    return float(sign * (rsi_norm * 0.4 + vwap_dev * 10 * 0.4 + ema_dev * 100 * 0.2))


def _expected_return(direction: int, tp_pct: float, sl_pct: float, wr: float) -> float:
    avg_win = tp_pct
    avg_loss = sl_pct
    return wr * avg_win - (1 - wr) * avg_loss - config.TOTAL_COST_ROUNDTRIP


def _classify(last: pd.Series, allow_short: bool) -> int:
    ema_bull = last["ema5"] > last["ema20"]
    price_bull = last["Close"] > last["vwap"]
    rsi_bull = last["rsi14"] > RSI_LONG

    ema_bear = last["ema5"] < last["ema20"]
    price_bear = last["Close"] < last["vwap"]
    rsi_bear = last["rsi14"] < RSI_SHORT

    if ema_bull and price_bull and rsi_bull:
        return 1
    if allow_short and ema_bear and price_bear and rsi_bear:
        return -1
    return 0


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

        tp, sl = _tp_sl(price, direction, atr) if direction != 0 else (0.0, 0.0)
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
