"""Trend-following core + fear-dip leverage tilt — PRIMARY return engine.

Backtests (2026-05-22) showed the conviction/fear-dip signal apparatus, while
high win-rate, is invested <10% of the time and badly trails buy-and-hold in
absolute return. A simple 200-day-SMA trend core on SPY captures most of the
drift with half the drawdown (+78.6% / Sharpe 0.99 / MDD -22% full OOS;
+45% / 1.39 / -10.5% in 2024+), and beats every blend with the satellite
signals. So the trend core is the engine.

The validated enhancement (user opted into leverage): on fear-dip days that
fall in an uptrend (the model's panic-bounce signal), tilt ~15% of capital
into SPXL (3x S&P) for ~1.3x effective exposure. Re-validated with REAL SPXL
prices (decay included): +92.2% full OOS / +48.7% 2024+, Sharpe maintained,
MDD unchanged. 1.5x (25% SPXL) returns more but with more decay drag.

Daily exposure recommendation:
    SPY > 200d SMA  &  fear-dip active  -> SPY (1-w) + SPXL w   (~1.3x, tilt)
    SPY > 200d SMA                       -> SPY 100%             (core long)
    SPY <= 200d SMA &  fear-dip active   -> SPY 100%             (dip re-entry, no leverage in downtrend)
    SPY <= 200d SMA                      -> cash                 (trend off)
"""
from __future__ import annotations

import pandas as pd

import config

SMA_WINDOW = 200
CORE_TICKER = "SPY"
LEVER_TICKER = "SPXL"        # Direxion 3x S&P bull
LEVER_MULT = 3.0
# fraction of capital shifted to SPXL on uptrend fear-dip days.
# 0.15 -> ~1.3x effective (recommended); 0.25 -> ~1.5x.
TILT_SPXL_WEIGHT = float(getattr(config, "TREND_CORE_TILT_WEIGHT", 0.15))


def effective_exposure(spxl_weight: float) -> float:
    return (1 - spxl_weight) * 1.0 + spxl_weight * LEVER_MULT


def evaluate(close_df: pd.DataFrame, fear_dip_active: bool) -> dict:
    """Return today's core exposure recommendation.

    close_df: OHLCV Close frame (needs CORE_TICKER column).
    fear_dip_active: True if a fear-dip position is currently open (within its
                     10-day window) — the panic-bounce tilt trigger.
    """
    spy = close_df[CORE_TICKER].dropna()
    if len(spy) < SMA_WINDOW + 1:
        return {"coreOn": None, "note": "200일 SMA 계산에 데이터 부족"}
    price = float(spy.iloc[-1])
    sma = float(spy.rolling(SMA_WINDOW).mean().iloc[-1])
    core_on = price > sma
    w = TILT_SPXL_WEIGHT

    if core_on and fear_dip_active:
        spy_w, spxl_w = 1 - w, w
        regime, note = "uptrend_panic", (
            f"상승추세 + 공포 — {LEVER_TICKER} {w*100:.0f}% 틸트 (≈{effective_exposure(w):.2f}x)")
    elif core_on:
        spy_w, spxl_w = 1.0, 0.0
        regime, note = "uptrend", "상승추세 — SPY 100% 보유"
    elif fear_dip_active:
        spy_w, spxl_w = 1.0, 0.0
        regime, note = "downtrend_panic", "하락추세 + 공포 — SPY 100% (저점 재진입, 무레버리지)"
    else:
        spy_w, spxl_w = 0.0, 0.0
        regime, note = "cash", "하락추세 — 현금 (200일선 회복까지 관망)"

    return {
        "coreOn": core_on,
        "price": round(price, 2), "sma200": round(sma, 2),
        "distPct": round((price / sma - 1) * 100, 2),
        "spyWeight": round(spy_w, 2), "spxlWeight": round(spxl_w, 2),
        "effExposure": round(effective_exposure(spxl_w), 2),
        "regime": regime, "note": note,
    }
