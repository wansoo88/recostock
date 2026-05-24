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
SMA_FAST = 50                # golden-cross fast leg
CORE_TICKER = "SPY"
QQQ_TICKER = "QQQ"           # second core sleeve (2026-05-24: diversification win)
LEVER_TICKER = "SPXL"        # Direxion 3x S&P bull
LEVER_MULT = 3.0
SPY_SLEEVE_WEIGHT = float(getattr(config, "TREND_CORE_SPY_WEIGHT", 0.5))
QQQ_SLEEVE_WEIGHT = 1.0 - SPY_SLEEVE_WEIGHT
# fraction of the SPY-sleeve shifted to SPXL on uptrend fear-dip days.
# 0.15 -> ~1.3x effective on the SPY sleeve.
TILT_SPXL_WEIGHT = float(getattr(config, "TREND_CORE_TILT_WEIGHT", 0.15))
# vol-adaptive filter: switch to golden-cross when VIX above this threshold.
VIX_REGIME_THRESHOLD = float(getattr(config, "TREND_CORE_VIX_THRESHOLD", 22.0))


def effective_exposure(spxl_weight: float) -> float:
    return (1 - spxl_weight) * 1.0 + spxl_weight * LEVER_MULT


def _trend_on(price: pd.Series, vix_latest: float | None) -> tuple[bool | None, str]:
    """Vol-adaptive trend filter: 200SMA in low-VIX, golden-cross in high-VIX.

    Below the VIX threshold (calm): price > 200d SMA — responsive, captures bull.
    Above the VIX threshold (stress): price > 50d SMA AND 50d > 200d — needs
    both to confirm. Cuts MDD in volatile regimes (golden-cross ~50% MDD
    reduction vs 200SMA in Full OOS), keeps bull responsiveness elsewhere.
    """
    if len(price) < SMA_WINDOW + 1:
        return None, "data short"
    p = float(price.iloc[-1])
    s200 = float(price.rolling(SMA_WINDOW).mean().iloc[-1])
    high_vol = vix_latest is not None and vix_latest >= VIX_REGIME_THRESHOLD
    if not high_vol:
        return p > s200, "200SMA (저변동)"
    s50 = float(price.rolling(SMA_FAST).mean().iloc[-1])
    on = (p > s50) and (s50 > s200)
    return on, f"골든크로스 (고변동 VIX≥{VIX_REGIME_THRESHOLD:.0f})"


def evaluate(close_df: pd.DataFrame, fear_dip_active: bool,
             vix_latest: float | None = None) -> dict:
    """Today's core exposure recommendation — vol-adaptive dual-asset (SPY+QQQ).

    Each sleeve runs its own vol-adaptive trend filter. SPY sleeve takes the
    SPXL leverage tilt on uptrend+fear-dip; QQQ sleeve is unleveraged.
    Sleeves are independent — if only SPY is in uptrend, only the SPY sleeve
    is invested; the QQQ portion sits in cash earning yield.
    """
    spy = close_df[CORE_TICKER].dropna()
    qqq = close_df[QQQ_TICKER].dropna() if QQQ_TICKER in close_df.columns else pd.Series(dtype=float)
    if len(spy) < SMA_WINDOW + 1:
        return {"coreOn": None, "note": "200일 SMA 계산에 데이터 부족"}

    spy_on, spy_filter = _trend_on(spy, vix_latest)
    qqq_on, _ = _trend_on(qqq, vix_latest) if len(qqq) >= SMA_WINDOW + 1 else (False, "")
    w_tilt = TILT_SPXL_WEIGHT

    # SPY sleeve (size = SPY_SLEEVE_WEIGHT of total capital)
    spy_w = spxl_w = 0.0
    if spy_on:
        if fear_dip_active:
            spy_w = SPY_SLEEVE_WEIGHT * (1 - w_tilt)
            spxl_w = SPY_SLEEVE_WEIGHT * w_tilt
        else:
            spy_w = SPY_SLEEVE_WEIGHT
    elif fear_dip_active:
        # Downtrend + fear: SPY sleeve goes 100% SPY (dip re-entry, no leverage).
        spy_w = SPY_SLEEVE_WEIGHT

    # QQQ sleeve
    qqq_w = QQQ_SLEEVE_WEIGHT if qqq_on else 0.0

    invested = spy_w + spxl_w + qqq_w
    cash_w = max(0.0, 1.0 - invested)
    eff_spy = (spy_w + spxl_w * LEVER_MULT) / max(SPY_SLEEVE_WEIGHT, 1e-9)
    total_eff = spy_w + spxl_w * LEVER_MULT + qqq_w

    if spy_on and qqq_on and fear_dip_active:
        regime, note = "dual_uptrend_panic", (
            f"양쪽 상승 + 공포 — SPY {spy_w*100:.0f}% + SPXL {spxl_w*100:.0f}% + QQQ {qqq_w*100:.0f}% "
            f"(SPY 슬리브 ≈{eff_spy:.2f}x)")
    elif spy_on and qqq_on:
        regime, note = "dual_uptrend", f"양쪽 상승 — SPY {spy_w*100:.0f}% + QQQ {qqq_w*100:.0f}%"
    elif spy_on:
        regime, note = "spy_only", (
            f"SPY만 상승 — SPY {spy_w*100:.0f}%"
            + (f" + SPXL {spxl_w*100:.0f}%" if spxl_w > 0 else "")
            + f" (QQQ 사분면 현금)")
    elif qqq_on:
        regime, note = "qqq_only", f"QQQ만 상승 — QQQ {qqq_w*100:.0f}% (SPY 사분면 현금)"
    elif fear_dip_active:
        regime, note = "downtrend_panic", f"양쪽 하락 + 공포 — SPY {spy_w*100:.0f}% (저점 재진입)"
    else:
        regime, note = "cash", "양쪽 하락 — 현금 100% (BIL/SGOV 등 단기채 파킹 권장 ~연 4-5%)"

    return {
        "coreOn": spy_on, "coreSpyOn": spy_on, "coreQqqOn": qqq_on,
        "trendFilter": spy_filter,
        "price": round(float(spy.iloc[-1]), 2),
        "sma200": round(float(spy.rolling(SMA_WINDOW).mean().iloc[-1]), 2),
        "distPct": round((float(spy.iloc[-1]) / float(spy.rolling(SMA_WINDOW).mean().iloc[-1]) - 1) * 100, 2),
        "spyWeight": round(spy_w, 2), "spxlWeight": round(spxl_w, 2),
        "qqqWeight": round(qqq_w, 2), "cashWeight": round(cash_w, 2),
        "effExposure": round(total_eff, 2),  # total effective exposure (SPY-equiv units)
        "regime": regime, "note": note,
    }
