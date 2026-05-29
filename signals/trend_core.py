"""Trend-following core + fear-dip leverage tilt — PRIMARY return engine.

Backtests (2026-05-22) showed the conviction/fear-dip signal apparatus, while
high win-rate, is invested <10% of the time and badly trails buy-and-hold in
absolute return. A simple 200-day-SMA trend core on SPY captures most of the
drift with half the drawdown (+78.6% / Sharpe 0.99 / MDD -22% full OOS;
+45% / 1.39 / -10.5% in 2024+), and beats every blend with the satellite
signals. So the trend core is the engine.

The validated leverage layers (user opted into leverage):
  1. Always-on 5% SPXL while the trend is on (2026-05-26).
  2. Fear-dip tilt: 15% SPXL on uptrend panic-bounce days (2026-05-22).
  3. Calm-uptrend boost (2026-05-30): when BOTH sleeves are in an uptrend AND
     VIX < CALM_VIX_MAX, the SPY-sleeve SPXL fraction rises 5% → STRONG_SPXL
     (20%). This spends the engine's unused risk budget (its -14% MDD sat far
     under the 25% gate) to stop trailing SPY in calm bull years — WITHOUT
     levering into a rising-vol top, so the drawdown profile is unchanged.
     Cost-adjusted walk-forward: Full OOS +109%→+131% (Sharpe 1.11→1.17,
     MDD -14.0% both), Holdout +56%→+72%. Beats baseline every year, 2022
     bear unchanged (-9.2%), 5/5 WF folds positive. See config.py.

Daily exposure recommendation (per sleeve, each gated by its own trend filter):
    both up & VIX<16                     -> SPY + 20% SPXL boost + QQQ  (~1.4x)
    SPY > SMA & fear-dip active          -> SPY + 15% SPXL tilt
    SPY > SMA                            -> SPY + 5% SPXL  (always-on)
    SPY <= SMA & fear-dip active         -> SPY 100% (dip re-entry, no leverage)
    SPY <= SMA                           -> cash (trend off, BIL/SGOV parking)
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
# baseline always-on SPXL when SPY trend is on (no panic) — adds +2.6%p Full
# OOS / +1.9%p Holdout (2026-05-26 backtest) by keeping a small lever active
# during the trend's 81% time-on. Goes to 0 when trend turns off.
ALWAYS_ON_SPXL = float(getattr(config, "TREND_CORE_ALWAYS_ON_SPXL", 0.05))
# calm-uptrend boost (2026-05-30): SPXL fraction rises to STRONG_SPXL when BOTH
# sleeves are in an uptrend AND VIX < CALM_VIX_MAX. Spends the engine's unused
# risk budget in confirmed calm bulls; never levers into a rising-vol top, so
# MDD is unchanged. Full OOS +109%→+131% / Sharpe 1.11→1.17, MDD -14.0% both.
# See config.py for the full validated backtest. Set STRONG=ALWAYS_ON to disable.
STRONG_SPXL = float(getattr(config, "TREND_CORE_STRONG_SPXL", 0.20))
CALM_VIX_MAX = float(getattr(config, "TREND_CORE_CALM_VIX_MAX", 16.0))
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
             vix_latest: float | None = None,
             fear_dip_open_date: str | None = None) -> dict:
    """Today's core exposure recommendation — vol-adaptive dual-asset (SPY+QQQ).

    Each sleeve runs its own vol-adaptive trend filter. SPY sleeve takes the
    SPXL leverage tilt on uptrend+fear-dip; QQQ sleeve is unleveraged.
    Sleeves are independent — if only SPY is in uptrend, only the SPY sleeve
    is invested; the QQQ portion sits in cash earning yield.

    fear_dip_open_date: ISO date string of the open fear-dip paper position's
    entry, used to compute days remaining in the 10-trading-day tilt window.
    """
    spy = close_df[CORE_TICKER].dropna()
    qqq = close_df[QQQ_TICKER].dropna() if QQQ_TICKER in close_df.columns else pd.Series(dtype=float)
    spxl = close_df[LEVER_TICKER].dropna() if LEVER_TICKER in close_df.columns else pd.Series(dtype=float)
    if len(spy) < SMA_WINDOW + 1:
        return {"coreOn": None, "note": "200일 SMA 계산에 데이터 부족"}

    spy_on, spy_filter = _trend_on(spy, vix_latest)
    qqq_on, _ = _trend_on(qqq, vix_latest) if len(qqq) >= SMA_WINDOW + 1 else (False, "")
    w_tilt = TILT_SPXL_WEIGHT

    # Calm uptrend = BOTH sleeves trending up AND VIX below the calm threshold.
    # This is the only regime that earns the SPXL boost. fear-dip days keep their
    # own tilt and take precedence (they are stress days, rarely calm anyway).
    calm_uptrend = bool(spy_on and qqq_on and vix_latest is not None
                        and vix_latest < CALM_VIX_MAX)

    # SPY sleeve (size = SPY_SLEEVE_WEIGHT of total capital). The SPXL fraction
    # of the sleeve is: 0.15 on fear-dip days, STRONG_SPXL in a calm uptrend,
    # ALWAYS_ON_SPXL otherwise. 0 when the SPY trend is off.
    spy_w = spxl_w = 0.0
    if spy_on:
        if fear_dip_active:
            spxl_frac = w_tilt
        elif calm_uptrend:
            spxl_frac = STRONG_SPXL
        else:
            spxl_frac = ALWAYS_ON_SPXL
        spy_w = SPY_SLEEVE_WEIGHT * (1 - spxl_frac)
        spxl_w = SPY_SLEEVE_WEIGHT * spxl_frac
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
    elif calm_uptrend:
        regime, note = "dual_uptrend_boost", (
            f"양쪽 상승 + 저변동(VIX<{CALM_VIX_MAX:.0f}) — SPY {spy_w*100:.0f}% + SPXL {spxl_w*100:.0f}% + "
            f"QQQ {qqq_w*100:.0f}% · 캄-불 레버 부스트 (SPY 슬리브 ≈{eff_spy:.2f}x)")
    elif spy_on and qqq_on:
        regime, note = "dual_uptrend", (
            f"양쪽 상승 — SPY {spy_w*100:.0f}%"
            + (f" + SPXL {spxl_w*100:.0f}%" if spxl_w > 0 else "")
            + f" + QQQ {qqq_w*100:.0f}%")
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

    # Prices for execution
    spy_px = float(spy.iloc[-1])
    spy_sma = float(spy.rolling(SMA_WINDOW).mean().iloc[-1])
    spy_50 = float(spy.rolling(SMA_FAST).mean().iloc[-1])
    qqq_px = float(qqq.iloc[-1]) if len(qqq) else None
    qqq_sma = float(qqq.rolling(SMA_WINDOW).mean().iloc[-1]) if len(qqq) >= SMA_WINDOW else None
    qqq_50 = float(qqq.rolling(SMA_FAST).mean().iloc[-1]) if len(qqq) >= SMA_FAST else None
    spxl_px = float(spxl.iloc[-1]) if len(spxl) else None
    # Active filter determines the actual stop level the user should watch.
    high_vol = vix_latest is not None and vix_latest >= VIX_REGIME_THRESHOLD
    spy_stop = spy_50 if high_vol else spy_sma
    qqq_stop = qqq_50 if (high_vol and qqq_50 is not None) else qqq_sma

    # Fear-dip tilt window: 10 trading days from open. Compute days remaining.
    tilt_days_left = None
    if fear_dip_open_date and spxl_w > 0:
        try:
            od = pd.Timestamp(fear_dip_open_date)
            elapsed = sum(1 for d in spy.index if od <= d <= spy.index[-1])
            tilt_days_left = max(0, 10 - elapsed + 1)  # +1 includes today
        except Exception:
            tilt_days_left = None

    return {
        "coreOn": spy_on, "coreSpyOn": spy_on, "coreQqqOn": qqq_on,
        "trendFilter": spy_filter,
        "calmBoost": calm_uptrend,
        "price": round(spy_px, 2), "sma200": round(spy_sma, 2),
        "distPct": round((spy_px / spy_sma - 1) * 100, 2),
        "spyWeight": round(spy_w, 2), "spxlWeight": round(spxl_w, 2),
        "qqqWeight": round(qqq_w, 2), "cashWeight": round(cash_w, 2),
        "effExposure": round(total_eff, 2),
        "regime": regime, "note": note,
        # Execution detail (added 2026-05-26 for actionable display):
        "exec": {
            "spy":  {"price": round(spy_px, 2),  "stop": round(spy_stop, 2)} if spy_on else None,
            "spxl": {"price": round(spxl_px, 2)} if (spxl_w > 0 and spxl_px) else None,
            "qqq":  {"price": round(qqq_px, 2),  "stop": round(qqq_stop, 2)} if (qqq_on and qqq_px and qqq_stop) else None,
            "tiltDaysLeft": tilt_days_left,
        },
    }
