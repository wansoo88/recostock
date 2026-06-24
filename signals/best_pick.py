"""Single best-pick "long shot" selector — concentrated weekly momentum pick.

WHY THIS EXISTS
    A /goal asked: "expand the universe, pick the single best long-shot per day,
    run it toward a weekly +3% return target." This module is the disciplined
    realisation of that ask. The validated edge it stands on is the SAME one the
    RSI sector sleeve uses (cross-sectional short-term momentum, IC +0.035) — but
    instead of holding the top-2 at 15% of capital, it names the SINGLE top pick
    for a concentrated bet.

    Researched 2026-06-24 (scripts/research_best_pick.py, cost-adjusted, look-
    ahead-safe day-by-day replay over an expanded universe of 6 sectors + 4× 3x
    leveraged longs + 50 large-cap single stocks). Honest verdict on the 3%/week
    target is in WEEKLY_TARGET_REALITY below: it is a TAKE-PROFIT target, NOT an
    achievable mean. Only the two GATED modes here cleared the Tier-1 gate; every
    ungated / pure-leverage / single-stock variant failed it on drawdown
    (-55% to -90% MDD).

THE TWO VALIDATED MODES (both Tier-1 PASS, both pick ONE name, weekly cadence)
    "disciplined" — pool = 6 GICS sectors, rank = RSI-14 × 60d-momentum.
        Full OOS 2021+  +182% / Sharpe 1.15 / MDD -19%   (6/6 years positive,
        +4.8% in the 2022 bear). Holdout 2024+ +72% / 1.41 / -17%. The robust
        choice — beats the shipped 85/15 blend on return at a comparable Sharpe.
    "longshot" — pool = 6 sectors + TQQQ/SOXL/SPXL/QLD, rank = RSI-14.
        Full OOS 2021+  +217% / Sharpe 0.86 / MDD -25% (AT the gate cap; 4/6
        years positive, -11.6% in 2022). Holdout 2024+ +165% / 1.41 / -22%.
        The literal "long shot" — higher ceiling, rides a 3x ETF ~21% of weeks,
        but real drawdown and two down years. MDD sits exactly on the -25% Tier-1
        limit, so it is FRAGILE — a worse bear than 2022 would breach it.

GATING (what keeps both modes inside the gate)
    Rebalance weekly (Friday close). Pick only when SPY > its 200d SMA (market
    uptrend) AND the candidate is above its OWN 200d SMA; otherwise hold cash
    (BIL/SGOV). The SPY gate parks the strategy ~51% of weeks — that idle-cash
    discipline, not the signal, is what holds the drawdown under the cap.

STATUS — RESEARCH / OPTIONAL SATELLITE, NOT THE LIVE ENGINE
    The shipped live allocation stays the trend-core + 15% RSI-sleeve blend. Per
    CLAUDE.md the live blend weights are frozen during the paper-validation window
    (~2026-08-29), and a concentrated single-name bet is a different risk profile
    than the blend the tracker is validating. So this is surfaced like the fear-
    dip / conviction satellites: full disclosure, user opts in manually. It is NOT
    wired into the telegram instruction (that stays blend-only) and does NOT move
    the tracker's hardwired target.

Pure function: prices in, single pick out. No I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from signals.sector_rotation import compute_rsi

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI"]
LEVER_LONGS = ["TQQQ", "SOXL", "SPXL", "QLD"]
SMA_WINDOW = 200
MOM_WINDOW = 60                       # 60-trading-day (~12-week) momentum
WEEKLY_TARGET = 0.03                  # the goal's +3%/week take-profit target
CORE_TREND_TICKER = "SPY"

_NAMES = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
    "XLV": "Health Care", "XLY": "Consumer Disc", "XLI": "Industrials",
    "TQQQ": "UltraPro QQQ 3x", "SOXL": "Semiconductor Bull 3x",
    "SPXL": "S&P500 Bull 3x", "QLD": "Ultra QQQ 2x",
}
_LEVERAGE = {"TQQQ": 3, "SOXL": 3, "SPXL": 3, "QLD": 2}

# Honest framing of the +3%/week target (scripts/research_best_pick.py, 2026-06-24,
# Full OOS 2021+). +3% is a TAKE-PROFIT TARGET, not an expected mean. No config
# of 48 reached a +3%/week MEAN; the max sustainable mean was +1.4%/week and only
# via an ungated 3x basket with -81% MDD (gate FAIL).
WEEKLY_TARGET_REALITY = {
    "target": WEEKLY_TARGET,
    "note": ("주간 +3%는 '목표(TP)'지 기대 평균이 아니다. 검증된 두 모드의 실제 주간 "
             "평균은 +0.4~0.5%이고, +3% 이상으로 마감하는 주는 약 14%(7주 중 1주). "
             "나머지 주는 0 근방 또는 손실(최악 주 disciplined -7.5% / longshot -18.2%)."),
}

# Verified cost-adjusted backtest (scripts/research_best_pick.py, data~2026-05-29).
# Single source of truth for honest display — do NOT restate elsewhere.
BACKTEST = {
    "disciplined": {
        "pool": "6 섹터", "rank": "RSI-14 × 60일 모멘텀",
        "fullRet": 182, "fullSharpe": 1.15, "fullMdd": -19,
        "holdRet": 72, "holdSharpe": 1.41, "holdMdd": -17,
        "wkMeanPct": 0.40, "wkHit3Pct": 14, "wkWorstPct": -7.5,
        "yearsPositive": "6/6", "bear2022Pct": 4.8, "gate": "PASS",
    },
    "longshot": {
        "pool": "6 섹터 + 3x 레버리지(TQQQ/SOXL/SPXL/QLD)", "rank": "RSI-14",
        "fullRet": 217, "fullSharpe": 0.86, "fullMdd": -25,
        "holdRet": 165, "holdSharpe": 1.41, "holdMdd": -22,
        "wkMeanPct": 0.47, "wkHit3Pct": 14, "wkWorstPct": -18.2,
        "yearsPositive": "4/6", "bear2022Pct": -11.6, "gate": "PASS (MDD가 -25% 한도에 밀착 — 취약)",
    },
}

_POOLS = {"disciplined": SECTORS, "longshot": SECTORS + LEVER_LONGS}


def _score(close: pd.Series, mode: str) -> float:
    """Strictly-causal ranking score for one candidate (uses closes up to today)."""
    s = close.dropna()
    if len(s) < SMA_WINDOW + 1:
        return np.nan
    rsi = compute_rsi(s).iloc[-1]
    if pd.isna(rsi):
        return np.nan
    if mode == "longshot":
        return float(rsi)
    # disciplined: RSI-14 × 60d momentum
    if len(s) <= MOM_WINDOW:
        return np.nan
    mom = s.iloc[-1] / s.iloc[-(MOM_WINDOW + 1)] - 1
    return float(rsi * mom)


def _above_sma(close: pd.Series) -> tuple[bool, float, float]:
    s = close.dropna()
    if len(s) < SMA_WINDOW + 1:
        return False, np.nan, np.nan
    px = float(s.iloc[-1])
    sma = float(s.rolling(SMA_WINDOW).mean().iloc[-1])
    return px > sma, px, sma


def select(close_df: pd.DataFrame, mode: str = "disciplined") -> dict:
    """Choose today's single best pick from the expanded universe.

    Gated: requires SPY uptrend AND the pick above its own 200d SMA, else cash.
    Returns a dict for the report/CLI:
      mode, pick (ticker or None), name, leverage, score, entry, tp (+3%),
      stop (200d SMA), distStopPct, ranked (all candidates, score desc),
      spyTrendOn, asOf, target (+3%), backtest, reality.
    Empty {} when there is not enough history.
    """
    if mode not in _POOLS:
        raise ValueError(f"mode must be one of {list(_POOLS)}, got {mode!r}")
    pool = [t for t in _POOLS[mode] if t in close_df.columns]
    if CORE_TREND_TICKER not in close_df.columns or len(pool) < 2:
        return {}

    spy_on, _, _ = _above_sma(close_df[CORE_TREND_TICKER])

    ranked = []
    for t in pool:
        sc = _score(close_df[t], mode)
        if pd.isna(sc):
            continue
        above, px, sma = _above_sma(close_df[t])
        ranked.append({
            "ticker": t, "name": _NAMES.get(t, t), "leverage": _LEVERAGE.get(t, 1),
            "score": round(float(sc), 4), "price": round(px, 2),
            "sma200": round(sma, 2), "above200": bool(above),
            "distStopPct": round((px / sma - 1) * 100, 1) if sma > 0 else None,
        })
    if len(ranked) < 2:
        return {}
    ranked.sort(key=lambda r: r["score"], reverse=True)

    # Pick = top-scoring candidate that is above its own 200SMA, only in an
    # SPY uptrend. Otherwise cash (the drawdown-controlling discipline).
    pick = None
    if spy_on:
        for r in ranked:
            if r["above200"]:
                pick = r
                break

    out = {
        "mode": mode,
        "spyTrendOn": bool(spy_on),
        "ranked": ranked,
        "asOf": str(close_df.index[-1])[:10],
        "target": WEEKLY_TARGET,
        "backtest": BACKTEST[mode],
        "reality": WEEKLY_TARGET_REALITY,
    }
    if pick is None:
        out.update({"pick": None, "note": (
            "현금 — SPY 추세 OFF" if not spy_on
            else "현금 — 200일선 위 후보 없음")})
        return out

    entry = pick["price"]
    out.update({
        "pick": pick["ticker"],
        "name": pick["name"],
        "leverage": pick["leverage"],
        "score": pick["score"],
        "entry": entry,
        "tp": round(entry * (1 + WEEKLY_TARGET), 2),
        "stop": pick["sma200"],
        "distStopPct": pick["distStopPct"],
        "note": (f"{pick['ticker']} 단일 베스트픽 — 진입 ${entry:.2f} · "
                 f"목표(TP) +3% ${entry*(1+WEEKLY_TARGET):.2f} · "
                 f"손절 200일선 ${pick['sma200']:.2f}"),
    })
    return out


def select_weekly(close_df: pd.DataFrame, mode: str = "disciplined") -> dict:
    """select() with the PICK pinned to the last Friday close — the validated
    cadence (scripts/research_best_pick.py rebalances Fridays only). `ranked`
    still reflects today's scores for display; only the actionable `pick` and its
    entry/tp/stop are held from Friday. Mirrors sector_rotation.evaluate_weekly.
    """
    today = select(close_df, mode)
    if not today or "pick" not in today:
        return today

    idx = close_df.dropna(how="all").index
    fridays = [d for d in idx if pd.Timestamp(d).dayofweek == 4]
    if not fridays:
        today["pickAsOf"] = today["asOf"]
        return today

    last_friday = fridays[-1]
    if pd.Timestamp(last_friday) == pd.Timestamp(idx[-1]):
        pinned = today
    else:
        pinned = select(close_df.loc[:last_friday], mode)
        if not pinned:
            pinned = today

    out = dict(today)
    for k in ("pick", "name", "leverage", "score", "entry", "tp", "stop",
              "distStopPct", "note"):
        if k in pinned:
            out[k] = pinned[k]
        elif k in out:
            out.pop(k)
    out["pick"] = pinned.get("pick")
    out["pickAsOf"] = str(pd.Timestamp(last_friday).date())
    return out
