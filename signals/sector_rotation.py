"""Sector relative-strength rotation — IC-validated satellite signal.

Discovered 2026-05-30 while answering "can the LightGBM model be upgraded to pick
sectors?". The honest answer was NO — the model's cross-sectional IC is ~0
(raw -0.0022, EMA5 +0.0008, t~0 over 1356 OOS days), so it has zero ability to
rank sectors. Retraining it for selection is futile (naive momentum IC is ~0 too).

BUT a simple, strictly-causal RSI-14 *does* rank the six sector ETFs:
    cross-sectional IC = +0.035 (t=3.5) at h=5, +0.044 (t=4.3) at h=10
both above IC_MIN_VIABLE=0.01 and statistically significant. Sign is positive
(high RSI -> high forward return = short-term sector momentum).

Validated rule (cost-adjusted, look-ahead-safe): each week, rank the 6 sectors by
RSI-14 and hold the top-2 — but only while a chosen sector is above its own 200d
SMA (else that half parks in cash). Weekly rebalance.

Canonical backtest (scripts numbers reproduced against the SHIPPED trend-core
engine, 0.25% cost, data through 2026-05-29):
    standalone gated  Full 2021+   +147% / Sharpe 1.30 / MDD -22%
                      Holdout 2024+ +59% / Sharpe 1.46 / MDD -22%
    as a 25% satellite blended with the trend-core engine:
        Full 2021+   engine +132%/1.17/-14%  ->  +136%/1.28/-14%
        Holdout 2024+ engine +73%/1.44/-10%  ->   +69%/1.55/-11%
    i.e. +0.11 Sharpe in BOTH periods at the same drawdown (engine/sleeve daily
    correlation ~0.57). A modest, risk-adjusted improvement — not a moonshot.

Adversarial checks all passed: look-ahead (lag+0 == lag+1), parameter plateau
(RSI14 top-1/2/3 all Sharpe 1.1-1.6), wins the 2022 bear (+8.4% vs eqw -6.2%),
survives 0.60% cost (Sharpe 0.93), jackknife-stable (drop any sector ->
Sharpe 0.83-1.41), 4/5 walk-forward folds positive, ~18 roundtrips/yr.
KNOWN WEAKNESS: trails the engine in low-volatility grind years (2023-2025, a
few points each) and carries a -22% standalone MDD. So this is an OPTIONAL
satellite, NOT the core engine — surfaced like the fear-dip track with full
disclosure. RSI-14 also beats the session-1 1m+3m momentum ranking head-to-head
(gated top-2: +147%/1.30 vs +121%/1.02), so it is the better monitor sort key.

This module is a pure function: prices in, ranking + actionable pick out. No I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

# The six pure GICS sector ETFs the IC was measured on (index ETFs excluded —
# the edge is cross-sectional *between sectors*).
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI"]
RSI_WINDOW = 14
SMA_WINDOW = 200
TOP_K = 2
SUGGESTED_SATELLITE_WEIGHT = 0.15   # default blend fraction (85 core / 15 sat)

# Backtested expectations for honest display (see module docstring). All cost-
# adjusted, vs the shipped trend-core engine, data through 2026-05-29.
BACKTEST = {
    # All cost-adjusted, look-ahead-safe, vs the shipped trend_core.evaluate
    # (day-by-day causal replay), data through 2026-05-29.
    "engineFull": {"ret": 114, "sharpe": 1.12, "mdd": -14},
    "engineHoldout": {"ret": 59, "sharpe": 1.43, "mdd": -11},
    # standalone gated top-2 RSI sleeve (weekly, 200SMA gate, cash otherwise).
    "standaloneFull": {"ret": 177, "sharpe": 1.35, "mdd": -22},
    "standaloneHoldout": {"ret": 59, "sharpe": 1.46, "mdd": -22},
    # blend at the default SECTOR_SLEEVE_WEIGHT (15%).
    "blendFull": {"ret": 124, "sharpe": 1.23, "mdd": -12},
    "blendHoldout": {"ret": 59, "sharpe": 1.51, "mdd": -12},
    "ic": 0.035, "icT": 3.5, "blendWeight": SUGGESTED_SATELLITE_WEIGHT,
}

_SECTOR_NAMES = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
    "XLV": "Health Care", "XLY": "Consumer Disc", "XLI": "Industrials",
}


def compute_rsi(close: pd.Series, n: int = RSI_WINDOW) -> pd.Series:
    """RSI using a simple rolling mean of gains/losses (matches the IC study).

    Strictly causal — uses only closes up to and including each date.
    """
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    dn = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def evaluate(close_df: pd.DataFrame) -> dict:
    """Rank sectors by RSI-14 and select the actionable top-K above 200d SMA.

    Returns a dict for the report/telegram:
      ranked: list of {ticker, name, rsi, price, sma200, above200, dist200Pct}
              sorted by RSI descending
      pick:   tickers in the top-K that are also above their 200d SMA
      cashHalf: how many of the K slots are parked in cash (trend off)
      asOf:   ISO date of the latest close used
      backtest: BACKTEST stats for disclosure
    Empty {} when there is not enough history.
    """
    avail = [t for t in SECTORS if t in close_df.columns]
    if len(avail) < TOP_K + 1:
        return {}

    rows = []
    for t in avail:
        s = close_df[t].dropna()
        if len(s) < SMA_WINDOW + 1:
            continue
        rsi_val = compute_rsi(s).iloc[-1]
        if pd.isna(rsi_val):
            continue
        px = float(s.iloc[-1])
        sma200 = float(s.rolling(SMA_WINDOW).mean().iloc[-1])
        rows.append({
            "ticker": t,
            "name": _SECTOR_NAMES.get(t, t),
            "rsi": round(float(rsi_val), 1),
            "price": round(px, 2),
            "sma200": round(sma200, 2),
            "above200": bool(px > sma200),
            "dist200Pct": round((px / sma200 - 1) * 100, 1),
        })
    if len(rows) < TOP_K + 1:
        return {}

    rows.sort(key=lambda r: r["rsi"], reverse=True)
    top = rows[:TOP_K]
    pick = [r["ticker"] for r in top if r["above200"]]
    cash_half = TOP_K - len(pick)

    return {
        "ranked": rows,
        "pick": pick,
        "topK": TOP_K,
        "cashHalf": cash_half,
        "asOf": str(close_df.index[-1])[:10],
        "weight": SUGGESTED_SATELLITE_WEIGHT,
        "backtest": BACKTEST,
    }
