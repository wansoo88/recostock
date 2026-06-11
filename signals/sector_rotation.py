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

Canonical cost-adjusted figures (0.25% roundtrip, look-ahead-safe day-by-day
replay vs the SHIPPED trend-core engine, data through 2026-05-29) live in the
`BACKTEST` dict below — the single source of truth. Do NOT restate them in this
prose (it previously drifted to a stale 25%-satellite parameterization that
contradicted the dict). The shipped config is a 15% sleeve
(`SUGGESTED_SATELLITE_WEIGHT`): blending the RSI sleeve onto the engine lifts
risk-adjusted return at the same drawdown (engine/sleeve daily correlation
~0.57). A modest improvement — not a moonshot.

REPRODUCTION NOTE (2026-06-01, scripts/sweep_blend_goal.py, data~05-29): the
cost-adjusted sweep reproduced the shipped 15% blend ~3% UNDER the dict
(+121%/1.22 vs the dict's +124%/1.23). The dict is deliberately NOT corrected
down right now because `BACKTEST["blendFull"]["sharpe"]` is the hardwired Tier-2
target that paper/portfolio_tracker._target_sharpe() validates realized NAV
against — editing it mid-paper-window would move the goalpost under the running
validation. Reconcile the dict to the reproduction only AFTER the ~2026-08-29
paper gate (a post-window correction, not a live edit).

Adversarial checks all passed: look-ahead (lag+0 == lag+1), parameter plateau
(RSI14 top-1/2/3 all Sharpe 1.1-1.6), wins the 2022 bear (+8.4% vs eqw -6.2%),
survives 0.60% cost (Sharpe 0.93), jackknife-stable (drop any sector ->
Sharpe 0.83-1.41), 4/5 walk-forward folds positive, ~18 roundtrips/yr.
KNOWN WEAKNESS: trails the engine in low-volatility grind years (2023-2025, a
few points each) and carries a ~-22% standalone MDD. So this is an OPTIONAL
satellite, NOT the core engine — surfaced like the fear-dip track with full
disclosure. RSI-14 also beats the session-1 1m+3m momentum ranking head-to-head,
so it is the better monitor sort key.

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


def evaluate_weekly(close_df: pd.DataFrame) -> dict:
    """evaluate() with the PICK pinned to the last Friday close — the validated cadence.

    The backtest that justifies the sleeve (scripts/sweep_blend_goal.py) rebalances
    on Fridays only and holds the prior pick in between. The daily pipeline used to
    call evaluate() directly, silently rotating the pick ANY day the RSI order
    flipped — 4 rotations in the first 6 paper-tracking days (2026-06-02..06-08),
    each charging turnover cost the backtest never paid. This wrapper restores the
    weekly cadence statelessly: the pick is recomputed from data truncated at the
    most recent Friday close (== what a Friday rebalance would have chosen, held
    since), while `ranked` still reflects today's RSI for display.

    Holiday-Friday edge: if Friday is missing from the index, the truncation lands
    on the previous week's Friday — identical to the backtest loop, where a skipped
    Friday simply keeps the existing pick another week.
    """
    today_eval = evaluate(close_df)
    if not today_eval:
        return today_eval

    idx = close_df.dropna(how="all").index
    fridays = [d for d in idx if pd.Timestamp(d).dayofweek == 4]
    if not fridays:
        # No Friday in history (tiny synthetic frames) — today's pick stands.
        today_eval["pickAsOf"] = today_eval["asOf"]
        return today_eval

    last_friday = fridays[-1]
    if pd.Timestamp(last_friday) == pd.Timestamp(idx[-1]):
        pinned = today_eval          # today IS the rebalance close
    else:
        pinned = evaluate(close_df.loc[:last_friday])
        if not pinned:
            pinned = today_eval

    out = dict(today_eval)
    out["pick"] = pinned.get("pick", [])
    out["cashHalf"] = pinned.get("cashHalf", 0)
    out["pickAsOf"] = str(pd.Timestamp(last_friday).date())
    return out
