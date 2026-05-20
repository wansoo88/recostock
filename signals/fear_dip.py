"""Fear-dip mean-reversion signal (LONG SPY) — EXPERIMENTAL, paper only.

Discovered 2026-05-20 while exhausting short strategies: every directional
short / pair / put expression of the bearish composite loses, but its MIRROR
— buying SPY when the composite hits a causal extreme — is OOS-consistent
(Holdout 2024+ Sharpe ~1.25, WR 60-73%, 10-day hold). Economic basis:
mean-reversion after VIX / credit / term-structure stress spikes.

NOT a live signal. Wired into run_daily as a paper-only track so it can
accumulate an out-of-sample record before any Tier gating. See
[[project-fear-dip-long]].

The composite is a 252-day z-score of 17 features, each oriented so that a
HIGHER value = more bearish (= more contrarian-bullish here). Entry fires
when the composite is at/above its causal expanding-quantile (default q85,
min 252 obs so the threshold never peeks ahead).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FEAR_DIP_Q = 0.85          # causal expanding-quantile entry threshold
FEAR_DIP_HOLD = 10         # trading-day holding period
FEAR_DIP_TICKER = "SPY"
_ZWIN = 252
_MACRO = Path("data/raw/macro")

# orientation: +1 means a HIGH raw value is bearish; -1 means a LOW raw value
# is bearish. Matches the validated screen_downside_ic composite.
_ORIENT = {
    "vix_term_3m": -1, "vix_term_9d": -1, "vix_level": -1, "tlt_ret5": -1,
    "gold_spy": -1, "dxy_chg5": -1, "skew_z": -1,
    "move_z": +1, "spy_dist_sma200": +1, "spy_rsi14": +1, "spy_mom20": +1,
    "hyg_lqd": +1, "y10_chg5": +1, "vvix_z": +1, "yc_10_2": +1,
    "hyg_ret5": +1, "vix_chg5": +1,
}


def _s(name: str) -> pd.Series:
    return pd.read_parquet(_MACRO / f"{name}.parquet").iloc[:, 0].dropna()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _zscore(s: pd.Series, w: int = 60) -> pd.Series:
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def build_bear_score(close_df: pd.DataFrame) -> pd.Series:
    """Compute the 252-day-standardized bearish composite, aligned to SPY dates.

    Reads macro parquet files from data/raw/macro/. Returns NaN for early dates
    where the rolling windows are not yet filled (causal — no look-ahead)."""
    spy = close_df[FEAR_DIP_TICKER].dropna()
    idx = spy.index

    f = pd.DataFrame(index=idx)
    vix, vix9d, vix3m = _s("vix"), _s("vix9d"), _s("vix3m")
    f["vix_level"] = vix.reindex(idx)
    f["vix_chg5"] = vix.reindex(idx).pct_change(5)
    f["vix_term_9d"] = (vix9d / vix).reindex(idx)
    f["vix_term_3m"] = (vix / vix3m).reindex(idx)
    f["vvix_z"] = _zscore(_s("vvix")).reindex(idx)
    f["move_z"] = _zscore(_s("move")).reindex(idx)
    f["skew_z"] = _zscore(_s("skew")).reindex(idx)
    hyg, lqd = _s("hyg"), _s("lqd")
    f["hyg_ret5"] = hyg.pct_change(5).reindex(idx)
    f["hyg_lqd"] = (hyg / lqd).pct_change(5).reindex(idx)
    f["tlt_ret5"] = _s("tlt").pct_change(5).reindex(idx)
    y10, y2 = _s("yield_10y"), _s("yield_2y")
    f["yc_10_2"] = (y10 - y2).reindex(idx)
    f["y10_chg5"] = y10.diff(5).reindex(idx)
    f["dxy_chg5"] = _s("dxy").pct_change(5).reindex(idx)
    f["gold_spy"] = (_s("gold").pct_change(5) - spy.pct_change(5)).reindex(idx)
    f["spy_dist_sma200"] = spy / spy.rolling(200).mean() - 1
    f["spy_rsi14"] = _rsi(spy)
    f["spy_mom20"] = spy.pct_change(20)

    z = pd.DataFrame(index=idx)
    for c in f.columns:
        zz = (f[c] - f[c].rolling(_ZWIN).mean()) / f[c].rolling(_ZWIN).std()
        z[c] = zz * _ORIENT.get(c, 1)
    return z.mean(axis=1)


def evaluate(close_df: pd.DataFrame, q: float = FEAR_DIP_Q) -> dict:
    """Evaluate today's fear-dip entry condition.

    Returns {date, score, threshold, percentile, is_entry, entry_price}.
    is_entry is True when the causal composite >= its expanding q-quantile."""
    score = build_bear_score(close_df)
    thr = score.expanding(min_periods=_ZWIN).quantile(q)
    today = score.index[-1]
    s_now, t_now = score.iloc[-1], thr.iloc[-1]
    # percentile rank of today's score within history up to today
    hist = score.loc[:today].dropna()
    pct = float((hist <= s_now).mean()) if len(hist) else float("nan")
    is_entry = bool(not np.isnan(s_now) and not np.isnan(t_now) and s_now >= t_now)
    return {
        "date": today,
        "score": float(s_now) if not np.isnan(s_now) else None,
        "threshold": float(t_now) if not np.isnan(t_now) else None,
        "percentile": round(pct, 4) if pct == pct else None,
        "is_entry": is_entry,
        "entry_price": float(close_df[FEAR_DIP_TICKER].iloc[-1]),
    }
