"""Macro / cross-asset features. Strictly causal — uses macro_collector cache.

Two layers:
1. GLOBAL features (every ticker sees them):
   - term spread, credit spread, DXY regime, gold/equity ratio,
     real-yield proxy, VVIX/VIX ratio, oil regime, TLT trend.
2. TICKER-SPECIFIC features (gated by `ticker` arg):
   - XLE: oil_z, xop_spread (sector vs upstream)
   - XLF: yield_curve_steepness, kre_spread (sector vs regional banks)
   - XLK / QQQ: smh_spread (sector vs semis)
   - SPY / DIA / XLY / XLI / XLV: term spread, credit spread, DXY emphasis

All transforms are pct_change / rolling z-score / log-ratio — never absolute
levels, so the model sees stationary signals across regimes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.macro_collector import load_macro_cache


def _z(s: pd.Series, window: int = 63) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return (s - mu) / sd.replace(0, np.nan)


def _log_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    aligned = pd.concat([num, den], axis=1, join="inner").dropna()
    return np.log(aligned.iloc[:, 0] / aligned.iloc[:, 1].replace(0, np.nan))


def build_global_macro(date_index: pd.DatetimeIndex,
                       macro: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Return DataFrame indexed on date_index. Same row for ALL tickers on that date.

    These features are NOT ticker-specific — they describe the macro regime.
    """
    if macro is None:
        macro = load_macro_cache()
    if not macro:
        return pd.DataFrame(index=date_index)

    df = pd.DataFrame(index=date_index)

    def get(name: str) -> pd.Series | None:
        s = macro.get(name)
        if s is None:
            return None
        return s.reindex(date_index, method="ffill")

    dxy = get("dxy")
    y10 = get("yield_10y")
    y2 = get("yield_2y")
    oil = get("oil")
    gold = get("gold")
    hyg = get("hyg")
    lqd = get("lqd")
    tlt = get("tlt")
    vvix = get("vvix")

    # Term spread (10y - 2y) — recession & risk-on indicator
    if y10 is not None and y2 is not None:
        df["term_spread"] = y10 - y2
        df["term_spread_chg_5d"] = df["term_spread"].diff(5)

    # 10y yield change — bond market repricing
    if y10 is not None:
        df["y10_chg_5d"] = y10.pct_change(5)
        df["y10_z"] = _z(y10, 63)

    # Credit spread proxy: HYG / LQD log-ratio falling = stress
    if hyg is not None and lqd is not None:
        df["hy_ig_logratio"] = _log_ratio(hyg, lqd).reindex(date_index).ffill()
        df["hy_ig_z"] = _z(df["hy_ig_logratio"], 63)

    # DXY regime
    if dxy is not None:
        df["dxy_z"] = _z(dxy, 63)
        df["dxy_chg_21d"] = dxy.pct_change(21)

    # Gold (risk-off) vs equity proxy — use gold pct only (equity baked in via factors)
    if gold is not None:
        df["gold_chg_21d"] = gold.pct_change(21)
        df["gold_z"] = _z(gold, 63)

    # Oil regime
    if oil is not None:
        df["oil_chg_5d"] = oil.pct_change(5)
        df["oil_chg_21d"] = oil.pct_change(21)
        df["oil_z"] = _z(oil, 63)

    # Long bonds trend (TLT) — flight to quality signal
    if tlt is not None:
        df["tlt_chg_21d"] = tlt.pct_change(21)
        df["tlt_z"] = _z(tlt, 63)

    # VVIX / vol-of-vol — extreme regime
    if vvix is not None:
        df["vvix_z"] = _z(vvix, 63)

    return df


# Ticker → which macro features apply most strongly. Used downstream to
# enable per-ticker conditional features (interactions) without exploding
# the global feature count.
TICKER_MACRO_AFFINITY: dict[str, list[str]] = {
    "SPY":  ["term_spread", "term_spread_chg_5d", "hy_ig_z", "dxy_z", "vvix_z", "y10_chg_5d"],
    "DIA":  ["term_spread", "hy_ig_z", "dxy_z", "vvix_z"],
    "QQQ":  ["dxy_z", "y10_z", "y10_chg_5d", "vvix_z"],
    "XLK":  ["dxy_z", "y10_z", "y10_chg_5d"],
    "XLF":  ["term_spread", "term_spread_chg_5d", "hy_ig_z", "y10_z"],
    "XLE":  ["oil_z", "oil_chg_5d", "oil_chg_21d", "dxy_z"],
    "XLI":  ["term_spread", "oil_chg_21d", "dxy_z"],
    "XLY":  ["term_spread", "hy_ig_z", "dxy_z", "tlt_chg_21d"],
    "XLV":  ["term_spread", "y10_z", "vvix_z"],
}


def build_ticker_specific(date_index: pd.DatetimeIndex,
                          close_df: pd.DataFrame,
                          ticker: str,
                          macro: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Spread features that pit a ticker's primary ETF against its proxy."""
    if macro is None:
        macro = load_macro_cache()

    df = pd.DataFrame(index=date_index)

    def get(name: str) -> pd.Series | None:
        s = macro.get(name)
        if s is None:
            return None
        return s.reindex(date_index, method="ffill")

    # If we don't have the underlying ETF, return empty
    if ticker not in close_df.columns:
        return df

    own = close_df[ticker].reindex(date_index, method="ffill")

    if ticker == "XLE":
        xop = get("xop")
        oil = get("oil")
        if xop is not None:
            # XLE/XOP ratio — diverges when integrated majors decouple from upstream
            df["xle_xop_spread"] = own.pct_change(21) - xop.pct_change(21)
        if oil is not None:
            # 21d β-like elasticity: XLE return / oil return  (clipped for outliers)
            r_own = own.pct_change(21)
            r_oil = oil.pct_change(21)
            ratio = (r_own / r_oil.replace(0, np.nan)).clip(-5, 5)
            df["xle_oil_elasticity"] = ratio.rolling(21).mean()

    elif ticker == "XLF":
        kre = get("kre")
        if kre is not None:
            df["xlf_kre_spread"] = own.pct_change(21) - kre.pct_change(21)

    elif ticker in {"XLK", "QQQ"}:
        smh = get("smh")
        if smh is not None:
            df[f"{ticker.lower()}_smh_spread"] = own.pct_change(21) - smh.pct_change(21)

    return df
