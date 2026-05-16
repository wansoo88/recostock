"""Macro / external indicator collection via yfinance.

FRED key is optional — fallback to yfinance proxies works without secrets:
    DXY      → DX-Y.NYB (ICE US Dollar Index)
    10Y      → ^TNX
    2Y proxy → ^IRX (13-week T-bill — short-end rate proxy)
    Oil      → USO ETF (more liquid than CL=F intraday)
    Gold     → GLD
    HY/IG    → HYG / LQD (credit spread proxy via ratio)
    VVIX     → ^VVIX (vol of vol)
    Semis    → SMH (sector context for QQQ/XLK)
    Regional → KRE (XLF context)
    Energy   → XOP (XLE upstream cross-check)
    Bonds    → TLT (long duration)

Each ticker is forward-filled in the consumer (factor builder), not here,
so this stays a thin caching layer.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

import config

log = logging.getLogger(__name__)
MACRO_DIR = Path("data/raw/macro")

MACRO_TICKERS = {
    # name        : yfinance symbol
    "dxy":       "DX-Y.NYB",     # ICE dollar index
    "yield_10y": "^TNX",         # CBOE 10y treasury yield (×10 in raw)
    "yield_2y":  "^IRX",         # CBOE 13w T-bill (short end proxy)
    "oil":       "USO",          # WTI proxy ETF
    "gold":      "GLD",          # gold ETF
    "hyg":       "HYG",          # high-yield corporate bonds
    "lqd":       "LQD",          # investment-grade corporate bonds
    "tlt":       "TLT",          # 20y+ treasuries
    "vvix":      "^VVIX",        # vol of vol
    "smh":       "SMH",          # semiconductors (XLK/QQQ context)
    "kre":       "KRE",          # regional banks (XLF context)
    "xop":       "XOP",          # oil & gas producers (XLE context)
}


def fetch_macro_series(years: int = config.HISTORY_YEARS,
                       end_date: pd.Timestamp | None = None) -> dict[str, pd.Series]:
    """Download all macro proxies. Returns {name: close-price series}.

    Each series strictly excludes today's incomplete bar (look-ahead guard).
    """
    end = end_date.date() if end_date else date.today()
    start = end - timedelta(days=years * 365)
    today_ts = pd.Timestamp(end)

    out: dict[str, pd.Series] = {}
    for name, sym in MACRO_TICKERS.items():
        try:
            df = yf.download(sym, start=str(start), end=str(end),
                             auto_adjust=True, progress=False, threads=False)
            if df.empty:
                log.warning("macro %s (%s): empty", name, sym)
                continue
            close = df["Close"].squeeze()
            close = close[close.index < today_ts]
            close.name = name
            out[name] = close
            log.info("macro %s (%s): %d bars [%s → %s]",
                     name, sym, len(close), close.index.min().date(),
                     close.index.max().date())
        except Exception as exc:
            log.warning("macro %s (%s) failed: %s", name, sym, exc)
    return out


def save_macro_cache(series_dict: dict[str, pd.Series]) -> None:
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    for name, s in series_dict.items():
        s.to_frame().to_parquet(MACRO_DIR / f"{name}.parquet")


def load_macro_cache() -> dict[str, pd.Series]:
    if not MACRO_DIR.exists():
        return {}
    out: dict[str, pd.Series] = {}
    for fp in MACRO_DIR.glob("*.parquet"):
        df = pd.read_parquet(fp)
        s = df.iloc[:, 0]
        s.name = fp.stem
        out[fp.stem] = s
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    series = fetch_macro_series()
    save_macro_cache(series)
    log.info("Saved %d macro series to %s", len(series), MACRO_DIR)
