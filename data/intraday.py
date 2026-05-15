"""Intraday 5-minute OHLCV data fetcher (yfinance, free tier = 15-min delay)."""
from __future__ import annotations

import logging
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# Core ETFs + their inverse pair for SHORT direction
CORE_TICKERS = ["SPY", "QQQ", "DIA"]
INVERSE_PAIR = {"SPY": "SH", "QQQ": "PSQ", "DIA": "DOG"}
SECTOR_TICKERS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI"]
INTRADAY_TICKERS = CORE_TICKERS + SECTOR_TICKERS + ["VXX"]

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
EOD_WARN = time(15, 45)   # 15 min before close — force-exit warning


def now_et() -> datetime:
    return datetime.now(ET)


def is_market_open() -> bool:
    n = now_et()
    if n.weekday() >= 5:
        return False
    return MARKET_OPEN <= n.time() < MARKET_CLOSE


def is_eod_warn_window() -> bool:
    n = now_et()
    if n.weekday() >= 5:
        return False
    return EOD_WARN <= n.time() < MARKET_CLOSE


def fetch_5min_ohlcv(tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Download today's 5-minute bars for each ticker. Returns {ticker: DataFrame}."""
    tickers = tickers or INTRADAY_TICKERS
    today = now_et().date()
    result: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            raw = yf.download(
                ticker, period="2d", interval="5m",
                progress=False, auto_adjust=True,
            )
            if raw.empty:
                continue
            # Flatten MultiIndex if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.index = pd.DatetimeIndex(raw.index).tz_convert(ET)
            df = raw[raw.index.date == today]
            if len(df) >= 5:
                result[ticker] = df
        except Exception as exc:
            log.debug("fetch_5min_ohlcv %s: %s", ticker, exc)

    return result


def get_vix_level(ohlcv: dict[str, pd.DataFrame]) -> float | None:
    """Extract latest VIX from already-fetched data, or fetch separately."""
    if "^VIX" in ohlcv:
        vix_df = ohlcv["^VIX"]
        if not vix_df.empty:
            return float(vix_df["Close"].iloc[-1])
    # Fallback: quick fetch
    try:
        vix = yf.download("^VIX", period="1d", interval="5m", progress=False, auto_adjust=True)
        if not vix.empty:
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            return float(vix["Close"].iloc[-1])
    except Exception:
        pass
    return None
