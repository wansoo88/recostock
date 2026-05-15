"""Data collection: yfinance (OHLCV + VIX) and FRED (macro).

Run standalone to refresh historical data:
    python -m data.collector
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

import config

log = logging.getLogger(__name__)
RAW_DIR = Path("data/raw")


def fetch_etf_ohlcv(tickers: list[str], years: int = config.HISTORY_YEARS) -> pd.DataFrame:
    """Download adjusted OHLCV for all tickers.

    Strictly excludes today's incomplete bar so features never see future data.
    """
    end = date.today()
    start = end - timedelta(days=years * 365)
    df = yf.download(
        tickers,
        start=str(start),
        end=str(end),
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    # Guard: drop any bar dated today or later (auto_adjust look-ahead risk)
    today_ts = pd.Timestamp(end)
    df = df[df.index < today_ts]
    log.info("Fetched %d bars for %d tickers", len(df), len(tickers))
    return df


def fetch_vix(years: int = config.HISTORY_YEARS) -> pd.Series:
    end = date.today()
    start = end - timedelta(days=years * 365)
    raw = yf.download(config.VIX_TICKER, start=str(start), end=str(end),
                      auto_adjust=True, progress=False)
    series = raw["Close"].squeeze()
    series.name = "VIX"
    return series[series.index < pd.Timestamp(end)]


def fetch_macro(fred_api_key: str) -> dict[str, pd.Series]:
    """Fetch FRED macro series. Returns empty dict when key not provided."""
    if not fred_api_key:
        log.warning("FRED_API_KEY not set — skipping macro data")
        return {}
    try:
        from fredapi import Fred  # optional dependency
    except ImportError:
        log.error("fredapi not installed — run: pip install fredapi")
        return {}

    fred = Fred(api_key=fred_api_key)
    result: dict[str, pd.Series] = {}
    for name, series_id in config.FRED_SERIES.items():
        try:
            result[name] = fred.get_series(series_id)
        except Exception as exc:
            log.warning("FRED %s failed: %s", series_id, exc)
    return result


def save_parquet(df: pd.DataFrame | pd.Series, name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.parquet"
    df.to_frame().to_parquet(path) if isinstance(df, pd.Series) else df.to_parquet(path)
    log.info("Saved %s -> %s", name, path)
    return path


def load_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR / f"{name}.parquet")


if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)

    from data.universe import get_tickers
    tickers = get_tickers(phase=4, leverage_education_done=False)
    log.info("Downloading %d tickers...", len(tickers))

    ohlcv = fetch_etf_ohlcv(tickers)
    save_parquet(ohlcv, "etf_ohlcv")

    vix = fetch_vix()
    save_parquet(vix, "vix")

    macro = fetch_macro(os.environ.get("FRED_API_KEY", ""))
    for k, v in macro.items():
        save_parquet(v, f"macro_{k}")

    log.info("Phase 0 data collection complete.")
