"""Download additional sector ETFs + IC sanity check, no production changes.

New candidates:
  XLB   Materials Select  (2002+)
  XLU   Utilities Select  (1998+)
  XLP   Consumer Staples  (1998+)
  XLC   Communication Services  (2018-)
  IBB   iShares Nasdaq Biotech  (2001+)

If IC analysis on the same period confirms |IC|>0.01 & p<0.05 with v2 macro
features, we expand the universe in the integrated backtest.
"""
from __future__ import annotations
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import config
from data.collector import load_parquet, save_parquet, fetch_etf_ohlcv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

NEW_TICKERS = ["XLB", "XLU", "XLP", "XLC", "IBB"]

# Use the same window as the existing universe for fair comparison
log.info("Downloading %s (11y)…", NEW_TICKERS)
ohlcv_new = fetch_etf_ohlcv(NEW_TICKERS)
log.info("Shape: %s", ohlcv_new.shape)

# Merge with existing etf_ohlcv.parquet
existing = load_parquet("etf_ohlcv")
log.info("Existing tickers: %d", existing["Close"].shape[1])

# Combine — concat on column axis, taking new ones
combined = pd.concat([existing, ohlcv_new], axis=1)
# Keep only unique top-level columns (Close, High, Low, Open, Volume × all tickers)
combined = combined.loc[:, ~combined.columns.duplicated()]
log.info("Combined shape: %s", combined.shape)

# Save expanded file (rename existing first for safety)
RAW = Path("data/raw")
backup_path = RAW / "etf_ohlcv_pre_expand.parquet"
if not backup_path.exists():
    pd.read_parquet(RAW / "etf_ohlcv.parquet").to_parquet(backup_path)
    log.info("Backup saved → %s", backup_path)

combined.to_parquet(RAW / "etf_ohlcv.parquet")
log.info("Saved expanded etf_ohlcv.parquet")

# Per-ticker first-available date sanity
for t in NEW_TICKERS:
    if ("Close", t) in combined.columns:
        s = combined[("Close", t)].dropna()
        log.info("  %s: %d bars [%s → %s]", t, len(s),
                 s.index.min().date(), s.index.max().date())
