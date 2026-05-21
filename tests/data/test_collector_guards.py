"""Robustness guards for yfinance partial-failure in fetch_etf_ohlcv.

yfinance silently returns partial data when some tickers fail. These tests
lock in the detection: warn on missing tickers, error on missing SPY, abort
when nothing comes back.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import pytest

import data.collector as collector


def _multi(tickers, present, n=300):
    """Build a yfinance-style MultiIndex (Price, Ticker) frame where only
    `present` tickers carry real data; the rest are all-NaN columns."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    fields = ["Open", "High", "Low", "Close", "Volume"]
    cols = pd.MultiIndex.from_product([fields, tickers], names=["Price", "Ticker"])
    df = pd.DataFrame(index=dates, columns=cols, dtype="float64")
    for f in fields:
        for t in tickers:
            df[(f, t)] = (100 + np.arange(n)) if t in present else np.nan
    return df


@pytest.fixture
def tickers():
    return ["SPY", "QQQ", "DIA", "XLK"]


def test_all_present_ok(monkeypatch, tickers, caplog):
    monkeypatch.setattr(collector.yf, "download", lambda *a, **k: _multi(tickers, set(tickers)))
    out = collector.fetch_etf_ohlcv(tickers)
    assert not out.empty


def test_missing_ticker_warns(monkeypatch, tickers, caplog):
    present = {"SPY", "QQQ", "DIA"}  # XLK fails
    monkeypatch.setattr(collector.yf, "download", lambda *a, **k: _multi(tickers, present))
    with caplog.at_level("WARNING"):
        collector.fetch_etf_ohlcv(tickers)
    assert any("XLK" in r.message for r in caplog.records)


def test_missing_spy_logs_critical(monkeypatch, tickers, caplog):
    present = {"QQQ", "DIA", "XLK"}  # SPY fails
    monkeypatch.setattr(collector.yf, "download", lambda *a, **k: _multi(tickers, present))
    with caplog.at_level("ERROR"):
        collector.fetch_etf_ohlcv(tickers)
    assert any("SPY" in r.message for r in caplog.records)


def test_all_missing_aborts(monkeypatch, tickers):
    monkeypatch.setattr(collector.yf, "download", lambda *a, **k: _multi(tickers, set()))
    with pytest.raises(RuntimeError):
        collector.fetch_etf_ohlcv(tickers)


def test_empty_tickers_raises():
    with pytest.raises(ValueError):
        collector.fetch_etf_ohlcv([])
