"""Tests for paper/tracker.py — position management and metrics."""
from __future__ import annotations

import math

import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from paper.tracker import (
    _align_schema,
    append_and_save,
    close_positions,
    compute_metrics,
    load_trades,
    open_positions,
    tier2_gate_check,
)
from signals.generator import Signal


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_signal(ticker="SPY", winrate=0.55, payoff=1.3, confidence=0.60,
                 entry=500.0, avg_win=0.013, avg_loss=0.010):
    import config
    expectancy = winrate * avg_win - (1 - winrate) * avg_loss - config.TOTAL_COST_ROUNDTRIP
    ci_low, ci_high = max(0.0, winrate - 0.1), min(1.0, winrate + 0.1)
    tp = round(entry * (1 + avg_win), 4)
    sl = round(entry * (1 - avg_loss), 4)
    return Signal(
        ticker=ticker, name=f"{ticker} ETF", direction="long",
        leverage=1, entry=entry, tp=tp, sl=sl,
        winrate=winrate, sample_n=30, ci_low=ci_low, ci_high=ci_high,
        payoff=payoff, expectancy=round(expectancy, 5), confidence=confidence,
    )


def _close_prices(*pairs):
    """Create a pd.Series of close prices from (ticker, price) pairs."""
    return pd.Series(dict(pairs))


# ── open_positions ─────────────────────────────────────────────────────────────

def test_open_positions_returns_rows():
    sig = _make_signal("SPY", entry=500.0)
    today = pd.Timestamp("2026-01-10")  # Friday
    closes = _close_prices(("SPY", 501.0))
    df = open_positions([sig], today, closes)
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "SPY"
    assert df.iloc[0]["entry_price"] == 501.0
    assert df.iloc[0]["close_date"] is None


def test_open_positions_skips_zero_price():
    sig = _make_signal("SPY", entry=0)
    today = pd.Timestamp("2026-01-10")
    closes = _close_prices(("SPY", 0.0))
    df = open_positions([sig], today, closes)
    assert df.empty


def test_open_positions_skips_missing_price():
    sig = _make_signal("QQQ")
    today = pd.Timestamp("2026-01-10")
    closes = _close_prices(("SPY", 450.0))   # QQQ not in closes
    df = open_positions([sig], today, closes)
    assert df.empty


# ── close_positions ────────────────────────────────────────────────────────────

def _make_open_trade(ticker="SPY", entry=500.0, open_date="2026-01-10"):
    return pd.DataFrame([{
        "open_date": open_date, "ticker": ticker, "entry_price": entry,
        "direction": "long", "ema_proba": 0.58, "winrate": 0.55,
        "payoff": 1.3, "expectancy": 0.002, "sample_n": 20,
        "close_date": None, "exit_price": None, "pnl_pct": None,
    }])


def test_close_positions_profit():
    import config
    trade = _make_open_trade("SPY", entry=500.0)
    closes = _close_prices(("SPY", 510.0))
    today = pd.Timestamp("2026-01-17")
    result = close_positions(trade, today, closes)
    row = result.iloc[0]
    assert row["close_date"] == "2026-01-17"
    assert row["exit_price"] == 510.0
    expected_net = (510.0 - 500.0) / 500.0 - config.TOTAL_COST_ROUNDTRIP
    assert abs(row["pnl_pct"] - expected_net) < 1e-6


def test_close_positions_loss():
    import config
    trade = _make_open_trade("SPY", entry=500.0)
    closes = _close_prices(("SPY", 490.0))
    today = pd.Timestamp("2026-01-17")
    result = close_positions(trade, today, closes)
    expected_net = (490.0 - 500.0) / 500.0 - config.TOTAL_COST_ROUNDTRIP
    assert result.iloc[0]["pnl_pct"] < 0
    assert abs(result.iloc[0]["pnl_pct"] - expected_net) < 1e-6


def test_close_positions_empty_trades():
    empty = pd.DataFrame()
    result = close_positions(empty, pd.Timestamp("2026-01-17"), _close_prices())
    assert result.empty


def test_close_positions_no_open():
    """If all trades are already closed, nothing changes."""
    trade = _make_open_trade()
    trade["close_date"] = "2026-01-10"
    trade["exit_price"] = 505.0
    trade["pnl_pct"] = 0.007
    result = close_positions(trade, pd.Timestamp("2026-01-17"), _close_prices(("SPY", 510.0)))
    assert result.iloc[0]["close_date"] == "2026-01-10"  # unchanged


# ── Same-day idempotency guards (added 2026-05-17 after duplicate-trade bug) ──

def test_open_positions_skips_same_day_duplicate():
    """Re-running on the same day must not re-open a ticker already opened today."""
    sig = _make_signal("SPY", entry=500.0)
    today = pd.Timestamp("2026-01-10")
    closes = _close_prices(("SPY", 501.0))

    # First call: opens SPY
    first = open_positions([sig], today, closes, trades=pd.DataFrame())
    assert len(first) == 1

    # Simulate a second invocation later the same day: pass `first` as history
    second = open_positions([sig], today, closes, trades=first)
    assert second.empty, "must not duplicate same-day open"


def test_close_exited_skips_same_day():
    """A position opened today must NOT be closed on the same day even if signal turned off."""
    today = pd.Timestamp("2026-01-10")
    trade = _make_open_trade("SPY", entry=500.0, open_date=today.date().isoformat())
    closes = _close_prices(("SPY", 510.0))
    result = close_positions(trade, today, closes)
    # close_positions calls close_exited_positions with empty signal_tickers.
    # Pre-fix: position closes with pnl = -0.0025 (cost only).
    # Post-fix: same-day guard prevents the close, position stays open.
    assert result.iloc[0]["close_date"] is None, "must not close on the same day as open"


# ── compute_metrics ────────────────────────────────────────────────────────────

def _make_closed_trades(pnl_list, close_dates=None):
    """Build a closed trades DataFrame from a list of PnL values."""
    n = len(pnl_list)
    if close_dates is None:
        # Weekly Fridays
        close_dates = [f"2026-{1 + i // 4:02d}-{10 + (i % 4) * 7:02d}" for i in range(n)]
    rows = []
    for i, pnl in enumerate(pnl_list):
        rows.append({
            "open_date": "2026-01-03", "ticker": "SPY", "entry_price": 500.0,
            "direction": "long", "ema_proba": 0.58, "winrate": 0.55,
            "payoff": 1.3, "expectancy": 0.002, "sample_n": 20,
            "close_date": close_dates[i], "exit_price": 505.0, "pnl_pct": pnl,
        })
    return pd.DataFrame(rows)


def test_compute_metrics_empty():
    m = compute_metrics(pd.DataFrame())
    assert m["n_trades"] == 0
    assert m["sharpe"] == 0.0


def test_compute_metrics_positive_returns():
    pnl = [0.01, 0.008, -0.003, 0.012, 0.005, -0.002, 0.009, 0.007]
    close_dates = [f"2026-01-{10 + i * 7}" if (10 + i * 7) <= 31 else f"2026-02-{(10 + i * 7) % 31}" for i in range(len(pnl))]
    # Use unique Fridays
    close_dates = ["2026-01-10", "2026-01-17", "2026-01-24", "2026-01-31",
                   "2026-02-07", "2026-02-14", "2026-02-21", "2026-02-28"]
    m = compute_metrics(_make_closed_trades(pnl, close_dates))
    assert m["n_trades"] == 8
    assert m["winrate"] > 0.5
    assert m["sharpe"] != 0.0


def test_compute_metrics_winrate_payoff():
    pnl = [0.02, 0.02, -0.01, -0.01, 0.02]
    close_dates = ["2026-01-10", "2026-01-17", "2026-01-24", "2026-01-31", "2026-02-07"]
    m = compute_metrics(_make_closed_trades(pnl, close_dates))
    assert abs(m["winrate"] - 0.6) < 0.01
    assert m["avg_win"] > m["avg_loss"]
    assert m["payoff"] > 1.0


# ── tier2_gate_check ───────────────────────────────────────────────────────────

def test_tier2_all_pass():
    metrics = {
        "sharpe": 0.7, "mdd": -0.10, "total_return": 0.15,
        "n_trades": 150, "n_weeks": 52, "winrate": 0.58,
        "avg_win": 0.012, "avg_loss": 0.009, "payoff": 1.33,
        "weeks_elapsed": 13,  # ~3 months
    }
    checks = tier2_gate_check(metrics, backtest_sharpe=1.0)
    assert all(passed for _, passed in checks), [c for c in checks if not c[1]]


def test_tier2_fail_short_period():
    metrics = {
        "sharpe": 0.7, "mdd": -0.10, "total_return": 0.10,
        "n_trades": 150, "n_weeks": 10, "winrate": 0.58,
        "avg_win": 0.012, "avg_loss": 0.009, "payoff": 1.33,
        "weeks_elapsed": 4,   # only 1 month
    }
    checks = tier2_gate_check(metrics, backtest_sharpe=1.0)
    failed = [label for label, passed in checks if not passed]
    assert any("month" in lbl.lower() for lbl in failed)


def test_tier2_fail_low_sharpe():
    metrics = {
        "sharpe": 0.3, "mdd": -0.10, "total_return": 0.05,
        "n_trades": 130, "n_weeks": 52, "winrate": 0.52,
        "avg_win": 0.008, "avg_loss": 0.009, "payoff": 0.9,
        "weeks_elapsed": 14,
    }
    checks = tier2_gate_check(metrics, backtest_sharpe=1.0)
    failed = [label for label, passed in checks if not passed]
    assert any("sharpe" in lbl.lower() for lbl in failed)


# ── schema alignment ───────────────────────────────────────────────────────────

def test_align_schema_adds_missing_cols():
    df = pd.DataFrame([{"ticker": "SPY", "entry_price": 500.0}])
    aligned = _align_schema(df)
    assert "close_date" in aligned.columns
    assert "pnl_pct" in aligned.columns
