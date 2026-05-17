"""Phase 4: Paper trading tracker.

Positions carry over from week to week when signal persists — roundtrip cost
is charged ONCE per entry/exit, not every Friday. This matches the Phase 3
backtest cost model (_portfolio_pnl charges only on position changes).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

TRADES_PATH = Path("data/paper/trades.parquet")

_SCHEMA_COLS = [
    "open_date", "ticker", "entry_price", "direction",
    "ema_proba", "winrate", "payoff", "expectancy", "sample_n",
    "close_date", "exit_price", "pnl_pct", "source",
]
_NULLABLE_COLS = {"close_date": "object", "exit_price": "float64",
                  "pnl_pct": "float64", "source": "object"}


# ── Persistence ───────────────────────────────────────────────────────────────

def load_trades() -> pd.DataFrame:
    if TRADES_PATH.exists():
        return pd.read_parquet(TRADES_PATH)
    return pd.DataFrame(columns=_SCHEMA_COLS)


def save_trades(trades: pd.DataFrame) -> None:
    TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(TRADES_PATH, index=False)


def get_open_tickers(trades: pd.DataFrame) -> set[str]:
    """Return set of ticker symbols with currently open (unclosed) positions."""
    if trades.empty or "close_date" not in trades.columns:
        return set()
    return set(trades[trades["close_date"].isna()]["ticker"].tolist())


# ── Schema alignment ──────────────────────────────────────────────────────────

def _align_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in _SCHEMA_COLS:
        if col not in df.columns:
            dtype = _NULLABLE_COLS.get(col, "object")
            df[col] = pd.Series([None] * len(df), dtype=dtype, index=df.index)
    extra = [c for c in df.columns if c not in _SCHEMA_COLS]
    return df[_SCHEMA_COLS + extra]


# ── Position management ───────────────────────────────────────────────────────

def _opened_today(trades: pd.DataFrame, today_str: str) -> set[str]:
    """Return tickers that already have a row with open_date == today.

    Used to make `open_positions` idempotent so repeated same-day runs
    (e.g., GitHub Actions concurrency edge cases) don't create duplicate
    entries. Pre-2026-05-17 saw 4 duplicate DIA rows on the same Friday
    when the cron fired twice and the signal flickered around threshold.
    """
    if trades.empty or "open_date" not in trades.columns:
        return set()
    today_mask = trades["open_date"].astype(str) == today_str
    return set(trades.loc[today_mask, "ticker"].tolist())


def open_positions(
    signals: list,
    today: pd.Timestamp,
    close_prices: pd.Series,
    already_open: set[str] | None = None,
    trades: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Open NEW positions (not carry-overs). already_open tickers are skipped.

    `trades` (full history) is consulted to skip tickers already opened today
    in a prior invocation — same-day idempotency guard.
    """
    already_open = already_open or set()
    today_str = today.date().isoformat()
    opened_today_set = _opened_today(trades, today_str) if trades is not None else set()
    rows = []
    skipped_dup = 0
    for sig in signals:
        if sig.ticker in already_open:
            continue   # carry-over: position persists, no new entry cost
        if sig.ticker in opened_today_set:
            skipped_dup += 1
            continue   # same-day idempotency: never open the same ticker twice on one date
        price = close_prices.get(sig.ticker)
        if price is None or price <= 0:
            continue
        rows.append({
            "open_date": today_str,
            "ticker": sig.ticker,
            "entry_price": float(price),
            "direction": sig.direction,
            "ema_proba": sig.confidence,
            "winrate": sig.winrate,
            "payoff": sig.payoff,
            "expectancy": sig.expectancy,
            "sample_n": sig.sample_n,
            "close_date": None,
            "exit_price": None,
            "pnl_pct": None,
            "source": "live",
        })
    if rows:
        log.info("Paper: opened %d new position(s) on %s", len(rows), today.date())
    if skipped_dup:
        log.info("Paper: skipped %d duplicate same-day open(s)", skipped_dup)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def close_exited_positions(
    trades: pd.DataFrame,
    today: pd.Timestamp,
    close_prices: pd.Series,
    signal_tickers: set[str],
) -> pd.DataFrame:
    """Close only positions whose signal has turned OFF (not carry-overs).

    Positions where signal is still ON are left open (carry-over, no cost).
    """
    if trades.empty:
        return trades

    open_mask = trades["close_date"].isna()
    if not open_mask.any():
        return trades

    trades = trades.copy()
    today_str = today.date().isoformat()
    closed_count = 0

    for idx in trades[open_mask].index:
        ticker = trades.at[idx, "ticker"]
        if ticker in signal_tickers:
            continue  # carry-over: signal still ON, keep position open

        # Same-day exit guard: never close on the same date the position
        # was opened. Pre-2026-05-17, signals oscillating around 0.58 in
        # multi-run scenarios produced open→close in one day, wasting only
        # the 0.25% cost. A real exit requires at least one trading day.
        if str(trades.at[idx, "open_date"]) == today_str:
            continue

        # Signal turned OFF: close the position
        exit_price = close_prices.get(ticker)
        if exit_price is None or exit_price <= 0:
            continue
        entry = trades.at[idx, "entry_price"]
        direction = trades.at[idx, "direction"]
        gross = (float(exit_price) - float(entry)) / float(entry) if direction == "long" else \
                (float(entry) - float(exit_price)) / float(entry)
        net = gross - config.TOTAL_COST_ROUNDTRIP
        trades.at[idx, "close_date"] = today_str
        trades.at[idx, "exit_price"] = round(float(exit_price), 4)
        trades.at[idx, "pnl_pct"] = round(net, 6)
        closed_count += 1

    if closed_count:
        log.info("Paper: closed %d position(s) on %s (signal turned off)", closed_count, today_str)
    return trades


def close_positions(
    trades: pd.DataFrame,
    today: pd.Timestamp,
    close_prices: pd.Series,
) -> pd.DataFrame:
    """Close ALL open positions regardless of signal (use for forced exit or testing)."""
    return close_exited_positions(trades, today, close_prices, signal_tickers=set())


def rebalance_friday(
    trades: pd.DataFrame,
    signals: list,
    today: pd.Timestamp,
    close_prices: pd.Series,
) -> pd.DataFrame:
    """Full Friday rebalance: close exits, open new entries, carry over unchanged.

    Cost is charged only on TRUE position changes (entry or exit).
    Returns updated trades DataFrame (not yet saved).
    """
    signal_tickers = {sig.ticker for sig in signals}
    open_tickers = get_open_tickers(trades)

    # Step 1: Close positions where signal turned OFF
    trades = close_exited_positions(trades, today, close_prices, signal_tickers)

    # Step 2: Open positions for new signals (not already in portfolio)
    # Pass `trades` for same-day idempotency check.
    new_rows = open_positions(signals, today, close_prices,
                              already_open=open_tickers, trades=trades)

    carry_count = len(signal_tickers & open_tickers)
    if carry_count:
        log.info("Paper: %d position(s) carried over on %s (no cost)", carry_count, today.date())

    return append_rows(trades, new_rows)


def append_rows(trades: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    if new_rows.empty:
        return trades
    t = _align_schema(trades)
    n = _align_schema(new_rows)
    for col in ("exit_price", "pnl_pct"):
        t[col] = pd.to_numeric(t[col], errors="coerce")
        n[col] = pd.to_numeric(n[col], errors="coerce")
    return pd.concat([t, n], ignore_index=True)


def append_and_save(trades: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    combined = append_rows(trades, new_rows)
    save_trades(combined)
    return combined


# ── Performance metrics ───────────────────────────────────────────────────────

def compute_metrics(trades: pd.DataFrame, include_backfill: bool = True) -> dict:
    """Compute paper trading performance from closed trades.

    Groups by close_date for equal-weight portfolio weekly Sharpe.
    """
    if not include_backfill and "source" in trades.columns:
        trades = trades[trades["source"] != "backfill"]
    if trades.empty or "close_date" not in trades.columns:
        return {"sharpe": 0.0, "mdd": 0.0, "total_return": 0.0, "n_trades": 0,
                "n_weeks": 0, "winrate": 0.0, "avg_win": 0.0,
                "avg_loss": 0.0, "payoff": 0.0, "weeks_elapsed": 0}

    closed = trades[trades["close_date"].notna()].copy()
    if closed.empty:
        return {"sharpe": 0.0, "mdd": 0.0, "total_return": 0.0, "n_trades": 0,
                "n_weeks": 0, "winrate": 0.0, "avg_win": 0.0,
                "avg_loss": 0.0, "payoff": 0.0, "weeks_elapsed": 0}

    closed["close_date"] = pd.to_datetime(closed["close_date"])
    closed["pnl_pct"] = closed["pnl_pct"].astype(float)

    weekly = closed.groupby("close_date")["pnl_pct"].mean().sort_index()
    equity = (1 + weekly).cumprod()
    total_return = float(equity.iloc[-1] - 1)

    if len(weekly) >= 2 and weekly.std() > 1e-10:
        sharpe = float(weekly.mean() / weekly.std() * np.sqrt(52))
    else:
        sharpe = 0.0

    peak = equity.cummax()
    mdd = float(((equity - peak) / peak).min())

    pnl = closed["pnl_pct"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    winrate = len(wins) / len(pnl)
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.0
    payoff = avg_win / avg_loss if avg_loss > 1e-10 else 0.0

    open_date_min = pd.to_datetime(closed["open_date"]).min()
    weeks_elapsed = int((weekly.index.max() - open_date_min).days / 7)

    return {
        "sharpe": round(sharpe, 4),
        "mdd": round(mdd, 4),
        "total_return": round(total_return, 4),
        "n_trades": len(closed),
        "n_weeks": len(weekly),
        "winrate": round(winrate, 4),
        "avg_win": round(avg_win, 5),
        "avg_loss": round(avg_loss, 5),
        "payoff": round(payoff, 3),
        "weeks_elapsed": weeks_elapsed,
    }


def tier2_gate_check(metrics: dict, backtest_sharpe: float) -> list[tuple[str, bool]]:
    months_elapsed = metrics["weeks_elapsed"] / 4.33
    paper_sharpe = metrics["sharpe"]
    # One-sided gap: penalise only UNDERPERFORMANCE vs backtest.
    # Outperformance means the live carry-over model is more efficient — not a failure.
    underperf = max(0.0, backtest_sharpe - paper_sharpe)
    gap = underperf / backtest_sharpe if backtest_sharpe > 0 else 1.0
    return [
        (f"Paper >= {config.TIER2_PAPER_MONTHS_MIN} months",
         months_elapsed >= config.TIER2_PAPER_MONTHS_MIN),
        (f"Paper Sharpe > {config.TIER2_PAPER_SHARPE_MIN}",
         paper_sharpe > config.TIER2_PAPER_SHARPE_MIN),
        (f"Backtest gap(under) < {config.TIER2_PAPER_BACKTEST_GAP_MAX:.0%}",
         gap < config.TIER2_PAPER_BACKTEST_GAP_MAX),
        (f"n_trades >= {config.TIER1_MIN_TRADING_DAYS}",
         metrics["n_trades"] >= config.TIER1_MIN_TRADING_DAYS),
    ]
