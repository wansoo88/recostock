"""Phase 4: Paper trading tracker.

Records Friday-to-Friday paper positions and computes live paper performance
against the Phase 3 backtest for the Tier 2 gate check.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

TRADES_PATH = Path("data/paper/trades.parquet")


@dataclass
class PaperTrade:
    open_date: str           # ISO date string
    ticker: str
    entry_price: float
    direction: str           # always "long" in Phase 4
    ema_proba: float
    winrate: float           # rolling-stats snapshot at open time
    payoff: float
    expectancy: float
    sample_n: int
    close_date: str | None = None
    exit_price: float | None = None
    pnl_pct: float | None = None  # net of TOTAL_COST_ROUNDTRIP


# ── Persistence ───────────────────────────────────────────────────────────────

def load_trades() -> pd.DataFrame:
    """Return all paper trades. Empty DataFrame if no file yet."""
    if TRADES_PATH.exists():
        return pd.read_parquet(TRADES_PATH)
    return pd.DataFrame(columns=[
        "open_date", "ticker", "entry_price", "direction",
        "ema_proba", "winrate", "payoff", "expectancy", "sample_n",
        "close_date", "exit_price", "pnl_pct",
    ])


def save_trades(trades: pd.DataFrame) -> None:
    TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(TRADES_PATH, index=False)


# ── Position management ───────────────────────────────────────────────────────

def open_positions(
    signals: list,
    today: pd.Timestamp,
    close_prices: pd.Series,
) -> pd.DataFrame:
    """Create new paper trade rows for each valid signal, return DataFrame of new rows."""
    rows = []
    for sig in signals:
        price = close_prices.get(sig.ticker)
        if price is None or price <= 0:
            continue
        rows.append({
            "open_date": today.date().isoformat(),
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
        })
    if rows:
        log.info("Paper: opened %d position(s) on %s", len(rows), today.date())
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def close_positions(
    trades: pd.DataFrame,
    today: pd.Timestamp,
    close_prices: pd.Series,
) -> pd.DataFrame:
    """Mark all open positions as closed at today's prices. Return updated trades."""
    if trades.empty:
        return trades

    open_mask = trades["close_date"].isna()
    n_open = open_mask.sum()
    if n_open == 0:
        return trades

    trades = trades.copy()
    today_str = today.date().isoformat()
    for idx in trades[open_mask].index:
        ticker = trades.at[idx, "ticker"]
        exit_price = close_prices.get(ticker)
        if exit_price is None or exit_price <= 0:
            continue
        entry = trades.at[idx, "entry_price"]
        direction = trades.at[idx, "direction"]
        if direction == "long":
            gross = (exit_price - entry) / entry
        else:
            gross = (entry - exit_price) / entry
        net = gross - config.TOTAL_COST_ROUNDTRIP
        trades.at[idx, "close_date"] = today_str
        trades.at[idx, "exit_price"] = round(float(exit_price), 4)
        trades.at[idx, "pnl_pct"] = round(net, 6)

    closed_now = trades[open_mask & trades["close_date"].notna()]
    log.info("Paper: closed %d position(s) on %s", len(closed_now), today_str)
    return trades


def append_and_save(trades: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    if new_rows.empty:
        return trades
    combined = pd.concat([trades, new_rows], ignore_index=True)
    save_trades(combined)
    return combined


# ── Performance metrics ───────────────────────────────────────────────────────

def compute_metrics(trades: pd.DataFrame) -> dict:
    """Compute paper trading performance from closed trades.

    Returns dict with: sharpe, mdd, total_return, n_trades, n_weeks,
                       winrate, avg_win, avg_loss, payoff, weeks_elapsed
    """
    closed = trades[trades["close_date"].notna()].copy()
    if closed.empty:
        return {"sharpe": 0.0, "mdd": 0.0, "total_return": 0.0, "n_trades": 0,
                "n_weeks": 0, "winrate": 0.0, "avg_win": 0.0,
                "avg_loss": 0.0, "payoff": 0.0, "weeks_elapsed": 0}

    closed["close_date"] = pd.to_datetime(closed["close_date"])
    closed["pnl_pct"] = closed["pnl_pct"].astype(float)

    # Equal-weight portfolio return per close_date (weekly)
    weekly = closed.groupby("close_date")["pnl_pct"].mean()
    weekly = weekly.sort_index()

    equity = (1 + weekly).cumprod()
    total_return = float(equity.iloc[-1] - 1) if len(equity) > 0 else 0.0

    # Sharpe (annualised, 52 weeks/year)
    if len(weekly) >= 2 and weekly.std() > 1e-10:
        sharpe = float(weekly.mean() / weekly.std() * np.sqrt(52))
    else:
        sharpe = 0.0

    # MDD
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    mdd = float(drawdown.min())

    # Win/loss stats
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
    """Evaluate Tier 2 paper trading gate. Returns list of (label, passed) tuples."""
    months_elapsed = metrics["weeks_elapsed"] / 4.33
    paper_sharpe = metrics["sharpe"]

    gap = abs(paper_sharpe - backtest_sharpe) / backtest_sharpe if backtest_sharpe > 0 else 1.0

    return [
        (f"Paper >= {config.TIER2_PAPER_MONTHS_MIN} months",
         months_elapsed >= config.TIER2_PAPER_MONTHS_MIN),
        (f"Paper Sharpe > {config.TIER2_PAPER_SHARPE_MIN}",
         paper_sharpe > config.TIER2_PAPER_SHARPE_MIN),
        (f"Backtest gap < {config.TIER2_PAPER_BACKTEST_GAP_MAX:.0%}",
         gap < config.TIER2_PAPER_BACKTEST_GAP_MAX),
        (f"n_trades >= {config.TIER1_MIN_TRADING_DAYS}",
         metrics["n_trades"] >= config.TIER1_MIN_TRADING_DAYS),
    ]
