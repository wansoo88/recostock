"""SQLite-based intraday trade log.

Tracks actual entries/exits entered by the user via Telegram buttons.
Computes realized winrate and P&L from closed trades.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
DB_PATH = Path("data/intraday_trades.db")

_CREATE = """
CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    date          TEXT    NOT NULL,
    ticker        TEXT    NOT NULL,
    action_ticker TEXT    NOT NULL,
    direction     TEXT    NOT NULL,
    signal_price  REAL    NOT NULL,
    entry_price   REAL,
    tp            REAL,
    sl            REAL,
    exit_price    REAL,
    pnl_pct       REAL,
    exit_reason   TEXT,
    status        TEXT    DEFAULT 'pending',
    created_at    TEXT,
    closed_at     TEXT
)
"""


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE)
    conn.commit()
    return conn


def log_entry(
    ticker: str,
    action_ticker: str,
    direction: str,
    signal_price: float,
    entry_price: float,
    tp: float,
    sl: float,
) -> int:
    """Record a new open trade. Returns the trade ID."""
    now = datetime.now(ET)
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO trades
               (date, ticker, action_ticker, direction, signal_price,
                entry_price, tp, sl, status, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                now.strftime("%Y-%m-%d"), ticker, action_ticker, direction,
                round(signal_price, 4), round(entry_price, 4),
                round(tp, 4), round(sl, 4),
                "open", now.isoformat(),
            ),
        )
        return cur.lastrowid


def log_exit(trade_id: int, exit_price: float, reason: str) -> float | None:
    """Close a trade and compute P&L. Returns pnl_pct or None if not found."""
    now = datetime.now(ET)
    with _conn() as conn:
        row = conn.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()
        if row is None or row["status"] != "open":
            return None
        entry = row["entry_price"]
        direction = row["direction"]
        if direction == "LONG":
            pnl = (exit_price - entry) / entry
        else:
            pnl = (entry - exit_price) / entry
        conn.execute(
            """UPDATE trades
               SET exit_price=?, pnl_pct=?, exit_reason=?, status='closed', closed_at=?
               WHERE id=?""",
            (round(exit_price, 4), round(pnl, 6), reason, now.isoformat(), trade_id),
        )
    return pnl


def get_open_trades() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE status='open' ORDER BY created_at"
        ).fetchall()
    return [dict(r) for r in rows]


def get_stats(date_str: str | None = None) -> dict:
    """Today's (or given date) realized stats from closed trades."""
    if date_str is None:
        date_str = datetime.now(ET).strftime("%Y-%m-%d")
    with _conn() as conn:
        rows = conn.execute(
            "SELECT pnl_pct FROM trades WHERE status='closed' AND date=?",
            (date_str,),
        ).fetchall()
    pnls = [r["pnl_pct"] for r in rows if r["pnl_pct"] is not None]
    if not pnls:
        return {"n": 0, "winrate": 0.0, "avg_pnl": 0.0, "total_pnl": 0.0}
    wins = [p for p in pnls if p > 0]
    return {
        "n": len(pnls),
        "winrate": round(len(wins) / len(pnls), 4),
        "avg_pnl": round(sum(pnls) / len(pnls), 5),
        "total_pnl": round(sum(pnls), 5),
    }


def skip_trade(ticker: str) -> None:
    """Record a skipped signal (status=skipped, no entry logged)."""
    now = datetime.now(ET)
    with _conn() as conn:
        conn.execute(
            """INSERT INTO trades (date, ticker, action_ticker, direction,
               signal_price, status, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (now.strftime("%Y-%m-%d"), ticker, ticker, "SKIP", 0.0, "skipped", now.isoformat()),
        )
