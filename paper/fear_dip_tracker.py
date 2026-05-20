"""Paper-only tracker for the experimental fear-dip signal.

Separate from paper/tracker.py (which is the conviction Friday-rebalance
carry-over model). Fear-dip is event-driven: enter on a signal day, hold a
fixed FEAR_DIP_HOLD trading days, then close at that day's price. One open
position at a time (no pyramiding) to keep the track record clean.

Records to data/paper/fear_dip_paper.parquet. NOT a live signal — this only
accumulates an out-of-sample record for later Tier evaluation.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import config
from signals.fear_dip import FEAR_DIP_HOLD, FEAR_DIP_TICKER

log = logging.getLogger(__name__)

PATH = Path("data/paper/fear_dip_paper.parquet")
_COLS = ["entry_date", "ticker", "entry_price", "score", "percentile",
         "target_hold", "exit_date", "exit_price", "pnl_pct", "status"]


def load() -> pd.DataFrame:
    if PATH.exists():
        return pd.read_parquet(PATH)
    return pd.DataFrame(columns=_COLS)


def save(df: pd.DataFrame) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PATH, index=False)


def has_open(trades: pd.DataFrame) -> bool:
    return not trades.empty and (trades["status"] == "open").any()


def update(close_df: pd.DataFrame, signal: dict, today: pd.Timestamp) -> pd.DataFrame:
    """Close any open position that has reached its holding horizon, then open
    a new one if today is a fresh entry and nothing is currently open.

    Holding horizon is measured in available trading days in close_df.index.
    """
    trades = load()
    idx = list(close_df.index)
    today = pd.Timestamp(today)

    # ── close matured positions ───────────────────────────────────────────
    if not trades.empty:
        for i in trades.index[trades["status"] == "open"]:
            ed = pd.Timestamp(trades.at[i, "entry_date"])
            if ed not in close_df.index:
                continue
            entry_pos = idx.index(ed)
            exit_pos = entry_pos + int(trades.at[i, "target_hold"])
            if exit_pos < len(idx) and idx[exit_pos] <= today:
                exit_price = float(close_df[FEAR_DIP_TICKER].iloc[exit_pos])
                entry_price = float(trades.at[i, "entry_price"])
                pnl = (exit_price / entry_price - 1) - config.TOTAL_COST_ROUNDTRIP
                trades.at[i, "exit_date"] = idx[exit_pos].date().isoformat()
                trades.at[i, "exit_price"] = round(exit_price, 4)
                trades.at[i, "pnl_pct"] = round(pnl, 6)
                trades.at[i, "status"] = "closed"
                log.info("Fear-dip paper: closed %s -> %s  pnl=%.2f%%",
                         trades.at[i, "entry_date"], idx[exit_pos].date(), pnl * 100)

    # ── open a new position on a fresh entry (one at a time) ──────────────
    if signal.get("is_entry") and not has_open(trades):
        today_str = today.date().isoformat()
        already = (not trades.empty) and (trades["entry_date"] == today_str).any()
        if not already:
            row = {
                "entry_date": today_str, "ticker": FEAR_DIP_TICKER,
                "entry_price": round(signal["entry_price"], 4),
                "score": round(signal["score"], 4) if signal["score"] is not None else None,
                "percentile": signal["percentile"],
                "target_hold": FEAR_DIP_HOLD,
                "exit_date": None, "exit_price": None, "pnl_pct": None, "status": "open",
            }
            new = pd.DataFrame([row])
            trades = new if trades.empty else pd.concat([trades, new], ignore_index=True)
            log.info("Fear-dip paper: OPEN %s @ %.2f (pct=%.0f%%)",
                     today_str, signal["entry_price"], (signal["percentile"] or 0) * 100)

    save(trades)
    return trades


def metrics(trades: pd.DataFrame) -> dict:
    closed = trades[trades["status"] == "closed"] if not trades.empty else trades
    if closed.empty:
        return {"n": 0, "winrate": 0.0, "avg_pnl": 0.0, "total": 0.0, "open": int(
            (trades["status"] == "open").sum()) if not trades.empty else 0}
    pnl = closed["pnl_pct"].astype(float)
    eq = (1 + pnl).prod() - 1
    return {
        "n": int(len(closed)),
        "winrate": round(float((pnl > 0).mean()), 4),
        "avg_pnl": round(float(pnl.mean()), 5),
        "total": round(float(eq), 4),
        "open": int((trades["status"] == "open").sum()),
    }
