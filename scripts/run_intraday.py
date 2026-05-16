#!/usr/bin/env python3
"""Intraday signal loop — run locally during US market hours.

Usage:
    TELEGRAM_BOT_TOKEN=xxx TELEGRAM_CHAT_ID=yyy python scripts/run_intraday.py

What it does every 5 minutes (09:30–16:00 ET, weekdays):
  1. Fetches 5-minute OHLCV for core + sector ETFs
  2. Computes EMA5/EMA20/VWAP/RSI signals
  3. Sends Telegram ONLY when signal direction changes
  4. Each signal alert has [진입] [패스] buttons for trade logging
  5. At 15:45 ET: sends force-close reminder
  6. Generates HTML report with charts on each signal change

Commands:
  /positions — open positions + [TP달성][SL히트][수동청산] buttons
  /stats     — today's realized winrate and P&L
  /help      — command list
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
# Suppress httpx logs that expose the bot token in URLs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from bot.intraday_bot import build_app


def main() -> None:
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not bot_token:
        print("ERROR: TELEGRAM_BOT_TOKEN not set")
        sys.exit(1)
    if not chat_id:
        print("ERROR: TELEGRAM_CHAT_ID not set")
        sys.exit(1)

    app = build_app(bot_token, chat_id)
    print("Intraday bot started. Ctrl+C to stop.")
    print("Commands: /positions  /stats  /help")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
