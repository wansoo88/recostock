#!/usr/bin/env python3
"""Daily signal pipeline entry point (GitHub Actions + local).

Usage:
    SYSTEM_PHASE=4 FRED_API_KEY=xxx python scripts/run_daily.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import date
from pathlib import Path

# Ensure repo root is on PYTHONPATH when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.collector import fetch_etf_ohlcv, fetch_vix, fetch_macro, save_parquet
from data.universe import get_active_universe
from report.builder import build_report
from bot.notifier import send_daily_signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except ValueError:
        return default


def _env_bool(key: str) -> bool:
    return os.environ.get(key, "false").strip().lower() in {"true", "1", "yes"}


async def main() -> None:
    today = date.today()
    phase = _env_int("SYSTEM_PHASE", 0)
    leverage_ok = _env_bool("LEVERAGE_EDUCATION_DONE")
    fred_key = os.environ.get("FRED_API_KEY", "")
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    pages_base = os.environ.get("GITHUB_PAGES_URL", "")

    log.info("Starting daily signal pipeline — phase=%d leverage_ok=%s", phase, leverage_ok)

    # ── Data collection ───────────────────────────────────────────────────────
    universe = get_active_universe(phase, leverage_ok)
    tickers = [e.ticker for e in universe]
    log.info("Universe: %d ETFs — %s", len(tickers), tickers)

    ohlcv = fetch_etf_ohlcv(tickers)
    save_parquet(ohlcv, "etf_ohlcv")

    vix = fetch_vix()
    save_parquet(vix, "vix")

    if fred_key:
        macro = fetch_macro(fred_key)
        for name, series in macro.items():
            save_parquet(series, f"macro_{name}")

    # ── Feature engineering + model inference (Phase 1-3 stubs) ──────────────
    signals: list = []
    regime: dict = {"label": "neutral", "exposure": 1.0}

    # TODO Phase 1: compute factors and measure IC
    # TODO Phase 2: baseline rule model
    # TODO Phase 3: LightGBM / GRU model inference
    # TODO Phase 4: signal generation with expectancy gate

    # ── Report ────────────────────────────────────────────────────────────────
    report_path = build_report(signals, regime, today)
    report_url = f"{pages_base}/{report_path.name}" if pages_base else ""

    # ── Notify ────────────────────────────────────────────────────────────────
    if bot_token and chat_id:
        await send_daily_signal(bot_token, chat_id, signals, regime, report_url, today)
    else:
        log.warning("Telegram credentials not set — notification skipped")

    log.info("Pipeline complete for %s", today)


if __name__ == "__main__":
    asyncio.run(main())
