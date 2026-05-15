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
import pandas as pd

from data.collector import fetch_etf_ohlcv, fetch_vix, fetch_macro, save_parquet
from data.universe import get_active_universe, UNIVERSE_BY_TICKER
from models.inference import (
    compute_rolling_stats,
    load_proba_history,
    save_proba_history,
    score_today,
)
from report.builder import build_report
from bot.notifier import send_daily_signal
from paper.tracker import (
    append_and_save as paper_append_and_save,
    close_positions as paper_close_positions,
    load_trades as paper_load_trades,
    open_positions as paper_open_positions,
)
from signals.generator import (
    Signal,
    compute_expectancy,
    compute_levels,
    compute_winrate_ci,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except ValueError:
        return default


def _env_bool(key: str) -> bool:
    return os.environ.get(key, "false").strip().lower() in {"true", "1", "yes"}


def _append_and_save(history, raw_proba):
    from models.inference import append_today_proba
    updated = append_today_proba(history, raw_proba)
    save_proba_history(updated)
    return updated


async def main() -> None:
    today = date.today()
    phase = _env_int("SYSTEM_PHASE", 0)
    leverage_ok = _env_bool("LEVERAGE_EDUCATION_DONE")
    fred_key = os.environ.get("FRED_API_KEY", "")
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    pages_base = os.environ.get("PAGES_URL", "")

    log.info("Starting daily signal pipeline — phase=%d leverage_ok=%s", phase, leverage_ok)

    # ── Data collection ───────────────────────────────────────────────────────
    # Data is always collected for the full Phase 4 universe regardless of current phase.
    # Signal generation is separately gated by phase below.
    DATA_PHASE = 4
    data_universe = get_active_universe(DATA_PHASE, leverage_education_done=False)
    tickers = [e.ticker for e in data_universe]
    log.info("Collecting data for %d ETFs — %s", len(tickers), tickers)

    ohlcv = fetch_etf_ohlcv(tickers)
    save_parquet(ohlcv, "etf_ohlcv")

    vix = fetch_vix()
    save_parquet(vix, "vix")

    if fred_key:
        macro = fetch_macro(fred_key)
        for name, series in macro.items():
            save_parquet(series, f"macro_{name}")

    # ── Phase 3/4: model inference + signal generation ───────────────────────
    signals: list[Signal] = []
    regime: dict = {"label": "neutral", "exposure": 1.0}

    if phase >= 3:
        raw_path = Path("data/raw/etf_ohlcv.parquet")
        vix_path = Path("data/raw/vix.parquet")

        if not raw_path.exists():
            log.error("etf_ohlcv.parquet not found — skipping inference")
        else:
            ohlcv_df = pd.read_parquet(raw_path)
            close_df = (
                ohlcv_df["Close"]
                if isinstance(ohlcv_df.columns, pd.MultiIndex)
                else ohlcv_df
            )
            vix_df = pd.read_parquet(vix_path) if vix_path.exists() else None

            # ── Regime detection from VIX level ──────────────────────────────
            if vix_df is not None and not vix_df.empty:
                vix_latest = float(vix_df.iloc[:, 0].dropna().iloc[-1])
                if vix_latest >= 30:
                    regime = {"label": "fear", "exposure": 0.5, "vix": vix_latest}
                elif vix_latest >= 20:
                    regime = {"label": "caution", "exposure": 0.75, "vix": vix_latest}
                else:
                    regime = {"label": "normal", "exposure": 1.0, "vix": vix_latest}
                log.info("Regime: %s (VIX=%.1f)", regime["label"], vix_latest)

            proba_history = load_proba_history()

            # ── Score today + rolling backtest stats ─────────────────────────
            try:
                score_result, raw_proba = score_today(close_df, vix_df, proba_history)
                rolling_stats = compute_rolling_stats(close_df, vix_df, proba_history)
                proba_history = _append_and_save(proba_history, raw_proba)
            except FileNotFoundError as exc:
                log.warning("Model not found — skipping inference: %s", exc)
                score_result = {}
                rolling_stats = pd.DataFrame()

            # ── Signal generation with expectancy gate (Phase 4) ─────────────
            if phase >= 4 and score_result:
                latest_close = close_df.iloc[-1]
                active_tickers = {e.ticker for e in get_active_universe(phase, leverage_ok)}

                for ticker, scores in score_result.items():
                    if ticker not in active_tickers:
                        continue
                    if scores["signal"] != 1:
                        continue

                    meta = UNIVERSE_BY_TICKER.get(ticker)
                    entry = float(latest_close.get(ticker, 0))
                    if entry <= 0:
                        continue

                    # Per-ETF rolling stats for TP/SL sizing
                    if ticker in rolling_stats.index:
                        row = rolling_stats.loc[ticker]
                        winrate = float(row["winrate"])
                        avg_win = float(row["avg_win"])
                        avg_loss = float(row["avg_loss"])
                        payoff = float(row["payoff"])
                        sample_n = int(row["sample_n"])
                    else:
                        # Insufficient history — skip
                        continue

                    expectancy = compute_expectancy(winrate, avg_win, avg_loss)
                    wins = round(winrate * sample_n)
                    ci_low, ci_high = compute_winrate_ci(wins, sample_n)
                    tp, sl = compute_levels(entry, "long", avg_win, avg_loss)

                    sig = Signal(
                        ticker=ticker,
                        name=meta.name if meta else ticker,
                        direction="long",
                        leverage=meta.leverage if meta else 1,
                        entry=round(entry, 4),
                        tp=tp,
                        sl=sl,
                        winrate=round(winrate, 4),
                        sample_n=sample_n,
                        ci_low=round(ci_low, 4),
                        ci_high=round(ci_high, 4),
                        payoff=round(payoff, 3),
                        expectancy=round(expectancy, 5),
                        confidence=scores["ema_proba"],
                    )
                    if sig.is_valid():
                        signals.append(sig)
                        log.info(
                            "Signal: %s  ema_proba=%.3f  wr=%.2f  payoff=%.2f  E=%.4f",
                            ticker, scores["ema_proba"], winrate, payoff, expectancy,
                        )

                log.info("%d valid signal(s) after expectancy gate", len(signals))

            # ── Paper trading update (Phase 4, Friday-only rebalance) ─────────
            if phase >= 4:
                today_ts = pd.Timestamp(today)
                is_friday = today_ts.dayofweek == 4
                paper_trades = paper_load_trades()

                if is_friday:
                    # Close last week's open positions at today's close
                    paper_trades = paper_close_positions(paper_trades, today_ts, latest_close)
                    # Open new positions for valid signals
                    new_rows = paper_open_positions(signals, today_ts, latest_close)
                    paper_trades = paper_append_and_save(paper_trades, new_rows)
                    if new_rows.empty:
                        log.info("Paper: no valid signals to open today (%s)", today)
                else:
                    log.info("Paper: not Friday (%s), skipping rebalance", today)

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
