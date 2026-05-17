#!/usr/bin/env python3
"""Daily signal pipeline entry point (GitHub Actions + local).

Usage:
    SYSTEM_PHASE=4 FRED_API_KEY=xxx python scripts/run_daily.py
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import sys
from datetime import date
from pathlib import Path

# Ensure repo root is on PYTHONPATH when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import pandas as pd

from data.collector import (
    fetch_etf_ohlcv, fetch_vix, fetch_macro,
    fetch_macro_yfinance, save_parquet,
)
from data.universe import get_active_universe, UNIVERSE_BY_TICKER
# v3 production inference: macro features + top-K selection (threshold 0.58)
# Switched from models.inference (v1) on 2026-05-17 per user approval (option 1).
from models.inference_v3 import (
    compute_rolling_stats,
    load_proba_history,
    save_proba_history,
    score_today,
)
from report.builder import build_report
from bot.notifier import send_daily_signal
from paper.tracker import (
    compute_metrics as paper_compute_metrics,
    load_trades as paper_load_trades,
    rebalance_friday as paper_rebalance_friday,
    save_trades as paper_save_trades,
)
from paper.tier2 import evaluate_tier2, is_phase5_ready
from signals.generator import (
    Signal,
    compute_expectancy,
    compute_levels,
    compute_winrate_ci,
)
from signals.conviction import select_conviction_signals

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
    from models.inference_v3 import append_today_proba
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

    # ── Phase 5 Tier 2 gate check ─────────────────────────────────────────────
    if phase >= 5:
        t2 = evaluate_tier2(include_backfill=False)
        if not t2["passed"]:
            log.warning("Phase 5 active but Tier 2 gate not fully passed — "
                        "leverage ETFs will be skipped unless leverage_ok=True and gate passed")
        else:
            log.info("Tier 2 gate PASSED — Phase 5 leverage ETFs eligible (if leverage_ok)")

    # ── Data collection ───────────────────────────────────────────────────────
    # Data is always collected for the full Phase 4 universe regardless of current phase.
    # Signal generation is separately gated by phase below.
    DATA_PHASE = max(4, phase)
    data_universe = get_active_universe(DATA_PHASE, leverage_education_done=leverage_ok and phase >= 5)
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
        del macro

    # v3 model requires yfinance-based macro (oil, gold, DXY, yields, HY/IG, etc.)
    try:
        fetch_macro_yfinance()
    except Exception as exc:
        log.warning("yfinance macro fetch failed (v3 inference may degrade): %s", exc)

    # SPY Put/Call ratio — monitoring only (added 2026-05-18, becomes a
    # candidate regime gate once 60+ days of history accumulate).
    try:
        from data.options_pc_collector import append_today as _append_pc, status_summary as _pc_status
        pc_row = _append_pc()
        log.info(_pc_status())
        if pc_row:
            log.info("Today's SPY P/C — vol=%.3f  OI=%.3f  (monitoring, NOT a gate yet)",
                     pc_row["vol_pc"], pc_row["oi_pc"])
    except Exception as exc:
        log.warning("Put/Call collector failed (non-fatal): %s", exc)

    # Free fetched objects before loading parquet for inference
    del ohlcv, vix
    gc.collect()

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

            # ── Regime detection: VIX + macro overlay ────────────────────────
            if vix_df is not None and not vix_df.empty:
                vix_latest = float(vix_df.iloc[:, 0].dropna().iloc[-1])
                if vix_latest >= 30:
                    regime_label, exposure = "fear", 0.5
                elif vix_latest >= 20:
                    regime_label, exposure = "caution", 0.75
                else:
                    regime_label, exposure = "normal", 1.0
                regime = {"label": regime_label, "exposure": exposure, "vix": vix_latest}

                # Macro overlay: credit spread and yield curve can escalate regime.
                # Credit spread (BAMLH0A0HYM2) normal ~3-4%, stress ≥5%, crisis ≥7%.
                # Yield spread (10y-2y) inversion < -0.5% signals growth concern.
                macro_triggers: list[str] = []

                cs_path = Path("data/raw/macro_credit_spread.parquet")
                if cs_path.exists():
                    cs_df = pd.read_parquet(cs_path).dropna()
                    if not cs_df.empty:
                        cs = float(cs_df.iloc[-1, 0])
                        regime["credit_spread"] = round(cs, 2)
                        if cs >= 7.0 and regime["label"] != "fear":
                            regime["label"] = "fear"
                            regime["exposure"] = 0.5
                            macro_triggers.append(f"credit_spread={cs:.1f}%≥7")
                        elif cs >= 5.0 and regime["label"] == "normal":
                            regime["label"] = "caution"
                            regime["exposure"] = 0.75
                            macro_triggers.append(f"credit_spread={cs:.1f}%≥5")

                y10_path = Path("data/raw/macro_yield_10y.parquet")
                y2_path = Path("data/raw/macro_yield_2y.parquet")
                if y10_path.exists() and y2_path.exists():
                    y10_df = pd.read_parquet(y10_path).dropna()
                    y2_df = pd.read_parquet(y2_path).dropna()
                    if not y10_df.empty and not y2_df.empty:
                        yield_spread = float(y10_df.iloc[-1, 0]) - float(y2_df.iloc[-1, 0])
                        regime["yield_spread"] = round(yield_spread, 2)
                        if yield_spread < -0.5 and regime["label"] == "normal":
                            regime["label"] = "caution"
                            regime["exposure"] = 0.75
                            macro_triggers.append(f"yield_curve={yield_spread:.2f}%")

                if macro_triggers:
                    log.info("Regime escalated by macro: %s", ", ".join(macro_triggers))
                log.info(
                    "Regime: %s (VIX=%.1f, exposure=%.2f, credit_spread=%s, yield_spread=%s)",
                    regime["label"], vix_latest, regime["exposure"],
                    regime.get("credit_spread", "n/a"), regime.get("yield_spread", "n/a"),
                )

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

                # Conviction strategy — productized from REVIEW_2026-05-17 §9/§10 experiments.
                # Activated by env STRATEGY_MODE=conviction_v1. Walk-forward holdout:
                #   v1 single-EMA:        n=36  WR 58.33%  Payoff 1.20  Total +12.85%
                #   v2 + Multi-EMA:       n=33  WR 63.64%  Payoff 1.20  Total +16.60%
                #   v3 + options regime:  n=20  WR 70.00%  Payoff 1.25  Total +14.37%  🎯
                strategy_mode = os.environ.get("STRATEGY_MODE", "production_v3")
                if strategy_mode == "conviction_v1":
                    spy_close_val = float(close_df["SPY"].iloc[-1]) if "SPY" in close_df.columns else None
                    spy_sma200 = (close_df["SPY"].rolling(200).mean().iloc[-1]
                                  if "SPY" in close_df.columns else None)
                    spy_sma200 = float(spy_sma200) if spy_sma200 is not None and not pd.isna(spy_sma200) else None
                    vix_latest = regime.get("vix")

                    # v3 options-regime inputs: VIX9D and SKEW z-score from macro cache
                    # v4 bond-vol input: MOVE z-score
                    vix9d_latest = None
                    skew_z = None
                    move_z = None

                    def _z_score_latest(path: Path, window: int) -> float | None:
                        if not path.exists():
                            return None
                        series = pd.read_parquet(path).iloc[:, 0].dropna()
                        if len(series) < window + 1:
                            return None
                        tail = series.tail(window + 1)
                        mu = tail.iloc[:-1].mean()
                        sigma = tail.iloc[:-1].std()
                        if sigma <= 0:
                            return None
                        return float((tail.iloc[-1] - mu) / sigma)

                    try:
                        vix9d_path = Path("data/raw/macro/vix9d.parquet")
                        if vix9d_path.exists():
                            vix9d_series = pd.read_parquet(vix9d_path).iloc[:, 0].dropna()
                            if not vix9d_series.empty:
                                vix9d_latest = float(vix9d_series.iloc[-1])
                        skew_z = _z_score_latest(
                            Path("data/raw/macro/skew.parquet"),
                            config.CONVICTION_SKEW_Z_WINDOW,
                        )
                        move_z = _z_score_latest(
                            Path("data/raw/macro/move.parquet"),
                            config.CONVICTION_MOVE_Z_WINDOW,
                        )
                    except Exception as exc:
                        log.warning("Conviction v3/v4: regime fetch failed: %s", exc)

                    signals = select_conviction_signals(
                        score_result=score_result,
                        latest_close=latest_close,
                        vix_latest=vix_latest,
                        spy_close=spy_close_val,
                        spy_sma200=spy_sma200,
                        active_tickers=active_tickers,
                        vix9d_latest=vix9d_latest,
                        skew_z=skew_z,
                        move_z=move_z,
                    )
                    # Filter via is_valid() to match honesty principle #3.
                    signals = [s for s in signals if s.is_valid()]
                    log.info("Conviction: %d valid signal(s) after regime + is_valid gates "
                             "(VIX9D=%s SKEW_z=%s MOVE_z=%s)", len(signals),
                             f"{vix9d_latest:.2f}" if vix9d_latest else "N/A",
                             f"{skew_z:+.2f}" if skew_z is not None else "N/A",
                             f"{move_z:+.2f}" if move_z is not None else "N/A")
                    # Skip the legacy loop below — we already produced signals.
                    score_result = {}  # neutralize legacy loop without altering its structure

                for ticker, scores in score_result.items():
                    if ticker not in active_tickers:
                        continue
                    if scores["signal"] != 1:
                        continue

                    meta = UNIVERSE_BY_TICKER.get(ticker)

                    # Phase 5 gate: leverage ETFs need high confidence + education
                    if meta and meta.requires_education:
                        if not leverage_ok:
                            continue
                        if scores["ema_proba"] < config.LEVERAGE_CONFIDENCE_THRESHOLD:
                            continue

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

            # ── Paper trading update (Phase 4, Friday carry-over rebalance) ──
            if phase >= 4:
                today_ts = pd.Timestamp(today)
                is_friday = today_ts.dayofweek == 4
                paper_trades = paper_load_trades()

                if is_friday:
                    # Carry-over model: close exits, open new entries only
                    paper_trades = paper_rebalance_friday(
                        paper_trades, signals, today_ts, latest_close
                    )
                    paper_save_trades(paper_trades)
                else:
                    log.info("Paper: not Friday (%s), skipping rebalance", today)

    # ── Paper metrics for report ──────────────────────────────────────────────
    paper_metrics: dict = {}
    paper_open_list: list[dict] = []
    if phase >= 4:
        try:
            _pt = paper_load_trades()
            paper_metrics = paper_compute_metrics(_pt)
            _open = _pt[_pt["close_date"].isna()] if not _pt.empty else _pt
            paper_open_list = _open[["open_date", "ticker", "entry_price", "ema_proba", "winrate", "payoff"]].to_dict("records")
        except Exception as exc:
            log.warning("Paper metrics unavailable: %s", exc)

    # ── Report ────────────────────────────────────────────────────────────────
    report_path = build_report(signals, regime, today,
                               paper_metrics=paper_metrics,
                               paper_open=paper_open_list)
    report_url = f"{pages_base}/{report_path.name}" if pages_base else ""

    # ── Notify ────────────────────────────────────────────────────────────────
    if bot_token and chat_id:
        await send_daily_signal(bot_token, chat_id, signals, regime, report_url, today)
    else:
        log.warning("Telegram credentials not set — notification skipped")

    log.info("Pipeline complete for %s", today)


if __name__ == "__main__":
    asyncio.run(main())
