"""Conviction strategy v1 — productized from walk-forward experiments.

Strategy summary (see REVIEW_2026-05-17.md §9 and scripts/experiment_tp_sweep.py):
    universe: long-only (core + sector, no inverse/VXX/leverage)
    regime:   VIX < 20 AND SPY > 200d SMA
    select:   K=1 by ema_proba, threshold 0.65
    exit:     SL 1.0% intraday OR TP 3.0% intraday OR Friday close after 5 days

Holdout 2024-01 ~ 2026-05 (n=36 trades, walk-forward):
    WR 58.33%, Payoff 1.20, Sharpe 1.67, MDD -11.1%, Total +13.22%

Activated when env STRATEGY_MODE=conviction_v1. Otherwise the existing
production path (top-5, threshold 0.58, dynamic TP/SL) runs unchanged.

Design notes
------------
- Returns at most 1 Signal per call (K=1). When regime gate fails, returns [].
- TP/SL are *fixed percentages* — backtested, not derived from per-ticker
  rolling_stats. The Signal carries the backtested expected winrate / payoff
  / sample_n (from config.CONVICTION_EXPECTED_*) so is_valid() passes.
- Pure function: takes proba + market data, returns signals. No I/O.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

import config
from data.universe import ETFMeta, UNIVERSE_BY_TICKER
from signals.generator import Signal, compute_expectancy, compute_levels, compute_winrate_ci

log = logging.getLogger(__name__)


def _is_long_only(meta: ETFMeta | None) -> bool:
    """Allow core + sector ETFs. Exclude inverse, volatility, leverage."""
    if meta is None:
        return False
    return meta.category in ("core", "sector")


def regime_ok(vix_latest: float | None,
              spy_close: float | None,
              spy_sma200: float | None) -> tuple[bool, str]:
    """Check VIX and SPY-vs-200d-SMA regime gates.

    Returns (passed, reason_string)."""
    if vix_latest is None:
        return False, "no VIX data"
    if vix_latest >= config.CONVICTION_VIX_MAX:
        return False, f"VIX {vix_latest:.1f} >= {config.CONVICTION_VIX_MAX}"
    if config.CONVICTION_REQUIRE_SPY_UPTREND:
        if spy_close is None or spy_sma200 is None or pd.isna(spy_sma200):
            return False, "no SPY trend reference"
        if spy_close <= spy_sma200:
            return False, f"SPY {spy_close:.2f} <= 200d SMA {spy_sma200:.2f}"
    return True, "regime ok"


def select_conviction_signals(
    score_result: dict[str, dict[str, Any]],
    latest_close: pd.Series,
    vix_latest: float | None,
    spy_close: float | None,
    spy_sma200: float | None,
    active_tickers: set[str],
) -> list[Signal]:
    """Apply regime gate + long-only universe + K=1 selection.

    Inputs:
      score_result    — from models.inference_v3.score_today: {ticker: {ema_proba,...}}
      latest_close    — today's close price per ticker
      vix_latest      — most recent VIX close
      spy_close, spy_sma200 — for trend gate
      active_tickers  — phase-allowed tickers (passed in by run_daily.py)

    Returns: 0 or 1 Signal objects. Empty list when regime fails or no eligible
    ticker meets the threshold."""
    ok, reason = regime_ok(vix_latest, spy_close, spy_sma200)
    if not ok:
        log.info("Conviction: regime gate FAIL — %s — no signal today", reason)
        return []

    # Build (ticker, ema_proba) list filtered by universe + threshold + active phase
    candidates: list[tuple[str, float]] = []
    for ticker, scores in score_result.items():
        if ticker not in active_tickers:
            continue
        meta = UNIVERSE_BY_TICKER.get(ticker)
        if not _is_long_only(meta):
            continue
        ema = float(scores.get("ema_proba", 0.0))
        if ema < config.CONVICTION_THRESHOLD:
            continue
        candidates.append((ticker, ema))

    if not candidates:
        log.info("Conviction: no candidate above threshold %.2f in long-only universe",
                 config.CONVICTION_THRESHOLD)
        return []

    # K=1: highest ema_proba wins
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_k = candidates[:config.CONVICTION_TOP_K]

    signals: list[Signal] = []
    for ticker, ema in top_k:
        entry = float(latest_close.get(ticker, 0.0))
        if entry <= 0:
            log.warning("Conviction: %s has no valid entry price", ticker)
            continue
        meta = UNIVERSE_BY_TICKER.get(ticker)

        tp, sl = compute_levels(entry, "long",
                                tp_pct=config.CONVICTION_TP_PCT,
                                sl_pct=config.CONVICTION_SL_PCT)

        # Use backtested expectations for the Signal stats. Per-ticker
        # rolling_stats can't reflect the new fixed-TP/SL exit policy, so we
        # plug in the holdout-measured averages.
        wr = config.CONVICTION_EXPECTED_WINRATE
        payoff = config.CONVICTION_EXPECTED_PAYOFF
        sample_n = config.CONVICTION_EXPECTED_SAMPLE_N
        # Derive avg_win and avg_loss consistent with TP/SL and payoff:
        # winrate*avg_win - (1-winrate)*avg_loss - cost = expectancy
        # avg_win ~ TP (full TP on win), avg_loss ~ SL (full SL on loss) is
        # an upper bound. Backtest's avg_win was ≈ 1.5%, avg_loss ≈ 1.0%.
        # That makes payoff=1.5/1.0=1.5 in raw terms; reduced to 1.20 after
        # not all wins hit TP. Reverse-engineer from the measured payoff:
        avg_loss = config.CONVICTION_SL_PCT          # 1.0%
        avg_win = avg_loss * payoff                  # 1.2% (slightly below TP 3% — some wins exit at hold-end)
        expectancy = compute_expectancy(wr, avg_win, avg_loss)

        wins = round(wr * sample_n)
        ci_low, ci_high = compute_winrate_ci(wins, sample_n)

        sig = Signal(
            ticker=ticker,
            name=meta.name if meta else ticker,
            direction="long",
            leverage=meta.leverage if meta else 1,
            entry=round(entry, 4),
            tp=tp,
            sl=sl,
            winrate=round(wr, 4),
            sample_n=sample_n,
            ci_low=round(ci_low, 4),
            ci_high=round(ci_high, 4),
            payoff=round(payoff, 3),
            expectancy=round(expectancy, 5),
            confidence=round(ema, 4),
        )
        signals.append(sig)
        log.info("Conviction: %s ema=%.3f entry=%.2f TP=%.2f SL=%.2f payoff=%.2f E=%.4f",
                 ticker, ema, entry, tp, sl, payoff, expectancy)

    return signals
