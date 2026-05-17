"""Conviction strategy — productized from walk-forward experiments.

v1 (initial): single EMA-5 ≥ 0.65, K=1, regime gate, fixed SL/TP.
v2 (Multi-EMA, 2026-05-17): also requires EMA-3 ≥ 0.65 AND EMA-7 ≥ 0.65.
v3 (Options regime, 2026-05-17): also requires SKEW z-score < 1.0 AND
                                  VIX9D/VIX < 1.0 (contango).

Strategy summary:
    universe: long-only (core + sector, no inverse/VXX/leverage)
    regime:   VIX < 20 AND SPY > 200d SMA
              v3 ALSO: VIX9D/VIX < 1.0 (term structure in contango)
                       SKEW z-score(60d) < 1.0 (tail risk not extreme)
    select:   K=1 by ema_proba (EMA-5), threshold 0.65
              v2 ALSO: ema_proba_3 ≥ 0.65 AND ema_proba_7 ≥ 0.65
    exit:     SL 1.0% intraday OR TP 3.0% intraday OR Friday close after 5 days

Holdout 2024-01 ~ 2026-05 (walk-forward):
    v1:  n=36  WR 58.33%  Payoff 1.20  Total +12.85%
    v2:  n=33  WR 63.64%  Payoff 1.20  Total +16.60%
    v3:  n=20  WR 70.00%  Payoff 1.25  Total +14.37%  🎯 (target reached)

Activated when env STRATEGY_MODE=conviction_v1. v2/v3 toggled via
config.CONVICTION_MULTI_EMA_CONFIRM, CONVICTION_SKEW_Z_MAX, CONVICTION_VIX_TERM_MAX.

Design notes
------------
- Returns at most 1 Signal per call (K=1). When regime gate fails, returns [].
- TP/SL are *fixed percentages* — backtested, not derived from per-ticker
  rolling_stats. The Signal carries the backtested expected winrate / payoff
  / sample_n (from config.CONVICTION_EXPECTED_*) so is_valid() passes.
- Pure function: takes proba + market data, returns signals. No I/O.
- Multi-EMA confirmation requires score_result entries to include
  `ema_proba_3` and `ema_proba_7`. Legacy callers without those fields
  fall back to v1 behavior with a WARNING log.
- v3 options gates require macro indices fetched by data/macro_collector.py
  (vix9d, vix3m, skew). If indices unavailable, conviction.py skips the
  options gates with a WARNING — degrades to v2.
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
              spy_sma200: float | None,
              vix9d_latest: float | None = None,
              skew_z: float | None = None,
              move_z: float | None = None) -> tuple[bool, str]:
    """Check regime gates: VIX, SPY trend, plus optional v3/v4 overlays.

    v3 gates (when config.CONVICTION_SKEW_Z_MAX / VIX_TERM_MAX are set):
      - vix9d_latest / vix_latest < CONVICTION_VIX_TERM_MAX (contango)
      - skew_z < CONVICTION_SKEW_Z_MAX (tail risk not extreme)
    v4 gate (when config.CONVICTION_MOVE_Z_MAX is set):
      - move_z < CONVICTION_MOVE_Z_MAX (bond vol not stressed)
    Any may be None to disable that gate (also via config = None).

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

    # v3: VIX term structure (9D/30D)
    term_max = getattr(config, "CONVICTION_VIX_TERM_MAX", None)
    if term_max is not None:
        if vix9d_latest is None:
            log.warning("Conviction v3: VIX9D unavailable — skipping term gate (degrades to v2)")
        else:
            ratio = vix9d_latest / vix_latest if vix_latest > 0 else float("inf")
            if ratio >= term_max:
                return False, f"VIX9D/VIX {ratio:.3f} >= {term_max} (backwardation)"

    # v3: SKEW z-score
    skew_max = getattr(config, "CONVICTION_SKEW_Z_MAX", None)
    if skew_max is not None:
        if skew_z is None or pd.isna(skew_z):
            log.warning("Conviction v3: SKEW z-score unavailable — skipping skew gate (degrades to v2)")
        else:
            if skew_z >= skew_max:
                return False, f"SKEW z {skew_z:.3f} >= {skew_max} (elevated tail risk)"

    # v4: MOVE bond-vol z-score
    move_max = getattr(config, "CONVICTION_MOVE_Z_MAX", None)
    if move_max is not None:
        if move_z is None or pd.isna(move_z):
            log.warning("Conviction v4: MOVE z-score unavailable — skipping move gate (degrades to v3)")
        else:
            if move_z >= move_max:
                return False, f"MOVE z {move_z:.3f} >= {move_max} (bond market stress)"

    return True, "regime ok"


def select_conviction_signals(
    score_result: dict[str, dict[str, Any]],
    latest_close: pd.Series,
    vix_latest: float | None,
    spy_close: float | None,
    spy_sma200: float | None,
    active_tickers: set[str],
    vix9d_latest: float | None = None,
    skew_z: float | None = None,
    move_z: float | None = None,
) -> list[Signal]:
    """Apply regime gate + long-only universe + K=1 selection.

    Inputs:
      score_result    — from models.inference_v3.score_today: {ticker: {ema_proba,...}}
      latest_close    — today's close price per ticker
      vix_latest      — most recent VIX close
      spy_close, spy_sma200 — for trend gate
      active_tickers  — phase-allowed tickers (passed in by run_daily.py)
      vix9d_latest    — most recent VIX9D close (for v3 term-structure gate)
      skew_z          — current SKEW z-score over CONVICTION_SKEW_Z_WINDOW days

    Returns: 0 or 1 Signal objects. Empty list when regime fails or no eligible
    ticker meets the threshold."""
    ok, reason = regime_ok(vix_latest, spy_close, spy_sma200,
                           vix9d_latest=vix9d_latest, skew_z=skew_z, move_z=move_z)
    if not ok:
        log.info("Conviction: regime gate FAIL — %s — no signal today", reason)
        return []

    # Build (ticker, ema_proba) list filtered by universe + threshold + active phase
    # v2: Multi-EMA confirmation — require EMA-3, EMA-5, EMA-7 all ≥ threshold
    candidates: list[tuple[str, float]] = []
    multi_ema = getattr(config, "CONVICTION_MULTI_EMA_CONFIRM", False)
    rejected_multi_ema = 0
    for ticker, scores in score_result.items():
        if ticker not in active_tickers:
            continue
        meta = UNIVERSE_BY_TICKER.get(ticker)
        if not _is_long_only(meta):
            continue
        ema = float(scores.get("ema_proba", 0.0))
        if ema < config.CONVICTION_THRESHOLD:
            continue
        if multi_ema:
            ema3 = scores.get("ema_proba_3")
            ema7 = scores.get("ema_proba_7")
            if ema3 is None or ema7 is None:
                log.warning("Conviction v2: %s missing ema_proba_3/7 — "
                            "falling back to single-EMA gate", ticker)
            else:
                if float(ema3) < config.CONVICTION_THRESHOLD or float(ema7) < config.CONVICTION_THRESHOLD:
                    rejected_multi_ema += 1
                    continue
        candidates.append((ticker, ema))

    if rejected_multi_ema:
        log.info("Conviction v2: rejected %d candidate(s) on Multi-EMA confirmation",
                 rejected_multi_ema)

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
