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
    build_feature_matrix_v3,
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
from signals.sector_rotation import (
    compute_rsi as _rsi14,
    evaluate_weekly as _sector_satellite,   # pick pinned to Friday close — the
    RSI_WINDOW,                             # cadence the blend backtest validated
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
    from models.inference_v3 import append_today_proba
    updated = append_today_proba(history, raw_proba)
    save_proba_history(updated)
    return updated


async def main() -> None:
    today = date.today()
    # Backup-trigger idempotency: the native `schedule` event on daily_signal.yml
    # is only a safety net behind the ubuntu-cron dispatch (primary, 13:00 UTC).
    # If today's report is already published this run has nothing to add — exit
    # before touching data so the user never gets a duplicate telegram.
    if _env_bool("BACKUP_RUN") and (Path("docs") / f"{today.isoformat()}.html").exists():
        log.info("Backup run: docs/%s.html already published — exiting", today.isoformat())
        return
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
    # Single close-price frame for the whole run — inference, fear-dip/trend-core
    # and sizing all reuse this instead of re-reading the parquet.
    close_df: pd.DataFrame | None = None

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

            # Last VALID close per ticker. The raw frame's final row can hold NaN
            # for individual tickers (per-ticker last-row gaps), and NaN entry
            # prices slipped past `entry <= 0` checks. Defined here (not inside
            # the signal block) so the Friday paper rebalance can't hit a
            # NameError when inference is skipped.
            latest_close = close_df.ffill().iloc[-1]

            # ── Stale-data guard ──────────────────────────────────────────────
            # The pipeline publishes signals stamped with TODAY's calendar date.
            # If yfinance silently returns stale prices (API outage, the
            # SPXL-disappearance-type incident), the user would manually trade on
            # old data with no warning. Compare the data's latest close to today
            # and surface the gap. Normal Fri->Mon lag is 3 calendar days, so a
            # gap > STALE_MAX_DAYS means a real freshness problem.
            try:
                from data.collector import data_freshness
                _last_close = close_df.dropna(how="all").index[-1]
                _fresh = data_freshness(_last_close, today)
                regime.update(_fresh)
                if _fresh["stale"]:
                    log.error("STALE DATA: latest close %s is %d days before today %s "
                              "— signals may be based on outdated prices",
                              _fresh["dataAsOf"], _fresh["staleDays"], today)
                else:
                    log.info("Data freshness OK: latest close %s (%d days old)",
                             _fresh["dataAsOf"], _fresh["staleDays"])
            except Exception as exc:
                log.warning("Stale-data check failed (non-fatal): %s", exc)

            # ── Regime detection: VIX + macro overlay ────────────────────────
            if vix_df is not None and not vix_df.empty:
                vix_latest = float(vix_df.iloc[:, 0].dropna().iloc[-1])
                if vix_latest >= 30:
                    regime_label, exposure = "fear", 0.5
                elif vix_latest >= 20:
                    regime_label, exposure = "caution", 0.75
                else:
                    regime_label, exposure = "normal", 1.0
                # update(), NOT reassignment — the stale-data guard above already
                # stored dataAsOf/stale/staleDays in this dict, and a fresh dict
                # here silently dropped them (report/telegram lost the freshness
                # banner whenever VIX data existed, i.e. always; found 2026-06-12).
                regime.update({"label": regime_label, "exposure": exposure, "vix": vix_latest})

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
            strategy_mode = os.environ.get("STRATEGY_MODE", "production_v3")

            # ── Score today + rolling backtest stats ─────────────────────────
            # Feature matrix is built ONCE and shared — score_today and
            # compute_rolling_stats consume the identical X by construction.
            try:
                X = build_feature_matrix_v3(close_df, vix_df)
                score_result, raw_proba = score_today(close_df, vix_df, proba_history, X=X)
                # rolling_stats feeds ONLY the legacy per-ticker loop. Production
                # (conviction mode) uses fixed backtested TP/SL stats instead, so
                # the rolling model replay over a year of history was pure waste
                # there — skip it.
                if strategy_mode == "conviction_v1":
                    rolling_stats = pd.DataFrame()
                else:
                    rolling_stats = compute_rolling_stats(close_df, vix_df, proba_history, X=X)
                proba_history = _append_and_save(proba_history, raw_proba)
            except FileNotFoundError as exc:
                log.warning("Model not found — skipping inference: %s", exc)
                score_result = {}
                rolling_stats = pd.DataFrame()

            # ── Signal generation with expectancy gate (Phase 4) ─────────────
            if phase >= 4 and score_result:
                active_tickers = {e.ticker for e in get_active_universe(phase, leverage_ok)}

                # Conviction strategy — productized from REVIEW_2026-05-17 §9/§10 experiments.
                # Activated by env STRATEGY_MODE=conviction_v1. Walk-forward holdout:
                #   v1 single-EMA:        n=36  WR 58.33%  Payoff 1.20  Total +12.85%
                #   v2 + Multi-EMA:       n=33  WR 63.64%  Payoff 1.20  Total +16.60%
                #   v3 + options regime:  n=20  WR 70.00%  Payoff 1.25  Total +14.37%  🎯
                if strategy_mode == "conviction_v1":
                    # dropna before taking the last value: the raw frame's final
                    # row held NaN for SPY on some days, which flowed into the
                    # trend gate as value=NaN -> silent auto-FAIL (seen live
                    # 2026-06-10: "SPY 추세" gate showed — with threshold null).
                    _spy = close_df["SPY"].dropna() if "SPY" in close_df.columns else pd.Series(dtype=float)
                    spy_close_val = float(_spy.iloc[-1]) if len(_spy) else None
                    spy_sma200_val = (_spy.rolling(200).mean().iloc[-1]
                                  if len(_spy) >= 200 else None)
                    spy_sma200 = float(spy_sma200_val) if spy_sma200_val is not None and not pd.isna(spy_sma200_val) else None
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
                    # Build gates panel for HTML report — user sees WHY signal fired/not
                    vix_term_ratio = (vix9d_latest / vix_latest) if (vix9d_latest and vix_latest) else None
                    regime["gates"] = [
                        {"name": "VIX 절댓값", "value": vix_latest, "threshold": config.CONVICTION_VIX_MAX,
                         "op": "<", "passed": vix_latest is not None and vix_latest < config.CONVICTION_VIX_MAX,
                         "format": ".2f", "desc": "공포지수 (≥20 = 패닉)"},
                        {"name": "SPY 추세", "value": spy_close_val, "threshold": spy_sma200,
                         "op": ">", "passed": spy_close_val is not None and spy_sma200 is not None and spy_close_val > spy_sma200,
                         "format": ".2f", "desc": "200일 이동평균 대비"},
                        {"name": "VIX 단기구조", "value": vix_term_ratio, "threshold": config.CONVICTION_VIX_TERM_MAX,
                         "op": "<", "passed": vix_term_ratio is not None and vix_term_ratio < config.CONVICTION_VIX_TERM_MAX,
                         "format": ".3f", "desc": "VIX9D/VIX (≥1.0 = backwardation 스트레스)"},
                        {"name": "SKEW z-score", "value": skew_z, "threshold": config.CONVICTION_SKEW_Z_MAX,
                         "op": "<", "passed": skew_z is not None and not pd.isna(skew_z) and skew_z < config.CONVICTION_SKEW_Z_MAX,
                         "format": "+.2f", "desc": "60일 SKEW 표준화 (≥1.0 = 꼬리리스크↑)"},
                        {"name": "MOVE z-score", "value": move_z, "threshold": config.CONVICTION_MOVE_Z_MAX,
                         "op": "<", "passed": move_z is not None and not pd.isna(move_z) and move_z < config.CONVICTION_MOVE_Z_MAX,
                         "format": "+.2f", "desc": "60일 채권변동성 (≥1.0 = bond stress)"},
                    ]
                    # WR is the weekly-K=1 cadence (one best name/week). Disclose
                    # sample size: the 73.7% holdout figure is n=19 — small. Full
                    # OOS is 61% (n=31). Threshold stays 0.65: a 2026-05-20 sweep
                    # showed raising it to 0.75 starves the strategy (n=2-5) and
                    # lowers total return, since K=1 already selects for quality.
                    regime["strategy"] = {
                        "version": "conviction_v4",
                        "description": "주간 K=1 + Multi-EMA(3,5,7) + 5-gate regime + 고정 TP3%/SL1%",
                        "holdout_wr": 0.7368, "holdout_n": 19,
                        "full_wr": 0.613, "full_n": 31,
                    }

                    # Sector/core RELATIVE-STRENGTH monitor (rebuilt 2026-05-29).
                    # The model's confidence proba has ~zero OOS rank correlation
                    # with wins (Spearman -0.02, see signals/calibration.py), so the
                    # previous "rank by confidence" watchlist was ranking by NOISE:
                    # it froze on one ticker (XLI) for weeks and stamped a fixed
                    # +3%/-1% target on every name regardless of its volatility.
                    # This ranks by relative strength (1m+3m momentum — rotates
                    # daily, economically grounded) and sizes the displayed range
                    # by each ETF's own realized volatility. It is a CONTEXT
                    # monitor, not a trade signal (live action = trend-core panel).
                    # `passed` is still the TRUE conviction gate (raw EMA >= 0.65
                    # multi-EMA) so the regime-gates banner stays accurate.
                    thr_c = config.CONVICTION_THRESHOLD
                    candidates: list[dict] = []
                    for tk, sc in score_result.items():
                        m = UNIVERSE_BY_TICKER.get(tk)
                        if m is None or m.category not in ("core", "sector"):
                            continue
                        if tk not in active_tickers:
                            continue
                        e5 = sc.get("ema_proba")
                        if e5 is None:
                            continue
                        e3, e7 = sc.get("ema_proba_3"), sc.get("ema_proba_7")
                        series = close_df[tk].dropna() if tk in close_df.columns else None
                        if series is None or len(series) < 64:
                            continue
                        px = float(series.iloc[-1])
                        if px <= 0:
                            continue
                        # Relative strength: blended 1-month + 3-month return.
                        rs = (0.5 * (px / float(series.iloc[-22]) - 1.0)
                              + 0.5 * (px / float(series.iloc[-64]) - 1.0))
                        # Trend posture vs 50/200-day SMA.
                        sma50 = float(series.tail(50).mean())
                        sma200 = float(series.tail(200).mean()) if len(series) >= 200 else sma50
                        above50, above200 = px > sma50, px > sma200
                        # Volatility-scaled 5-day expected range (+/-1 sigma) — the
                        # honest, per-ETF replacement for the fixed 3% target.
                        dvol = float(series.pct_change().dropna().tail(20).std())
                        band = dvol * (5 ** 0.5) if dvol == dvol else 0.0
                        # RSI-14 — the IC-validated cross-sectional ranking key
                        # (sectors: IC +0.035 t=3.5 @h=5). See signals/sector_rotation.
                        rsi14 = (float(_rsi14(series).iloc[-1])
                                 if len(series) >= RSI_WINDOW + 1 else float("nan"))
                        # True conviction gate (drives the regime-gates banner only).
                        passed = (float(e5) >= thr_c and e3 is not None and e7 is not None
                                  and float(e3) >= thr_c and float(e7) >= thr_c)
                        # NO calWin/estEv here: the isotonic calibration is flat
                        # (~57% for every name — the model can't rank names), so
                        # per-ETF "win prob" is fake precision. Removed from the
                        # payload entirely (CLAUDE.md: 컬럼 부활 금지).
                        candidates.append({
                            "ticker": tk, "name": m.name,
                            "confidence": round(float(e5), 4),
                            "rs": round(rs, 4),
                            "rsi": round(rsi14, 1) if rsi14 == rsi14 else None,
                            "above50": bool(above50), "above200": bool(above200),
                            "entry": round(px, 2),
                            "hi": round(px * (1 + band), 2),
                            "lo": round(px * (1 - band), 2),
                            "bandPct": round(band, 4),
                            "passed": bool(passed),
                        })
                    # Rank by RSI-14 — the IC-validated cross-sectional key
                    # (sectors IC +0.035 t=3.5; beats the 1m+3m momentum sort
                    # head-to-head +147%/1.30 vs +121%/1.02). NaN RSI sinks last.
                    candidates.sort(
                        key=lambda c: (c["rsi"] if c.get("rsi") is not None else -1.0),
                        reverse=True,
                    )
                    regime["candidates"] = candidates
                    regime["candidateThreshold"] = thr_c

                    # RSI-14 sector-rotation SATELLITE (validated 2026-05-30).
                    # The LightGBM model has ~zero cross-sectional skill (IC~0),
                    # but a causal RSI-14 ranks the 6 sectors (IC +0.035 t=3.5).
                    # Optional value-add layer (like fear-dip), NOT the core —
                    # carries -22% standalone MDD and trails in low-vol years.
                    try:
                        regime["sectorSatellite"] = _sector_satellite(close_df)
                    except Exception as exc:
                        log.warning("Sector satellite eval failed (non-fatal): %s", exc)

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

    # ── Fear-dip experimental signal (paper-only, added 2026-05-20) ───────────
    # Mirror of the exhausted short search: every directional/pair/put short of
    # the bearish composite lost, but BUYING SPY on its causal extreme is
    # OOS-consistent. Paper track only — accumulates an out-of-sample record
    # before any Tier gating. NOT a live signal.
    if phase >= 3:
        try:
            from signals.fear_dip import evaluate as fear_dip_eval
            import paper.fear_dip_tracker as fdt
            if close_df is not None:
                _c = close_df
                fd_sig = fear_dip_eval(_c)
                fd_trades = fdt.update(_c, fd_sig, pd.Timestamp(today))
                fd_metrics = fdt.metrics(fd_trades)
                regime["fearDip"] = {
                    "isEntry": fd_sig["is_entry"], "score": fd_sig["score"],
                    "threshold": fd_sig["threshold"], "percentile": fd_sig["percentile"],
                    "entry": fd_sig["entry_price"], "paper": fd_metrics,
                }
                log.info("Fear-dip(exp): entry=%s pct=%s%% paper(n=%d wr=%.0f%% tot=%.1f%% open=%d)",
                         fd_sig["is_entry"],
                         f"{(fd_sig['percentile'] or 0)*100:.0f}",
                         fd_metrics["n"], fd_metrics["winrate"] * 100,
                         fd_metrics["total"] * 100, fd_metrics["open"])

                # Trend-following core + fear-dip leverage tilt — PRIMARY engine.
                # fear-dip is "active" while a paper position is open (its 10-day
                # window), which is the panic-bounce tilt trigger.
                from signals.trend_core import evaluate as trend_eval
                fd_active = bool(fd_metrics.get("open", 0) > 0 or fd_sig["is_entry"])
                # Pass the open paper position's entry date (for tilt-window countdown).
                _open = fd_trades[fd_trades["status"] == "open"] if not fd_trades.empty else fd_trades
                fd_open_date = str(_open.iloc[0]["entry_date"]) if len(_open) else None
                regime["trendCore"] = trend_eval(_c, fd_active, regime.get("vix"), fd_open_date)
                tc = regime["trendCore"]
                log.info("Trend-core: regime=%s exposure=%.2fx (SPY %.0f%% / SPXL %.0f%%)",
                         tc.get("regime"), tc.get("effExposure", 0),
                         tc.get("spyWeight", 0) * 100, tc.get("spxlWeight", 0) * 100)

                # ── Composed portfolio: trend-core engine + RSI sector sleeve ──
                # The single actionable allocation the user executes. The sleeve
                # (regime["sectorSatellite"], set in the conviction block) adds the
                # one validated cross-sectional edge the engine lacks. Weight is
                # config.SECTOR_SLEEVE_WEIGHT (default 15%); falls back to the pure
                # engine when the sleeve is unavailable.
                try:
                    from signals.portfolio import compose as _compose
                    regime["portfolio"] = _compose(tc, regime.get("sectorSatellite"))
                    pf = regime["portfolio"]
                    log.info("Portfolio(blend): eff=%.2fx core=%.0f%% sleeve=%.0f%% -> %s",
                             pf.get("effExposure", 0), pf.get("coreWeight", 1) * 100,
                             pf.get("sleeveWeight", 0) * 100 if pf.get("enabled") else 0,
                             pf.get("weights"))
                except Exception as exc:
                    log.warning("Portfolio compose failed (non-fatal): %s", exc)

                # ── Best-pick "long shot" satellite (researched 2026-06-24) ────
                # A concentrated single-name weekly pick from the expanded
                # universe (sectors / sectors+3x). REPORT-ONLY reference, like
                # fear-dip — NOT in the telegram instruction and NOT moving the
                # live blend (frozen during paper validation). Both gated modes
                # cleared Tier-1 in scripts/research_best_pick.py; +3%/wk is a
                # take-profit target, not an expected mean (see signals.best_pick).
                try:
                    from signals import best_pick as _bestpick
                    regime["bestPick"] = {
                        m: _bestpick.select_weekly(_c, m)
                        for m in ("disciplined", "longshot")
                    }
                    _bp = regime["bestPick"]["disciplined"]
                    log.info("Best-pick(satellite): disciplined=%s longshot=%s",
                             _bp.get("pick") or "cash",
                             regime["bestPick"]["longshot"].get("pick") or "cash")
                except Exception as exc:
                    log.warning("Best-pick eval failed (non-fatal): %s", exc)

                # ── Today's single decision: target blend vs current holdings ──
                # The ONE instruction the user executes. Diffs today's target
                # against the last tracker record (what the user holds), so the
                # telegram/report can lead with "오늘 할 일" instead of making the
                # user reconcile several parallel recommendation panels.
                try:
                    from signals.decision import build_decision
                    import paper.portfolio_tracker as _pfpaper
                    if regime.get("portfolio"):
                        # Same record-date convention as portfolio_tracker.update.
                        _data_date = pd.Timestamp(_c.index[-1]).normalize()
                        # Latest valid close per target ticker — feeds the
                        # report's share-count calculator.
                        _prices = {}
                        for _tk in (regime["portfolio"].get("weights") or {}):
                            _v = latest_close.get(_tk)
                            if _v is not None and _v == _v and float(_v) > 0:
                                _prices[_tk] = round(float(_v), 2)
                        _prev = _pfpaper.last_weights_before(_data_date)
                        # Real holdings (read-only Toss snapshot, synced by the
                        # ubuntu server) replace the tracker ASSUMPTION when
                        # fresh; the broker-vs-tracker drift is reported so a
                        # missed manual execution surfaces the next morning.
                        # Absent/stale snapshot → tracker fallback (pre-
                        # integration behavior, e.g. while key approval pends).
                        try:
                            from broker import reconcile as _brk
                            _bprev = _brk.load_holdings(today=_data_date)
                            if _bprev:
                                _dr = _brk.drift(_bprev, _prev)
                                if _dr:
                                    regime["brokerReconcile"] = _dr
                                _prev = _bprev
                        except Exception as exc:
                            log.warning("Broker holdings unavailable (non-fatal): %s", exc)
                        regime["decision"] = build_decision(
                            regime["portfolio"], tc,
                            prev=_prev,
                            satellite=regime.get("sectorSatellite"),
                            fear_dip=regime.get("fearDip"),
                            vix=regime.get("vix"),
                            prices=_prices,
                        )
                        _d = regime["decision"]
                        log.info("Decision: %s — %d trade(s), vs %s holdings of %s",
                                 _d["stance"], _d["nTrades"],
                                 _d.get("prevSource") or "n/a",
                                 _d.get("prevDate") or "n/a")
                except Exception as exc:
                    log.warning("Decision build failed (non-fatal): %s", exc)

                # ── Portfolio NAV paper validation (3-month Tier-2 track) ──────
                # Forward out-of-sample: record today's blend allocation + the
                # realized return from holding the prior run's weights. After 3
                # months the realized Sharpe is checked vs the backtest (drift
                # gate). Reporting only — never auto-trades. NO backfill.
                try:
                    import paper.portfolio_tracker as _pfpaper
                    if regime.get("portfolio"):
                        _pfpaper.update(_c, regime["portfolio"])
                        regime["portfolioPaper"] = _pfpaper.metrics()
                        # One-time Tier-2 maturity alert: the first run where the
                        # 3-month window completes flags the telegram so the
                        # checkpoint day (~2026-08-29) can't pass unnoticed. The
                        # marker lives in data/paper/ (committed by the workflow)
                        # so it fires exactly once.
                        _t2flag = Path("data/paper/tier2_maturity_alerted.flag")
                        if regime["portfolioPaper"].get("monthsOk") and not _t2flag.exists():
                            regime["portfolioPaper"]["maturityAlert"] = True
                            _t2flag.parent.mkdir(parents=True, exist_ok=True)
                            _t2flag.write_text(str(today), encoding="utf-8")
                            log.info("Tier-2 paper window MATURED — one-time alert flagged")
                        # NAV chart + per-leg attribution for the report
                        # (display only — additive approximation, labeled).
                        regime["portfolioPaper"]["history"] = _pfpaper.nav_history()
                        try:
                            _attr = _pfpaper.attribution(_c)
                            if _attr:
                                regime["portfolioPaper"]["attribution"] = _attr
                        except Exception as exc:
                            log.warning("Attribution failed (non-fatal): %s", exc)
                        _pp = regime["portfolioPaper"]
                        log.info("Portfolio paper: day %d / %.1f months · NAV %+.1f%% · "
                                 "Sharpe %.2f (target %.2f, gap %s) · %s",
                                 _pp["nDays"], _pp["months"], _pp["totalReturn"] * 100,
                                 _pp["annSharpe"], _pp["targetSharpe"],
                                 f"{_pp['gap']:.0%}" if _pp["gap"] is not None else "n/a",
                                 _pp["status"])
                except Exception as exc:
                    log.warning("Portfolio paper tracking failed (non-fatal): %s", exc)
        except Exception as exc:
            log.warning("Fear-dip/trend-core eval failed (non-fatal): %s", exc, exc_info=True)

    # ── All-weather ensemble verdict (added 2026-05-22) ───────────────────────
    # conviction (trend, calm uptrend regime) and fear-dip (mean-reversion,
    # stress regime) fire in OPPOSITE regimes — 0 overlapping days in the OOS
    # backtest. Combined: 105 trades (3.4x conviction alone), Holdout Sharpe
    # 2.41, +96% Full. One unified daily action drawn from whichever fires;
    # conviction is LIVE, fear-dip is EXPERIMENTAL (paper).
    live_sig = signals[0] if signals else None
    fd = regime.get("fearDip") or {}
    if live_sig is not None:
        ensemble = {"action": "conviction", "live": True,
                    "ticker": live_sig.ticker, "name": live_sig.name,
                    "entry": live_sig.entry, "tp": live_sig.tp, "sl": live_sig.sl,
                    "note": "추세추종 (calm 상승 레짐) — 실전 신호"}
    elif fd.get("isEntry"):
        ensemble = {"action": "feardip", "live": False,
                    "ticker": "SPY", "name": "SPDR S&P 500",
                    "entry": fd.get("entry"),
                    "note": "공포 평균회귀 (스트레스 레짐) — 실험·페이퍼, 10일 보유"}
    else:
        ensemble = {"action": "none", "live": False,
                    "note": "오늘은 두 전략 모두 미발사 — 관망"}
    ensemble["backtest"] = {"combinedTrades": 105, "holdoutSharpe": 2.41,
                            "overlapDays": 0, "fullTotal": 96.1}
    # Volatility-targeted position size for the actionable ticker.
    if ensemble["action"] != "none":
        try:
            from signals.sizing import position_size_pct
            if close_df is not None:
                tk = ensemble["ticker"]
                if tk in close_df.columns:
                    ensemble["sizing"] = position_size_pct(close_df[tk].dropna())
        except Exception as exc:
            log.warning("Sizing failed (non-fatal): %s", exc)
    regime["ensemble"] = ensemble
    _sz = (ensemble.get("sizing") or {}).get("sizePct")
    log.info("Ensemble verdict: %s (live=%s) size=%s", ensemble["action"],
             ensemble["live"], f"{_sz*100:.0f}%" if _sz else "n/a")

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
