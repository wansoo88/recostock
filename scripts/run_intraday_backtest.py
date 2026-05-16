#!/usr/bin/env python3
"""60-day intraday backtest — replays the live signal logic on historical 5-min bars.

Walks `signals.intraday_generator._classify` bar-by-bar with `compute_intraday_features`
on each rolling window, simulates entries/exits with realistic fills, and reports
realized WR / payoff / expectancy / Sharpe / MDD net of 0.25% roundtrip cost.

Conventions (intentionally conservative):
  - Signal decision is made at bar t close; entry/exit fill at bar t+1 open.
  - Same-bar TP+SL touch is treated as SL (assume worst).
  - Position is closed on opposite-direction signal flip; FLAT keeps the
    position alive (matches the live bot: it warns but never auto-closes).
  - All positions force-close at the last bar of the session (LOC behavior).
  - Daily VIX (prior-day close) drives sector-ETF gating and TP/SL widening.
  - Shorts on core ETFs use the underlying's price with sign-flipped PnL —
    equivalent to entering the paired inverse ETF; avoids inverse-ETF price drift.

Usage:
    python scripts/run_intraday_backtest.py [--days 60] [--out data/intraday_backtest.csv]
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.intraday import CORE_TICKERS, INVERSE_PAIR, SECTOR_TICKERS
from features.intraday_factors import compute_intraday_features
from signals.intraday_generator import (
    ADX_TREND_MIN,
    ATR_PERIOD,
    MIN_BARS,
    MIN_SL_PCT,
    MIN_TP_PCT,
    ORB_MINUTES,
    RSI_LONG,
    RSI_SHORT,
    SL_ATR_MULT_BASE,
    SL_ATR_MULT_HIGHVOL,
    STOCHRSI_LONG,
    STOCHRSI_SHORT,
    TP_ATR_MULT_BASE,
    TP_ATR_MULT_HIGHVOL,
)

log = logging.getLogger("backtest")
ET = ZoneInfo("America/New_York")


# ── Data fetch (yfinance) ─────────────────────────────────────────────────────

# yfinance period caps per interval (informational; we just request the smaller of N days and cap)
INTERVAL_MAX_DAYS = {"5m": 60, "15m": 60, "30m": 60, "1h": 730}


def fetch_history(tickers: list[str], days: int, interval: str = "5m") -> dict[str, pd.DataFrame]:
    """Fetch `days` of bars per ticker at the requested interval. ET-indexed."""
    cap = INTERVAL_MAX_DAYS.get(interval, 60)
    period = f"{min(days, cap)}d"
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            raw = yf.download(t, period=period, interval=interval,
                              progress=False, auto_adjust=True)
            if raw.empty:
                log.warning("empty: %s", t)
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.index = pd.DatetimeIndex(raw.index).tz_convert(ET)
            out[t] = raw
            log.info("%s [%s]: %d bars  %s -> %s", t, interval, len(raw),
                     raw.index[0].date(), raw.index[-1].date())
        except Exception as exc:
            log.warning("fetch %s failed: %s", t, exc)
    return out


# Backwards-compat alias (older sweep script imports this name)
def fetch_history_5m(tickers: list[str], days: int) -> dict[str, pd.DataFrame]:
    return fetch_history(tickers, days, interval="5m")


def fetch_daily_vix(start: datetime, end: datetime) -> pd.Series:
    """Daily VIX close, indexed by date. Used as prior-day regime input."""
    raw = yf.download("^VIX", start=start.date(), end=end.date(),
                      interval="1d", progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    s = raw["Close"]
    s.index = pd.DatetimeIndex(s.index).date
    return s


# ── Bar-level classifier (mirrors signals.intraday_generator._classify) ───────

def classify_row(
    last: pd.Series,
    allow_short: bool,
    reverse: bool = False,
    min_atr_pct: float = 0.0,
    vix_level: float | None = None,
    min_vix: float = 0.0,
) -> int:
    """Returns +1 / -1 / 0.

    When `reverse=True`, swaps the LONG/SHORT outcomes — mean-reversion mode.
    Gates (ORB, ADX trend strength, volume) are direction-agnostic and
    apply identically. Bull/bear branches are evaluated regardless of
    allow_short so that a sector ETF can still register an "oversold" event
    that gets flipped to LONG.

    Volatility gates (new):
      min_atr_pct : require ATR(14)/Close >= this fraction to take any signal
      min_vix     : require prior-day VIX >= this level (uses vix_level arg)
    Both default to 0 (disabled).
    """
    if float(last.get("minutes_from_open", 60)) < ORB_MINUTES:
        return 0
    if float(last.get("adx14", 0.0)) < ADX_TREND_MIN:
        return 0
    if float(last.get("vol_ratio", 0.0)) <= 1.0:
        return 0

    # Volatility-conditioned gates
    if min_atr_pct > 0:
        close_px = float(last["Close"])
        if close_px > 0:
            atr_est = float(last.get("atr_for_sizing", float("nan")))
            if math.isnan(atr_est):
                # Approximate via high-low range of the bar
                atr_est = float(last["High"]) - float(last["Low"])
            if (atr_est / close_px) < min_atr_pct:
                return 0
    if min_vix > 0 and (vix_level is None or vix_level < min_vix):
        return 0

    obv_slope = float(last.get("obv_slope", 0.0))

    ema_bull = last["ema5"] > last["ema20"]
    price_bull = last["Close"] > last["vwap"]
    rsi_bull = last["rsi14"] > RSI_LONG
    stochrsi_bull = float(last.get("stochrsi_k", 50)) > STOCHRSI_LONG
    bull_fired = ema_bull and price_bull and rsi_bull and stochrsi_bull and obv_slope > 0

    ema_bear = last["ema5"] < last["ema20"]
    price_bear = last["Close"] < last["vwap"]
    rsi_bear = last["rsi14"] < RSI_SHORT
    stochrsi_bear = float(last.get("stochrsi_k", 50)) < STOCHRSI_SHORT
    bear_fired = ema_bear and price_bear and rsi_bear and stochrsi_bear and obv_slope < 0

    if reverse:
        # Mean-reversion: overbought (bull conditions) -> SHORT, oversold -> LONG
        if bull_fired:
            return -1 if allow_short else 0
        if bear_fired:
            return 1
        return 0

    # Trend-following (original)
    if bull_fired:
        return 1
    if bear_fired and allow_short:
        return -1
    return 0


def tp_sl(
    price: float,
    direction: int,
    atr: float,
    vix: float | None,
    tp_mult_base: float = TP_ATR_MULT_BASE,
    tp_mult_hv: float = TP_ATR_MULT_HIGHVOL,
    sl_mult_base: float = SL_ATR_MULT_BASE,
    sl_mult_hv: float = SL_ATR_MULT_HIGHVOL,
    min_tp_pct: float = MIN_TP_PCT,
    min_sl_pct: float = MIN_SL_PCT,
) -> tuple[float, float]:
    high_vol = vix is not None and vix >= 20
    tp_mult = tp_mult_hv if high_vol else tp_mult_base
    sl_mult = sl_mult_hv if high_vol else sl_mult_base
    tp_gap = max(atr * tp_mult, price * min_tp_pct)
    sl_gap = max(atr * sl_mult, price * min_sl_pct)
    if direction == 1:
        return price + tp_gap, price - sl_gap
    return price - tp_gap, price + sl_gap


# ── Trade simulation ──────────────────────────────────────────────────────────

@dataclass
class Trade:
    ticker: str
    action_ticker: str
    direction: int
    entry_date: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    exit_reason: str      # TP / SL / FLIP / EOD
    hold_bars: int
    gross_pnl: float      # before cost
    net_pnl: float        # after 0.0025 roundtrip cost


def simulate_day(
    ticker: str,
    day_df: pd.DataFrame,
    vix_level: float | None,
    allow_short: bool,
    min_bars: int = MIN_BARS,
    tp_mult: float = TP_ATR_MULT_BASE,
    tp_mult_hv: float = TP_ATR_MULT_HIGHVOL,
    sl_mult: float = SL_ATR_MULT_BASE,
    sl_mult_hv: float = SL_ATR_MULT_HIGHVOL,
    time_stop_bars_before_close: int = 0,
    min_tp_pct: float = MIN_TP_PCT,
    min_sl_pct: float = MIN_SL_PCT,
    reverse: bool = False,
    min_atr_pct: float = 0.0,
    min_vix: float = 0.0,
) -> list[Trade]:
    """One ticker / one session. Returns list of trades closed during the day."""
    feat = compute_intraday_features(day_df)
    # Pre-compute ATR(14)/Close (atr_for_sizing) so classify_row can use it
    hi, lo, cl = feat["High"], feat["Low"], feat["Close"]
    prev_cl = cl.shift(1)
    tr = pd.concat([hi - lo, (hi - prev_cl).abs(), (lo - prev_cl).abs()], axis=1).max(axis=1)
    feat["atr_for_sizing"] = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()
    bars = feat.reset_index().rename(columns={"index": "ts", feat.index.name or "Datetime": "ts"})
    if "ts" not in bars.columns:
        bars["ts"] = feat.index
    bars["ts"] = pd.to_datetime(bars["ts"])

    trades: list[Trade] = []
    position: int = 0       # 0/+1/-1
    entry_idx: int = -1
    entry_px: float = 0.0
    tp_px: float = 0.0
    sl_px: float = 0.0
    n = len(bars)

    time_stop_idx = n - 1 - time_stop_bars_before_close if time_stop_bars_before_close > 0 else None

    for i in range(min_bars, n - 1):  # need bar i+1 for fill
        last = bars.iloc[i]
        next_bar = bars.iloc[i + 1]

        # ── Time-based force-close (e.g., 15:30 ET = 5 bars before 16:00 close) ──
        if position != 0 and time_stop_idx is not None and i >= time_stop_idx:
            exit_px = float(next_bar["Open"])
            gross = position * (exit_px - entry_px) / entry_px
            trades.append(Trade(
                ticker=ticker,
                action_ticker=(INVERSE_PAIR.get(ticker, ticker) if position == -1 else ticker),
                direction=position,
                entry_date=str(bars.iloc[entry_idx]["ts"].date()),
                entry_time=str(bars.iloc[entry_idx]["ts"].time()),
                entry_price=entry_px,
                exit_time=str(next_bar["ts"].time()),
                exit_price=exit_px,
                exit_reason="TIME",
                hold_bars=(i + 1) - entry_idx,
                gross_pnl=gross,
                net_pnl=gross - config.TOTAL_COST_ROUNDTRIP,
            ))
            position = 0
            continue

        # ── Manage existing position first (TP/SL touch within next bar) ──
        if position != 0:
            hi, lo = float(next_bar["High"]), float(next_bar["Low"])
            close_now = False
            reason = ""
            exit_px = 0.0
            if position == 1:
                hit_tp = hi >= tp_px
                hit_sl = lo <= sl_px
                if hit_sl:                # worst-case: SL first
                    exit_px, reason, close_now = sl_px, "SL", True
                elif hit_tp:
                    exit_px, reason, close_now = tp_px, "TP", True
            else:
                hit_tp = lo <= tp_px
                hit_sl = hi >= sl_px
                if hit_sl:
                    exit_px, reason, close_now = sl_px, "SL", True
                elif hit_tp:
                    exit_px, reason, close_now = tp_px, "TP", True

            if close_now:
                gross = position * (exit_px - entry_px) / entry_px
                trades.append(Trade(
                    ticker=ticker,
                    action_ticker=(INVERSE_PAIR.get(ticker, ticker) if position == -1 else ticker),
                    direction=position,
                    entry_date=str(bars.iloc[entry_idx]["ts"].date()),
                    entry_time=str(bars.iloc[entry_idx]["ts"].time()),
                    entry_price=entry_px,
                    exit_time=str(next_bar["ts"].time()),
                    exit_price=exit_px,
                    exit_reason=reason,
                    hold_bars=(i + 1) - entry_idx,
                    gross_pnl=gross,
                    net_pnl=gross - config.TOTAL_COST_ROUNDTRIP,
                ))
                position = 0
                continue   # already used next_bar to exit; do not enter same bar

        # ── New signal evaluation at bar i close ──
        new_dir = classify_row(
            last, allow_short, reverse=reverse,
            min_atr_pct=min_atr_pct, vix_level=vix_level, min_vix=min_vix,
        )

        # Opposite direction → flip: close at next bar open, no re-entry same step
        if position != 0 and new_dir != 0 and new_dir != position:
            exit_px = float(next_bar["Open"])
            gross = position * (exit_px - entry_px) / entry_px
            trades.append(Trade(
                ticker=ticker,
                action_ticker=(INVERSE_PAIR.get(ticker, ticker) if position == -1 else ticker),
                direction=position,
                entry_date=str(bars.iloc[entry_idx]["ts"].date()),
                entry_time=str(bars.iloc[entry_idx]["ts"].time()),
                entry_price=entry_px,
                exit_time=str(next_bar["ts"].time()),
                exit_price=exit_px,
                exit_reason="FLIP",
                hold_bars=(i + 1) - entry_idx,
                gross_pnl=gross,
                net_pnl=gross - config.TOTAL_COST_ROUNDTRIP,
            ))
            position = 0
            continue

        # Open new position on signal from FLAT
        if position == 0 and new_dir != 0:
            entry_idx = i + 1
            entry_px = float(next_bar["Open"])
            atr_val = float(last.get("atr_for_sizing", float("nan")))
            if math.isnan(atr_val):
                # compute ATR on the fly from day_df up to i
                hi = day_df["High"].iloc[: i + 1]
                lo = day_df["Low"].iloc[: i + 1]
                cl = day_df["Close"].iloc[: i + 1]
                prev_cl = cl.shift(1)
                tr = pd.concat([hi - lo, (hi - prev_cl).abs(),
                                (lo - prev_cl).abs()], axis=1).max(axis=1)
                atr_val = float(tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean().iloc[-1])
            tp_px, sl_px = tp_sl(
                entry_px, new_dir, atr_val, vix_level,
                tp_mult_base=tp_mult, tp_mult_hv=tp_mult_hv,
                sl_mult_base=sl_mult, sl_mult_hv=sl_mult_hv,
                min_tp_pct=min_tp_pct, min_sl_pct=min_sl_pct,
            )
            position = new_dir

    # ── EOD force-close ──
    if position != 0:
        last_bar = bars.iloc[-1]
        exit_px = float(last_bar["Close"])
        gross = position * (exit_px - entry_px) / entry_px
        trades.append(Trade(
            ticker=ticker,
            action_ticker=(INVERSE_PAIR.get(ticker, ticker) if position == -1 else ticker),
            direction=position,
            entry_date=str(bars.iloc[entry_idx]["ts"].date()),
            entry_time=str(bars.iloc[entry_idx]["ts"].time()),
            entry_price=entry_px,
            exit_time=str(last_bar["ts"].time()),
            exit_price=exit_px,
            exit_reason="EOD",
            hold_bars=len(bars) - entry_idx,
            gross_pnl=gross,
            net_pnl=gross - config.TOTAL_COST_ROUNDTRIP,
        ))

    return trades


# ── Stats ─────────────────────────────────────────────────────────────────────

def summarize(trades: list[Trade]) -> dict:
    if not trades:
        return {"n": 0}
    df = pd.DataFrame([asdict(t) for t in trades])
    wins = df[df["net_pnl"] > 0]
    losses = df[df["net_pnl"] <= 0]
    n = len(df)
    wr = len(wins) / n
    avg_win = wins["net_pnl"].mean() if not wins.empty else 0.0
    avg_loss = losses["net_pnl"].mean() if not losses.empty else 0.0
    payoff = (avg_win / abs(avg_loss)) if avg_loss != 0 else float("inf")
    expectancy = df["net_pnl"].mean()
    gross_expectancy = df["gross_pnl"].mean()

    daily = df.groupby("entry_date")["net_pnl"].sum()
    if len(daily) > 1 and daily.std() > 0:
        sharpe = daily.mean() / daily.std() * math.sqrt(252)
    else:
        sharpe = float("nan")

    equity = (1 + daily).cumprod()
    peak = equity.cummax()
    mdd = ((equity - peak) / peak).min() if not equity.empty else 0.0

    by_reason = df["exit_reason"].value_counts().to_dict()
    by_ticker = df.groupby("ticker").agg(
        n=("net_pnl", "size"),
        wr=("net_pnl", lambda s: (s > 0).mean()),
        avg=("net_pnl", "mean"),
    ).round(4)

    return {
        "n": n,
        "winrate": round(wr, 4),
        "avg_win": round(avg_win, 5),
        "avg_loss": round(avg_loss, 5),
        "payoff": round(payoff, 3),
        "expectancy_net": round(expectancy, 5),
        "expectancy_gross": round(gross_expectancy, 5),
        "sharpe_daily": round(sharpe, 3),
        "mdd": round(mdd, 4),
        "by_reason": by_reason,
        "by_ticker": by_ticker,
        "n_days": len(daily),
        "avg_trades_per_day": round(n / max(len(daily), 1), 2),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--out", type=str, default="data/intraday_backtest.csv")
    ap.add_argument("--summary-out", type=str, default="data/intraday_backtest_summary.txt")
    ap.add_argument("--min-bars", type=int, default=MIN_BARS,
                    help="warm-up bars before signal evaluation (default 40)")
    ap.add_argument("--tp-mult", type=float, default=TP_ATR_MULT_BASE,
                    help="TP ATR multiplier (base regime, VIX<20)")
    ap.add_argument("--tp-mult-hv", type=float, default=TP_ATR_MULT_HIGHVOL,
                    help="TP ATR multiplier (high-vol, VIX>=20)")
    ap.add_argument("--sl-mult", type=float, default=SL_ATR_MULT_BASE)
    ap.add_argument("--sl-mult-hv", type=float, default=SL_ATR_MULT_HIGHVOL)
    ap.add_argument("--time-stop-bars", type=int, default=0,
                    help="force-close N bars before session end (0 = disabled)")
    ap.add_argument("--label", type=str, default="",
                    help="label for sweep runs (appended to output filenames)")
    ap.add_argument("--interval", type=str, default="5m", choices=["5m", "15m", "30m"],
                    help="bar interval (default 5m)")
    ap.add_argument("--min-tp-pct", type=float, default=MIN_TP_PCT,
                    help="TP floor as fraction of price (default 0.010 = 1%)")
    ap.add_argument("--min-sl-pct", type=float, default=MIN_SL_PCT,
                    help="SL floor as fraction of price (default 0.004 = 0.4%)")
    ap.add_argument("--reverse", action="store_true",
                    help="flip LONG/SHORT (mean-reversion mode based on negative-IC factors)")
    ap.add_argument("--min-atr-pct", type=float, default=0.0,
                    help="volatility gate: require ATR(14)/Close >= this fraction (e.g., 0.001 = 0.1%)")
    ap.add_argument("--min-vix", type=float, default=0.0,
                    help="volatility gate: require prior-day VIX >= this level (e.g., 18)")
    args = ap.parse_args()

    suffix = f"_{args.label}" if args.label else ""
    if args.label:
        args.out = args.out.replace(".csv", f"{suffix}.csv")
        args.summary_out = args.summary_out.replace(".txt", f"{suffix}.txt")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tickers = CORE_TICKERS + SECTOR_TICKERS
    log.info("fetching %dd of %s bars: %s", args.days, args.interval, tickers)
    ohlcv = fetch_history(tickers, args.days, interval=args.interval)

    if not ohlcv:
        log.error("no data fetched — aborting")
        sys.exit(1)

    start = min(df.index[0] for df in ohlcv.values())
    end = max(df.index[-1] for df in ohlcv.values())
    vix = fetch_daily_vix(start, end + pd.Timedelta(days=1))

    all_trades: list[Trade] = []
    for ticker, df in ohlcv.items():
        allow_short = ticker in CORE_TICKERS
        # Split by session date
        for date_, day_df in df.groupby(df.index.date):
            if len(day_df) < args.min_bars + 2:
                continue
            prior_dates = vix.index[vix.index < date_]
            vix_level = float(vix.loc[prior_dates[-1]]) if len(prior_dates) else None
            # fear regime: skip sector ETFs (matches live behavior)
            if vix_level is not None and vix_level >= 30 and ticker in SECTOR_TICKERS:
                continue
            trades = simulate_day(
                ticker, day_df, vix_level, allow_short,
                min_bars=args.min_bars,
                tp_mult=args.tp_mult, tp_mult_hv=args.tp_mult_hv,
                sl_mult=args.sl_mult, sl_mult_hv=args.sl_mult_hv,
                time_stop_bars_before_close=args.time_stop_bars,
                min_tp_pct=args.min_tp_pct, min_sl_pct=args.min_sl_pct,
                reverse=args.reverse,
                min_atr_pct=args.min_atr_pct, min_vix=args.min_vix,
            )
            all_trades.extend(trades)

    stats = summarize(all_trades)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if all_trades:
        pd.DataFrame([asdict(t) for t in all_trades]).to_csv(out_path, index=False)
        log.info("wrote %d trades → %s", len(all_trades), out_path)

    # Pretty print + persist summary
    lines = [
        "=" * 64,
        f"INTRADAY BACKTEST -- {args.days}d, {len(ohlcv)} tickers",
        "=" * 64,
        f"trades:        {stats.get('n', 0)}",
    ]
    if stats.get("n", 0) > 0:
        lines += [
            f"sessions:      {stats['n_days']}  (avg {stats['avg_trades_per_day']} trades/day)",
            f"winrate:       {stats['winrate']:.2%}",
            f"avg win:       {stats['avg_win']:+.3%}   avg loss: {stats['avg_loss']:+.3%}",
            f"payoff:        {stats['payoff']:.2f}",
            f"expectancy:    gross {stats['expectancy_gross']:+.3%}  |  "
            f"net {stats['expectancy_net']:+.3%}  (after 0.25% cost)",
            f"sharpe (d):    {stats['sharpe_daily']:.2f}",
            f"max drawdown:  {stats['mdd']:.2%}",
            f"by reason:     {stats['by_reason']}",
            "",
            "by ticker:",
            str(stats["by_ticker"]),
        ]
    summary = "\n".join(lines)
    Path(args.summary_out).write_text(summary, encoding="utf-8")
    try:
        print(summary)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(summary.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
