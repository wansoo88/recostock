#!/usr/bin/env python3
"""Pairs / relative-strength intraday backtest.

For each (ticker, bar), computes a z-score of the log-price ratio between
the target and a benchmark ETF over a rolling 60-bar window. When the ratio
is statistically stretched (|z| > threshold), enters a SINGLE-LEG trade
betting on mean reversion.

Benchmarks:
  Sector ETF -> SPY                  (most natural beta reference)
  SPY         -> QQQ                  (broad-index cross-check)
  QQQ         -> SPY
  DIA         -> SPY

Single-leg execution means cost stays at 0.25% per round trip (not 0.50%).

5-fold rolling walk-forward is built in to immediately check robustness.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.intraday import CORE_TICKERS, INVERSE_PAIR, SECTOR_TICKERS
from scripts.run_intraday_backtest import (
    fetch_history,
    fetch_daily_vix,
    summarize,
    Trade,
    ATR_PERIOD,
)

log = logging.getLogger("pairs")

# Pairing rule: target -> benchmark
PAIRS = {
    "SPY": "QQQ",
    "QQQ": "SPY",
    "DIA": "SPY",
    "XLK": "SPY",
    "XLF": "SPY",
    "XLE": "SPY",
    "XLV": "SPY",
    "XLY": "SPY",
    "XLI": "SPY",
}


def compute_pair_zscore(
    target_close: pd.Series,
    bench_close: pd.Series,
    lookback: int = 60,
) -> pd.Series:
    """Z-score of log(target) - log(bench) within a rolling window.

    Both series must share the same DatetimeIndex (5min bars, same trading session).
    """
    aligned = pd.concat([target_close, bench_close], axis=1, join="inner").dropna()
    aligned.columns = ["t", "b"]
    spread = np.log(aligned["t"]) - np.log(aligned["b"])
    mean = spread.rolling(lookback, min_periods=lookback // 2).mean()
    std = spread.rolling(lookback, min_periods=lookback // 2).std()
    z = (spread - mean) / std.replace(0, np.nan)
    return z


def simulate_pair_day(
    target_ticker: str,
    bench_ticker: str,
    day_target: pd.DataFrame,
    day_bench: pd.DataFrame,
    allow_short: bool,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    min_bars: int = 22,
    min_tp_pct: float = 0.015,
    min_sl_pct: float = 0.010,
    lookback: int = 60,
) -> list[Trade]:
    """Single-leg mean-reversion on the spread z-score.

    Entry:
      z > z_entry     -> spread is rich (target overpriced vs bench) -> SHORT target
      z < -z_entry    -> spread is cheap (target underpriced) -> LONG target

    Exit:
      TP / SL / EOD / spread z crosses back through z_exit (mean reversion happened)
    """
    bars_t = day_target.copy()
    bars_b = day_bench.copy()
    # Align indexes
    aligned = pd.concat([bars_t["Close"].rename("t_close"),
                         bars_b["Close"].rename("b_close")], axis=1).dropna()
    z = compute_pair_zscore(aligned["t_close"], aligned["b_close"], lookback=lookback)
    if z.dropna().empty:
        return []

    # Attach z to target bars
    bars_t = bars_t.reindex(aligned.index)
    bars_t["z"] = z

    bars = bars_t.reset_index().rename(columns={bars_t.index.name or "Datetime": "ts"})
    if "ts" not in bars.columns:
        bars["ts"] = aligned.index
    bars["ts"] = pd.to_datetime(bars["ts"])

    n = len(bars)
    if n < min_bars + 2:
        return []

    trades = []
    position = 0
    entry_idx = -1
    entry_px = 0.0
    tp_px = 0.0
    sl_px = 0.0

    for i in range(min_bars, n - 1):
        last = bars.iloc[i]
        nxt = bars.iloc[i + 1]
        z_now = float(last["z"])
        if not np.isfinite(z_now):
            continue

        # TP/SL check
        if position != 0:
            hi, lo = float(nxt["High"]), float(nxt["Low"])
            exit_px, reason = 0.0, ""
            if position == 1:
                if lo <= sl_px:
                    exit_px, reason = sl_px, "SL"
                elif hi >= tp_px:
                    exit_px, reason = tp_px, "TP"
            else:
                if hi >= sl_px:
                    exit_px, reason = sl_px, "SL"
                elif lo <= tp_px:
                    exit_px, reason = tp_px, "TP"

            # Spread mean-reverted -> exit
            if not reason:
                if position == 1 and z_now >= -z_exit:
                    exit_px, reason = float(nxt["Open"]), "ZEXIT"
                elif position == -1 and z_now <= z_exit:
                    exit_px, reason = float(nxt["Open"]), "ZEXIT"

            if reason:
                gross = position * (exit_px - entry_px) / entry_px
                trades.append(_make_trade(target_ticker, position, bars, entry_idx,
                                          entry_px, nxt, exit_px, reason,
                                          (i + 1) - entry_idx, gross))
                position = 0
                continue

        # New entry decision
        if position == 0:
            new_dir = 0
            if z_now <= -z_entry:
                new_dir = 1   # LONG cheap target
            elif z_now >= z_entry and allow_short:
                new_dir = -1  # SHORT rich target
            if new_dir != 0:
                entry_idx = i + 1
                entry_px = float(nxt["Open"])
                tp_gap = entry_px * min_tp_pct
                sl_gap = entry_px * min_sl_pct
                if new_dir == 1:
                    tp_px, sl_px = entry_px + tp_gap, entry_px - sl_gap
                else:
                    tp_px, sl_px = entry_px - tp_gap, entry_px + sl_gap
                position = new_dir

    # EOD close
    if position != 0:
        last_bar = bars.iloc[-1]
        exit_px = float(last_bar["Close"])
        gross = position * (exit_px - entry_px) / entry_px
        trades.append(_make_trade(target_ticker, position, bars, entry_idx,
                                  entry_px, last_bar, exit_px, "EOD",
                                  n - entry_idx, gross))
    return trades


def _make_trade(ticker, direction, bars, entry_idx, entry_px,
                exit_row, exit_px, reason, hold, gross):
    return Trade(
        ticker=ticker,
        action_ticker=(INVERSE_PAIR.get(ticker, ticker) if direction == -1 else ticker),
        direction=direction,
        entry_date=str(bars.iloc[entry_idx]["ts"].date()),
        entry_time=str(bars.iloc[entry_idx]["ts"].time()),
        entry_price=entry_px,
        exit_time=str(exit_row["ts"].time()),
        exit_price=exit_px,
        exit_reason=reason,
        hold_bars=hold,
        gross_pnl=gross,
        net_pnl=gross - config.TOTAL_COST_ROUNDTRIP,
    )


# ── Driver ────────────────────────────────────────────────────────────────────

def run_pairs(
    ohlcv: dict,
    sessions: set | None = None,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    min_tp_pct: float = 0.015,
    min_sl_pct: float = 0.010,
    lookback: int = 60,
):
    """Run pairs strategy. If sessions is None, use all available."""
    all_trades = []
    for target, bench in PAIRS.items():
        if target not in ohlcv or bench not in ohlcv:
            continue
        df_t = ohlcv[target]
        df_b = ohlcv[bench]
        allow_short = target in CORE_TICKERS
        for date_, day_t in df_t.groupby(df_t.index.date):
            if sessions is not None and date_ not in sessions:
                continue
            day_b = df_b[df_b.index.date == date_]
            if day_b.empty:
                continue
            trades = simulate_pair_day(
                target, bench, day_t, day_b, allow_short,
                z_entry=z_entry, z_exit=z_exit,
                min_tp_pct=min_tp_pct, min_sl_pct=min_sl_pct,
                lookback=lookback,
            )
            all_trades.extend(trades)
    return all_trades


def rolling_wf(ohlcv: dict, params: dict, n_folds: int = 5) -> pd.DataFrame:
    all_dates = sorted({d for df in ohlcv.values() for d in set(df.index.date)})
    n = len(all_dates)
    min_train = max(15, n // 4)
    oos_size = max(5, (n - min_train) // n_folds)
    rows = []
    for k in range(n_folds):
        train_end = min_train + k * oos_size
        if train_end >= n:
            break
        test_end = min(train_end + oos_size, n)
        oos_dates = set(all_dates[train_end:test_end])
        oos_trades = run_pairs(ohlcv, sessions=oos_dates, **params)
        stats = summarize(oos_trades)
        rows.append({
            "fold": k + 1,
            "oos_end": str(all_dates[test_end - 1]),
            "n": stats.get("n", 0),
            "winrate": stats.get("winrate"),
            "gross": stats.get("expectancy_gross"),
            "net": stats.get("expectancy_net"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--z-entry", type=float, default=2.0)
    ap.add_argument("--z-exit", type=float, default=0.5)
    ap.add_argument("--tp-pct", type=float, default=0.015)
    ap.add_argument("--sl-pct", type=float, default=0.010)
    ap.add_argument("--lookback", type=int, default=60)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tickers = CORE_TICKERS + SECTOR_TICKERS
    log.info("fetching %dd of 5m bars", args.days)
    ohlcv = fetch_history(tickers, args.days, interval="5m")
    if not ohlcv:
        sys.exit(1)

    params = dict(
        z_entry=args.z_entry,
        z_exit=args.z_exit,
        min_tp_pct=args.tp_pct,
        min_sl_pct=args.sl_pct,
        lookback=args.lookback,
    )

    # Full-period baseline
    all_trades = run_pairs(ohlcv, **params)
    full_stats = summarize(all_trades)
    if all_trades:
        pd.DataFrame([asdict(t) for t in all_trades]).to_csv("data/pairs_trades.csv", index=False)

    # 5-fold rolling WF
    wf = rolling_wf(ohlcv, params, n_folds=5)
    nets = wf["net"].dropna().values * 100 if not wf.empty else np.array([])

    lines = ["=" * 80,
             f"PAIRS BACKTEST  --  z_entry={args.z_entry}, z_exit={args.z_exit}, "
             f"TP={args.tp_pct}, SL={args.sl_pct}, lookback={args.lookback}",
             "=" * 80,
             "",
             "FULL PERIOD:",
             f"  trades: {full_stats.get('n', 0)}"]
    if full_stats.get("n", 0) > 0:
        lines += [
            f"  winrate: {full_stats['winrate']:.2%}",
            f"  gross:   {full_stats['expectancy_gross']*100:+.3f}%",
            f"  net:     {full_stats['expectancy_net']*100:+.3f}%",
            f"  payoff:  {full_stats['payoff']:.2f}",
            f"  by_reason: {full_stats['by_reason']}",
            "",
            "by ticker:",
            str(full_stats["by_ticker"]),
        ]
    lines += ["", "=" * 80, "ROLLING WALK-FORWARD (5-fold):", "=" * 80]
    if not wf.empty:
        lines.append(wf.to_string(index=False))
        if len(nets) >= 2:
            mean = nets.mean()
            std = nets.std()
            t_stat = mean / (std / np.sqrt(len(nets))) if std > 0 else 0
            pos = (nets > 0).sum()
            lines += [
                "",
                f"folds positive: {pos}/{len(nets)}",
                f"OOS net mean: {mean:+.3f}%   std: {std:.3f}%   t-stat: {t_stat:+.2f}",
            ]
    out = "\n".join(lines)
    Path("data/intraday_pairs.txt").write_text(out, encoding="utf-8")
    try:
        print(out)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
