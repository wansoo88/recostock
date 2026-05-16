#!/usr/bin/env python3
"""Parameter sweep for intraday backtest — single data fetch, many configs.

Sweeps (min_bars, tp_mult, time_stop_bars) and prints a ranked leaderboard.

Usage:
    python scripts/run_intraday_sweep.py [--days 60]
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.intraday import CORE_TICKERS, SECTOR_TICKERS
from scripts.run_intraday_backtest import (
    fetch_history_5m,
    fetch_daily_vix,
    simulate_day,
    summarize,
)

log = logging.getLogger("sweep")


def run_config(
    ohlcv: dict[str, pd.DataFrame],
    vix: pd.Series,
    min_bars: int,
    min_tp_pct: float,
    min_sl_pct: float,
    time_stop_bars: int,
    reverse: bool = False,
) -> dict:
    """Run one configuration over the pre-fetched data.

    ATR multipliers fixed at base values — 5min ATR is too small to dominate
    over the percentage floors, so the floor IS the effective TP/SL.
    """
    all_trades = []
    for ticker, df in ohlcv.items():
        allow_short = ticker in CORE_TICKERS
        for date_, day_df in df.groupby(df.index.date):
            if len(day_df) < min_bars + 2:
                continue
            prior_dates = vix.index[vix.index < date_]
            vix_level = float(vix.loc[prior_dates[-1]]) if len(prior_dates) else None
            if vix_level is not None and vix_level >= 30 and ticker in SECTOR_TICKERS:
                continue
            trades = simulate_day(
                ticker, day_df, vix_level, allow_short,
                min_bars=min_bars,
                time_stop_bars_before_close=time_stop_bars,
                min_tp_pct=min_tp_pct,
                min_sl_pct=min_sl_pct,
                reverse=reverse,
            )
            all_trades.extend(trades)
    return summarize(all_trades), all_trades


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--out", type=str, default="data/intraday_sweep_results.csv")
    ap.add_argument("--reverse", action="store_true", help="run sweep in mean-reversion mode")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tickers = CORE_TICKERS + SECTOR_TICKERS
    log.info("fetching %dd of 5m bars once: %s", args.days, tickers)
    ohlcv = fetch_history_5m(tickers, args.days)
    if not ohlcv:
        log.error("no data — aborting")
        sys.exit(1)

    start = min(df.index[0] for df in ohlcv.values())
    end = max(df.index[-1] for df in ohlcv.values())
    vix = fetch_daily_vix(start, end + pd.Timedelta(days=1))

    # ── Grid: floor percentages (the actual binding TP/SL on 5min ATR) ─────────
    # For mean-reversion: small TP + wider SL is the natural pattern
    min_bars_grid = [22, 30]
    tp_floor_grid = [0.002, 0.003, 0.005, 0.008]
    sl_floor_grid = [0.003, 0.005, 0.008, 0.012]
    time_stop_grid = [0, 6]   # 0 = none, 6 = 6 bars (=30min) before close

    rows = []
    for mb, tpf, slf, ts in itertools.product(
        min_bars_grid, tp_floor_grid, sl_floor_grid, time_stop_grid
    ):
        log.info("config: min_bars=%d  tp_pct=%.3f  sl_pct=%.3f  time_stop=%d",
                 mb, tpf, slf, ts)
        stats, _ = run_config(ohlcv, vix, mb, tpf, slf, ts, reverse=args.reverse)
        if stats.get("n", 0) == 0:
            continue
        rows.append({
            "min_bars": mb,
            "tp_pct": tpf,
            "sl_pct": slf,
            "time_stop_bars": ts,
            "n_trades": stats["n"],
            "winrate": stats["winrate"],
            "payoff": stats["payoff"],
            "ev_gross": stats["expectancy_gross"],
            "ev_net": stats["expectancy_net"],
            "sharpe": stats["sharpe_daily"],
            "mdd": stats["mdd"],
            "tp_hits": stats["by_reason"].get("TP", 0),
            "sl_hits": stats["by_reason"].get("SL", 0),
            "eod_exits": stats["by_reason"].get("EOD", 0),
            "time_exits": stats["by_reason"].get("TIME", 0),
            "flip_exits": stats["by_reason"].get("FLIP", 0),
        })

    if not rows:
        log.error("no configs produced trades")
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values("ev_net", ascending=False).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # Print leaderboard
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    out = "\n".join([
        "=" * 100,
        f"INTRADAY PARAMETER SWEEP -- {args.days}d, sorted by net expectancy",
        "=" * 100,
        df.to_string(index=False),
        "",
        f"saved: {args.out}",
    ])
    summary_path = args.out.replace(".csv", ".txt")
    Path(summary_path).write_text(out, encoding="utf-8")
    try:
        print(out)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
