#!/usr/bin/env python3
"""Volatility-gated parameter sweep — recommendation #1.

Tests whether restricting entries to high-volatility bars/sessions amplifies
gross EV beyond the +0.031% baseline of plain reverse-mode rule-based.

Two gates:
  min_atr_pct : per-bar ATR(14)/Close >= this fraction
  min_vix     : prior-day VIX >= this level

Both gates are applied on top of reverse-direction mean-reversion classifier
(the best baseline from the rule-based exploration).
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.intraday import CORE_TICKERS, SECTOR_TICKERS
from scripts.run_intraday_backtest import (
    fetch_history,
    fetch_daily_vix,
    simulate_day,
    summarize,
)

log = logging.getLogger("vol-sweep")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--out", type=str, default="data/intraday_vol_sweep.csv")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tickers = CORE_TICKERS + SECTOR_TICKERS
    log.info("fetching %dd of 5m bars", args.days)
    ohlcv = fetch_history(tickers, args.days, interval="5m")
    if not ohlcv:
        log.error("no data")
        sys.exit(1)

    start = min(df.index[0] for df in ohlcv.values())
    end = max(df.index[-1] for df in ohlcv.values())
    vix = fetch_daily_vix(start, end + pd.Timedelta(days=1))

    # Fixed best-known params for reverse mode (from prior sweep):
    fixed = dict(
        min_bars=22,
        tp_mult=2.0, tp_mult_hv=2.5, sl_mult=1.0, sl_mult_hv=1.5,
        time_stop_bars_before_close=0,
        min_tp_pct=0.005, min_sl_pct=0.005,
        reverse=True,
    )

    # Gate grid
    atr_pct_grid = [0.0, 0.0005, 0.0008, 0.0012, 0.0018, 0.0025]   # bar ATR/price
    vix_grid     = [0.0, 15.0, 18.0, 22.0, 28.0]                    # prior-day VIX

    rows = []
    for atr_min, vix_min in itertools.product(atr_pct_grid, vix_grid):
        log.info("gate: atr_pct>=%.4f  vix>=%.0f", atr_min, vix_min)
        all_trades = []
        for ticker, df in ohlcv.items():
            allow_short = ticker in CORE_TICKERS
            for date_, day_df in df.groupby(df.index.date):
                if len(day_df) < fixed["min_bars"] + 2:
                    continue
                prior = vix.index[vix.index < date_]
                vix_level = float(vix.loc[prior[-1]]) if len(prior) else None
                if vix_level is not None and vix_level >= 30 and ticker in SECTOR_TICKERS:
                    continue
                trades = simulate_day(
                    ticker, day_df, vix_level, allow_short,
                    **fixed,
                    min_atr_pct=atr_min, min_vix=vix_min,
                )
                all_trades.extend(trades)

        stats = summarize(all_trades)
        if stats.get("n", 0) == 0:
            rows.append({"min_atr_pct": atr_min, "min_vix": vix_min,
                         "n_trades": 0, "winrate": None,
                         "ev_gross": None, "ev_net": None,
                         "payoff": None, "sharpe": None, "mdd": None})
            continue
        rows.append({
            "min_atr_pct": atr_min,
            "min_vix": vix_min,
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
        })

    df_out = pd.DataFrame(rows).sort_values("ev_gross", ascending=False, na_position="last").reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    out = "\n".join([
        "=" * 100,
        f"INTRADAY VOL-GATED SWEEP (reverse mode) -- {args.days}d, sorted by gross EV",
        f"baseline (no gate): gross +0.031%, net -0.219%",
        "=" * 100,
        df_out.to_string(index=False),
    ])
    Path(args.out.replace(".csv", ".txt")).write_text(out, encoding="utf-8")
    try:
        print(out)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
