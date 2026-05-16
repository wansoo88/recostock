#!/usr/bin/env python3
"""Direct IS/OOS split for the specific 'winner' config without grid search.

If even this fails OOS, the entire vol-gated reverse strategy was a 60-day
in-sample artifact. No grid search snooping involved here.
"""
from __future__ import annotations

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
from scripts.run_intraday_walkforward import run_window

log = logging.getLogger("oos-direct")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tickers = CORE_TICKERS + SECTOR_TICKERS
    ohlcv = fetch_history(tickers, 60, interval="5m")
    all_dates = sorted({d for df in ohlcv.values() for d in set(df.index.date)})
    n = len(all_dates)
    start = min(df.index[0] for df in ohlcv.values())
    end = max(df.index[-1] for df in ohlcv.values())
    vix = fetch_daily_vix(start, end + pd.Timedelta(days=1))

    # The two "winner" configs from full-60d
    configs = {
        "C1: ATR>=0.0025, TP=1.5%, SL=1.0%": dict(
            min_atr_pct=0.0025, min_tp_pct=0.015, min_sl_pct=0.010),
        "C2: ATR>=0.0025, TP=2.0%, SL=1.0%": dict(
            min_atr_pct=0.0025, min_tp_pct=0.020, min_sl_pct=0.010),
        "C3: ATR>=0.0025, TP=1.2%, SL=0.8%": dict(
            min_atr_pct=0.0025, min_tp_pct=0.012, min_sl_pct=0.008),
    }
    base = dict(
        min_bars=22,
        tp_mult=2.0, tp_mult_hv=2.5, sl_mult=1.0, sl_mult_hv=1.5,
        time_stop_bars_before_close=0,
        reverse=True,
        min_vix=0.0,
    )

    for split_ratio in [0.5, 0.6, 0.4]:
        split_idx = int(n * split_ratio)
        is_dates = set(all_dates[:split_idx])
        oos_dates = set(all_dates[split_idx:])
        print(f"\n{'='*80}\nSPLIT {split_ratio*100:.0f}/{100-split_ratio*100:.0f}  "
              f"IS={len(is_dates)}d  OOS={len(oos_dates)}d")
        print("=" * 80)
        for name, extra in configs.items():
            params = {**base, **extra}
            is_stats = run_window(ohlcv, vix, is_dates, params)
            oos_stats = run_window(ohlcv, vix, oos_dates, params)

            def fmt(s):
                if s.get("n", 0) == 0:
                    return "n=0  -- no trades"
                return (f"n={s['n']:3d}  WR={s['winrate']:.0%}  "
                        f"gross={s['expectancy_gross']*100:+.3f}%  "
                        f"net={s['expectancy_net']*100:+.3f}%")

            print(f"\n{name}")
            print(f"   IS  {fmt(is_stats)}")
            print(f"   OOS {fmt(oos_stats)}")


if __name__ == "__main__":
    main()
