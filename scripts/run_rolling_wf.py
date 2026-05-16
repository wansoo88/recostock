#!/usr/bin/env python3
"""Rolling walk-forward + per-ticker stability analysis for vol-gated reverse strategy.

Goals:
  1. Apply 3 fixed candidate configs across 5 expanding-window folds.
     No grid search per fold -> no data snooping in evaluation.
  2. For each config: report fold-by-fold OOS net EV, mean, std,
     fraction of folds positive.
  3. Per-ticker decomposition: which ETFs contribute the alpha consistently?

Honest interpretation:
  - 4/5 or 5/5 positive folds with consistent magnitude → real edge
  - 3/5 with mixed sign → ambiguous; needs more data
  - 2/5 or less → not robust
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
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

log = logging.getLogger("rolling-wf")


CONFIGS = {
    "C1: TP=1.5/SL=1.0": dict(min_atr_pct=0.0025, min_tp_pct=0.015, min_sl_pct=0.010),
    "C2: TP=2.0/SL=1.0": dict(min_atr_pct=0.0025, min_tp_pct=0.020, min_sl_pct=0.010),
    "C3: TP=1.2/SL=0.8": dict(min_atr_pct=0.0025, min_tp_pct=0.012, min_sl_pct=0.008),
}

BASE_PARAMS = dict(
    min_bars=22,
    tp_mult=2.0, tp_mult_hv=2.5, sl_mult=1.0, sl_mult_hv=1.5,
    time_stop_bars_before_close=0,
    reverse=True,
    min_vix=0.0,
)


def run_with_per_ticker(ohlcv, vix, sessions, params):
    """Like run_window but returns per-ticker trade breakdown too."""
    from dataclasses import asdict
    all_trades = []
    for ticker, df in ohlcv.items():
        allow_short = ticker in CORE_TICKERS
        for date_, day_df in df.groupby(df.index.date):
            if date_ not in sessions:
                continue
            if len(day_df) < params["min_bars"] + 2:
                continue
            prior = vix.index[vix.index < date_]
            vix_level = float(vix.loc[prior[-1]]) if len(prior) else None
            if vix_level is not None and vix_level >= 30 and ticker in SECTOR_TICKERS:
                continue
            trades = simulate_day(ticker, day_df, vix_level, allow_short, **params)
            all_trades.extend(trades)
    stats = summarize(all_trades)
    if not all_trades:
        return stats, pd.DataFrame()
    df_trades = pd.DataFrame([asdict(t) for t in all_trades])
    return stats, df_trades


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tickers = CORE_TICKERS + SECTOR_TICKERS
    ohlcv = fetch_history(tickers, 60, interval="5m")
    all_dates = sorted({d for df in ohlcv.values() for d in set(df.index.date)})
    n = len(all_dates)
    log.info("dates: %d (%s to %s)", n, all_dates[0], all_dates[-1])

    start = min(df.index[0] for df in ohlcv.values())
    end = max(df.index[-1] for df in ohlcv.values())
    vix = fetch_daily_vix(start, end + pd.Timedelta(days=1))

    # ── 5-fold expanding window ────────────────────────────────────────────────
    n_folds = 5
    min_train = max(15, n // 4)
    oos_size = max(5, (n - min_train) // n_folds)
    log.info("WF: min_train=%d, oos_size=%d, n_folds<=%d", min_train, oos_size, n_folds)

    fold_specs = []
    for k in range(n_folds):
        train_end = min_train + k * oos_size
        if train_end >= n:
            break
        test_end = min(train_end + oos_size, n)
        is_dates = set(all_dates[:train_end])
        oos_dates = set(all_dates[train_end:test_end])
        fold_specs.append({
            "fold": k + 1,
            "is_end": all_dates[train_end - 1],
            "oos_start": all_dates[train_end],
            "oos_end": all_dates[test_end - 1],
            "is_dates": is_dates,
            "oos_dates": oos_dates,
        })

    # Collect results per config
    per_config_results = {name: [] for name in CONFIGS}
    per_ticker_oos = {name: {t: [] for t in tickers} for name in CONFIGS}

    for f in fold_specs:
        log.info("fold %d: IS<=%s, OOS %s..%s (%d sessions)",
                 f["fold"], f["is_end"], f["oos_start"], f["oos_end"], len(f["oos_dates"]))
        for name, extra in CONFIGS.items():
            params = {**BASE_PARAMS, **extra}
            oos_stats, oos_trades = run_with_per_ticker(ohlcv, vix, f["oos_dates"], params)
            per_config_results[name].append({
                "fold": f["fold"],
                "oos_end": str(f["oos_end"]),
                "n": oos_stats.get("n", 0),
                "winrate": oos_stats.get("winrate", None),
                "gross": oos_stats.get("expectancy_gross", None),
                "net": oos_stats.get("expectancy_net", None),
            })
            if not oos_trades.empty:
                for t in tickers:
                    sub = oos_trades[oos_trades["ticker"] == t]
                    per_ticker_oos[name][t].append(sub["net_pnl"].sum() if not sub.empty else 0.0)
            else:
                for t in tickers:
                    per_ticker_oos[name][t].append(0.0)

    # ── Report ────────────────────────────────────────────────────────────────
    lines = ["=" * 100,
             "ROLLING WALK-FORWARD (5-fold expanding) -- vol-gated reverse, ATR>=0.0025",
             "=" * 100]

    for name, rows in per_config_results.items():
        rdf = pd.DataFrame(rows)
        nets = [r for r in rdf["net"] if r is not None]
        grosses = [r for r in rdf["gross"] if r is not None]
        pos_folds = sum(1 for x in nets if x > 0)

        lines += [
            "",
            f"--- {name} ---",
            rdf.to_string(index=False),
            "",
            f"Folds: {len(nets)}  |  positive: {pos_folds}/{len(nets)}",
        ]
        if nets:
            arr_net = np.array(nets) * 100
            arr_gross = np.array(grosses) * 100
            lines += [
                f"OOS net  EV (%/trade):  mean={arr_net.mean():+.3f}  std={arr_net.std():.3f}  "
                f"min={arr_net.min():+.3f}  max={arr_net.max():+.3f}",
                f"OOS gross EV (%/trade):  mean={arr_gross.mean():+.3f}  std={arr_gross.std():.3f}",
            ]
            # t-stat on net EV across folds
            if len(nets) >= 2 and arr_net.std() > 0:
                t = arr_net.mean() / (arr_net.std() / np.sqrt(len(arr_net)))
                lines += [f"t-stat (mean/SE):  {t:+.2f}  (|t|>2 = robust)"]

    # Per-ticker contribution
    lines += ["", "=" * 100, "PER-TICKER OOS NET PNL SUM ACROSS FOLDS", "=" * 100]
    for name in CONFIGS:
        lines += [f"\n--- {name} ---"]
        rows = []
        for t in tickers:
            sums = per_ticker_oos[name][t]
            tot = sum(sums)
            pos = sum(1 for x in sums if x > 0)
            rows.append({"ticker": t, "total_pnl_pct": round(tot * 100, 3),
                         "pos_folds": f"{pos}/{len(sums)}"})
        lines.append(pd.DataFrame(rows).to_string(index=False))

    out = "\n".join(lines)
    Path("data/intraday_rolling_wf.txt").write_text(out, encoding="utf-8")
    try:
        print(out)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
