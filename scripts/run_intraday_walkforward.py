#!/usr/bin/env python3
"""Walk-forward validation of vol-gated reverse strategy.

Splits the 60-day window into in-sample (first half) and out-of-sample (second).
1. On IS, sweep parameters around the candidate winner (ATR=0.25%, TP=1.5%, SL=1.0%).
2. Pick best by IS net EV.
3. Apply that exact config to OOS.
4. Report IS vs OOS EV — large drop means overfit.

If OOS net EV stays positive (or close), the signal has real (if small) alpha.
If OOS collapses to negative, the IS result was overfit to the specific 60-day window.
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

log = logging.getLogger("wf")


def run_window(ohlcv: dict, vix: pd.Series, sessions: set, params: dict) -> dict:
    """Run backtest restricted to specific sessions."""
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
            trades = simulate_day(
                ticker, day_df, vix_level, allow_short, **params
            )
            all_trades.extend(trades)
    return summarize(all_trades)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--split-ratio", type=float, default=0.5,
                    help="fraction of sessions used as in-sample (0.5 = first half)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tickers = CORE_TICKERS + SECTOR_TICKERS
    log.info("fetching %dd of 5m bars", args.days)
    ohlcv = fetch_history(tickers, args.days, interval="5m")
    if not ohlcv:
        sys.exit(1)

    all_dates = sorted({d for df in ohlcv.values() for d in set(df.index.date)})
    n = len(all_dates)
    split_idx = int(n * args.split_ratio)
    is_sessions = set(all_dates[:split_idx])
    oos_sessions = set(all_dates[split_idx:])
    log.info("IS: %d sessions (%s - %s)  OOS: %d sessions (%s - %s)",
             len(is_sessions), all_dates[0], all_dates[split_idx - 1],
             len(oos_sessions), all_dates[split_idx], all_dates[-1])

    start = min(df.index[0] for df in ohlcv.values())
    end = max(df.index[-1] for df in ohlcv.values())
    vix = fetch_daily_vix(start, end + pd.Timedelta(days=1))

    base_params = dict(
        min_bars=22,
        tp_mult=2.0, tp_mult_hv=2.5, sl_mult=1.0, sl_mult_hv=1.5,
        time_stop_bars_before_close=0,
        reverse=True,
        min_vix=0.0,
    )

    # ── IS sweep around the candidate winner ─────────────────────────────────
    atr_grid = [0.0020, 0.0022, 0.0025, 0.0028, 0.0030]
    tp_grid  = [0.010, 0.012, 0.015, 0.018, 0.020]
    sl_grid  = [0.007, 0.008, 0.010, 0.012]

    log.info("IS sweep: %d configs", len(atr_grid) * len(tp_grid) * len(sl_grid))
    is_rows = []
    for atr, tp, sl in itertools.product(atr_grid, tp_grid, sl_grid):
        params = {**base_params, "min_atr_pct": atr, "min_tp_pct": tp, "min_sl_pct": sl}
        stats = run_window(ohlcv, vix, is_sessions, params)
        if stats.get("n", 0) == 0:
            continue
        is_rows.append({
            "min_atr_pct": atr, "min_tp_pct": tp, "min_sl_pct": sl,
            "n": stats["n"],
            "winrate": stats["winrate"],
            "ev_gross": stats["expectancy_gross"],
            "ev_net": stats["expectancy_net"],
            "payoff": stats["payoff"],
            "tp_hits": stats["by_reason"].get("TP", 0),
            "sl_hits": stats["by_reason"].get("SL", 0),
        })

    is_df = pd.DataFrame(is_rows).sort_values("ev_net", ascending=False).reset_index(drop=True)
    log.info("IS top 5 configs:")
    log.info("\n%s", is_df.head(5).to_string(index=False))

    # ── Pick best (require min n=10 to avoid tiny-sample winners) ─────────────
    is_qualified = is_df[is_df["n"] >= 10]
    if is_qualified.empty:
        log.error("no IS config with n>=10")
        sys.exit(1)
    best = is_qualified.iloc[0]
    log.info("PICKED: atr=%.4f tp=%.4f sl=%.4f IS net=%.4f%%",
             best["min_atr_pct"], best["min_tp_pct"], best["min_sl_pct"], best["ev_net"] * 100)

    # ── OOS application of picked config ────────────────────────────────────
    best_params = {
        **base_params,
        "min_atr_pct": float(best["min_atr_pct"]),
        "min_tp_pct": float(best["min_tp_pct"]),
        "min_sl_pct": float(best["min_sl_pct"]),
    }
    oos_stats = run_window(ohlcv, vix, oos_sessions, best_params)

    # ── Report ──────────────────────────────────────────────────────────────
    lines = [
        "=" * 80,
        f"WALK-FORWARD VALIDATION  --  IS {len(is_sessions)}d / OOS {len(oos_sessions)}d",
        "=" * 80,
        f"Selected (IS-best, n>=10):  ATR>={best['min_atr_pct']:.4f}  "
        f"TP={best['min_tp_pct']:.4f}  SL={best['min_sl_pct']:.4f}",
        "",
        f"IS  results (n={best['n']:.0f}):",
        f"   winrate {best['winrate']:.2%}   gross {best['ev_gross']*100:+.3f}%   "
        f"net {best['ev_net']*100:+.3f}%   payoff {best['payoff']:.2f}",
        f"   TP={best['tp_hits']:.0f}  SL={best['sl_hits']:.0f}",
        "",
        f"OOS results (n={oos_stats.get('n', 0)}):",
    ]
    if oos_stats.get("n", 0) > 0:
        lines += [
            f"   winrate {oos_stats['winrate']:.2%}   "
            f"gross {oos_stats['expectancy_gross']*100:+.3f}%   "
            f"net {oos_stats['expectancy_net']*100:+.3f}%   "
            f"payoff {oos_stats['payoff']:.2f}",
            f"   by_reason: {oos_stats['by_reason']}",
            "",
            f"DEGRADATION:  IS gross {best['ev_gross']*100:+.3f}% -> OOS gross "
            f"{oos_stats['expectancy_gross']*100:+.3f}%  "
            f"(delta {(oos_stats['expectancy_gross'] - best['ev_gross'])*100:+.3f}%)",
        ]
    else:
        lines += ["   (no OOS trades)"]

    out = "\n".join(lines)
    Path("data/intraday_walkforward.txt").write_text(out, encoding="utf-8")
    is_df.to_csv("data/intraday_walkforward_is.csv", index=False)
    try:
        print(out)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
