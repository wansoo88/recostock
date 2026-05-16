#!/usr/bin/env python3
"""5-minute IC analysis — gate test before LightGBM.

Computes Spearman IC between each intraday factor at bar t and the forward
log-return over horizon h bars. Uses time-series IC per ticker, then averages
across tickers (different from daily cross-sectional IC: with 9 tickers,
cross-sectional samples are too small per bar). Reports mean IC, t-stat,
and KEEP/REJECT verdict against config.IC_MIN_VIABLE = 0.01.

Why this matters: if no factor has |IC| > 0.01 at any horizon, LightGBM has
no usable raw material and will overfit. Cheap gate before committing to ML.

Usage:
    python scripts/run_intraday_ic.py [--days 60] [--horizons 6,12,30]
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.intraday import CORE_TICKERS, SECTOR_TICKERS
from features.intraday_factors import compute_intraday_features
from features.ic_analysis import _two_sided_pvalue
from scripts.run_intraday_backtest import fetch_history

log = logging.getLogger("ic")

FACTOR_COLS = [
    "ema5", "ema20",
    "vwap_dev_sd",
    "rsi14", "stochrsi_k",
    "adx14",
    "obv_slope",
    "vol_ratio",
]
# Also derived signed factors: ema_spread, price_vs_vwap
DERIVED = ["ema_spread_pct", "price_vs_vwap_pct"]


def build_factor_table(ohlcv: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """For each ticker, return a DataFrame of factor columns (within-day, NaN across day boundaries)."""
    out: dict[str, pd.DataFrame] = {}
    for ticker, df in ohlcv.items():
        rows = []
        for date_, day_df in df.groupby(df.index.date):
            if len(day_df) < 25:
                continue
            feat = compute_intraday_features(day_df)
            feat["ema_spread_pct"] = (feat["ema5"] - feat["ema20"]) / feat["ema20"]
            feat["price_vs_vwap_pct"] = (feat["Close"] - feat["vwap"]) / feat["vwap"]
            keep_cols = ["Close"] + FACTOR_COLS + DERIVED
            rows.append(feat[keep_cols].copy())
        if rows:
            out[ticker] = pd.concat(rows, axis=0).sort_index()
    return out


def forward_log_return(close: pd.Series, horizon_bars: int) -> pd.Series:
    """Forward log return over horizon, intra-session only.

    We compute via shift(-h) on the full series, then NaN out crossings of date
    boundaries so we never use overnight returns as if they were intraday.
    """
    fwd = np.log(close.shift(-horizon_bars) / close)
    # Mask cross-day crossings
    same_day = (
        pd.Series(close.index.date, index=close.index)
        == pd.Series(close.index.date, index=close.index).shift(-horizon_bars)
    )
    return fwd.where(same_day)


def compute_ic_for_factor(factor_series: pd.Series, fwd_series: pd.Series) -> float:
    """Spearman correlation between factor at t and forward return over [t, t+h]."""
    s = pd.concat([factor_series.rename("f"), fwd_series.rename("r")], axis=1).dropna()
    if len(s) < 100:
        return float("nan")
    return float(s["f"].corr(s["r"], method="spearman"))


def summarize_ic(ics: list[float]) -> dict:
    """Aggregate per-ticker ICs into a single verdict."""
    arr = np.array([x for x in ics if not math.isnan(x)])
    if len(arr) < 3:
        return {"mean_ic": float("nan"), "std_ic": float("nan"),
                "t_stat": float("nan"), "p_value": float("nan"),
                "n_tickers": len(arr), "verdict": "INSUFFICIENT"}
    mean_ic = float(arr.mean())
    std_ic = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    t_stat = mean_ic / (std_ic / math.sqrt(len(arr))) if std_ic > 0 else 0.0
    p_value = _two_sided_pvalue(t_stat)
    viable = abs(mean_ic) >= config.IC_MIN_VIABLE and p_value < 0.05
    return {
        "mean_ic": round(mean_ic, 4),
        "std_ic": round(std_ic, 4),
        "t_stat": round(t_stat, 2),
        "p_value": round(p_value, 4),
        "n_tickers": len(arr),
        "verdict": "KEEP" if viable else "REJECT",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--horizons", type=str, default="6,12,30",
                    help="forward-return horizons in 5min bars (6=30min,12=60min,30=2.5h)")
    ap.add_argument("--out", type=str, default="data/intraday_ic_results.csv")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    horizons = [int(x) for x in args.horizons.split(",")]

    tickers = CORE_TICKERS + SECTOR_TICKERS
    log.info("fetching %dd of 5m bars: %s", args.days, tickers)
    ohlcv = fetch_history(tickers, args.days, interval="5m")
    if not ohlcv:
        log.error("no data")
        sys.exit(1)

    log.info("building factor tables")
    factor_tables = build_factor_table(ohlcv)
    log.info("computing IC across %d tickers x %d factors x %d horizons",
             len(factor_tables), len(FACTOR_COLS + DERIVED), len(horizons))

    rows = []
    for factor_name in FACTOR_COLS + DERIVED:
        for h in horizons:
            per_ticker_ic = []
            for ticker, table in factor_tables.items():
                fwd = forward_log_return(table["Close"], h)
                ic = compute_ic_for_factor(table[factor_name], fwd)
                per_ticker_ic.append(ic)
            agg = summarize_ic(per_ticker_ic)
            rows.append({
                "factor": factor_name,
                "horizon_bars": h,
                "horizon_min": h * 5,
                **agg,
            })

    df = pd.DataFrame(rows)
    df["abs_ic"] = df["mean_ic"].abs()
    df = df.sort_values("abs_ic", ascending=False).drop(columns="abs_ic").reset_index(drop=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    out = "\n".join([
        "=" * 90,
        f"INTRADAY IC ANALYSIS -- {args.days}d, 5m bars",
        f"viable threshold |IC| >= {config.IC_MIN_VIABLE}",
        "=" * 90,
        df.to_string(index=False),
        "",
        f"KEEP factors: {(df['verdict']=='KEEP').sum()}  /  {len(df)} total tests",
    ])
    Path(args.out.replace(".csv", ".txt")).write_text(out, encoding="utf-8")
    try:
        print(out)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
