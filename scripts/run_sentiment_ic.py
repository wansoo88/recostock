#!/usr/bin/env python3
"""Per-source IC for sentiment features.

Run when sentiment_daily.parquet has accumulated ≥ 20 trading days. Earlier
runs report "INSUFFICIENT" — that's expected and not a failure.

Tests three candidate features per source:
    polarity_mean                — average FinBERT/override polarity per day
    mention_count_log            — log(1 + mention_count), retail attention proxy
    polarity_mention_product     — polarity_mean × log(1+mention_count),
                                   the volume-weighted version

Per-source so that, e.g., StockTwits Bull/Bear tags are scored separately
from Yahoo RSS headline sentiment — different signal generators should not
average each other away.

Output:
    Console table per (source, feature, horizon)
    data/logs/sentiment_ic_report.csv

Rejection rule mirrors features/ic_analysis.py: |mean IC| < IC_MIN_VIABLE (0.01).
"""
from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

import config
from features.ic_analysis import compute_forward_returns, compute_ic_series, ic_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
log = logging.getLogger(__name__)

SENTIMENT_PATH = Path("data/sentiment/sentiment_daily.parquet")
OHLCV_PATH = Path("data/raw/etf_ohlcv.parquet")
OUTPUT_PATH = Path("data/logs/sentiment_ic_report.csv")
HORIZONS = [1, 5]
MIN_TRADING_DAYS = 20  # IC's normal-approx p-value needs n > ~30; 20 is the floor for any signal


def _load_close() -> pd.DataFrame:
    if not OHLCV_PATH.exists():
        log.error("OHLCV missing: %s — run `python -m data.collector` first", OHLCV_PATH)
        sys.exit(1)
    ohlcv = pd.read_parquet(OHLCV_PATH)
    close = ohlcv["Close"] if isinstance(ohlcv.columns, pd.MultiIndex) else ohlcv
    return close


def _load_sentiment() -> pd.DataFrame:
    if not SENTIMENT_PATH.exists():
        log.error("Sentiment parquet missing: %s — run scripts/collect_sentiment.py first", SENTIMENT_PATH)
        sys.exit(1)
    df = pd.read_parquet(SENTIMENT_PATH)
    df = df[df["source"] != "__empty__"].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


def _build_features(sentiment: pd.DataFrame) -> dict[str, pd.Series]:
    """Return {feature_name: Series indexed by (date, ticker)} per source.

    Key format: f"{source}:{feature_name}" so the IC loop iterates flat.
    Sentiment is daily but markets are business days only — the inner join
    with returns drops weekends automatically.
    """
    sentiment["mention_count_log"] = np.log1p(sentiment["mention_count"])
    sentiment["polarity_mention_product"] = (
        sentiment["polarity_mean"].fillna(0.0) * sentiment["mention_count_log"]
    )

    feats: dict[str, pd.Series] = {}
    for source, grp in sentiment.groupby("source"):
        idx = pd.MultiIndex.from_arrays([grp["date"], grp["ticker"]], names=["date", "ticker"])
        for col in ("polarity_mean", "mention_count_log", "polarity_mention_product"):
            series = pd.Series(grp[col].values, index=idx, name=f"{source}:{col}")
            # Keep NaN polarity_mean values out — they represent unscored days,
            # not a "zero" signal. The IC fn drops them via dropna.
            feats[f"{source}:{col}"] = series
    return feats


def main() -> int:
    sentiment = _load_sentiment()
    n_days = sentiment["date"].nunique()
    log.info("Sentiment data: %d rows across %d days [%s..%s]",
             len(sentiment), n_days,
             sentiment["date"].min().date(), sentiment["date"].max().date())

    if n_days < MIN_TRADING_DAYS:
        log.warning("Only %d days of sentiment data — need ≥ %d for IC. "
                    "Exiting with INSUFFICIENT verdict (not a failure).",
                    n_days, MIN_TRADING_DAYS)
        return 0

    close = _load_close()
    fwd_returns = compute_forward_returns(close, horizons=HORIZONS)
    # Convert wide forward-return frames to (date, ticker) MultiIndex Series.
    fwd_long = {h: r.stack(future_stack=True).rename("r") for h, r in fwd_returns.items()}

    features = _build_features(sentiment)
    log.info("Computing IC for %d (source × feature) combinations × %d horizons",
             len(features), len(HORIZONS))

    rows: list[dict] = []
    for fname, fseries in features.items():
        source, feat = fname.split(":", 1)
        for h, r in fwd_long.items():
            ic_series = compute_ic_series(fseries, r)
            summary = ic_summary(ic_series, factor_name=fname, horizon=h)
            summary["source"] = source
            summary["feature"] = feat
            rows.append(summary)

    report = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 90)
    print("  SENTIMENT — PER-SOURCE IC REPORT")
    print(f"  Rejection threshold: |mean IC| < {config.IC_MIN_VIABLE}")
    print("=" * 90)
    for h in HORIZONS:
        sub = report[report["horizon"] == h]
        print(f"\n── {h}-DAY FORWARD ──")
        print(f"{'Source':<12} {'Feature':<28} {'MeanIC':>8} {'ICIR':>7} {'p-val':>7} {'N':>5}  Verdict")
        print("-" * 80)
        for _, row in sub.iterrows():
            mark = "✅" if row["verdict"] == "KEEP" else ("⏸" if row["verdict"] == "INSUFFICIENT" else "❌")
            print(f"{row['source']:<12} {row['feature']:<28} "
                  f"{row['mean_ic']:>8.4f} {row['icir']:>7.3f} "
                  f"{row['p_value']:>7.4f} {row['n_obs']:>5}  {mark} {row['verdict']}")

    keeps = report[report["verdict"] == "KEEP"]
    print("\n" + "=" * 90)
    print(f"KEEP   ({len(keeps)}):")
    for _, row in keeps.iterrows():
        print(f"  - {row['source']}:{row['feature']} @ {row['horizon']}d  IC={row['mean_ic']:+.4f}")
    print(f"\nReport saved → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
