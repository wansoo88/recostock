"""Aggregate raw documents → daily long-format sentiment rows.

Output schema:
    date | ticker | source | mention_count | polarity_mean | polarity_n

- mention_count: # of distinct articles per (date, ticker, source).
- polarity_mean: mean FinBERT polarity (pos - neg) ∈ [-1, +1] over the
  articles that were successfully scored. NaN when polarity_n == 0.
- polarity_n:    # of articles whose text was actually scored (≤ mention_count).
                 0 when the FinBERT scorer is unavailable (no transformers
                 installed locally) or the doc text was empty.

Per-source rows (not one combined count) so we can:
  - Inspect IC per source before mixing.
  - Add sources later without changing schema.
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from sentiment.ticker_extract import extract_tickers, TRACKED_TICKERS

log = logging.getLogger(__name__)

PARQUET_PATH = Path("data/sentiment/sentiment_daily.parquet")

# Accept articles published within this many days of `today` (UTC).
# Reason: collector fires at 13:00 UTC; at that moment most US-market news
# from the *prior* trading session sits at T-1. A 2-day window is the
# smallest that reliably captures both same-day morning items and the
# previous afternoon's US close coverage.
LOOKBACK_DAYS = 2


def aggregate(docs: list[dict], today: date | None = None,
              polarities: Sequence[float | None] | None = None) -> pd.DataFrame:
    """Long-format DataFrame labeled with the run's `today` (UTC date).

    Documents published within the last LOOKBACK_DAYS are all counted into
    `today` — daily collection snapshots the recent mention volume.

    `polarities[i]` is the FinBERT polarity for `docs[i]`, or None if not
    scored. Each ticker matched in a doc inherits that doc's polarity (one
    article about XLE and XLF gives both groups the same score)."""
    if today is None:
        today = datetime.now(timezone.utc).date()
    if polarities is None:
        polarities = [None] * len(docs)
    elif len(polarities) != len(docs):
        raise ValueError(f"polarities length {len(polarities)} != docs length {len(docs)}")
    earliest = today - timedelta(days=LOOKBACK_DAYS)

    counter: dict[tuple[str, str], int] = {}
    polarity_sum: dict[tuple[str, str], float] = {}
    polarity_n: dict[tuple[str, str], int] = {}
    seen_keys: set[tuple[str, str, str]] = set()  # de-dup per source+title+ticker

    for d, pol in zip(docs, polarities):
        pub = d.get("published")
        if pub is None:
            continue
        pub_date = pub.astimezone(timezone.utc).date()
        if pub_date < earliest or pub_date > today:
            continue
        source = d.get("source", "unknown")
        text = " ".join(filter(None, [d.get("title"), d.get("body")]))
        if not text:
            continue
        title_key = (d.get("title") or "")[:200]
        for t in extract_tickers(text):
            dedup = (source, title_key, t)
            if dedup in seen_keys:
                continue
            seen_keys.add(dedup)
            key = (source, t)
            counter[key] = counter.get(key, 0) + 1
            if pol is not None and not math.isnan(pol):
                polarity_sum[key] = polarity_sum.get(key, 0.0) + float(pol)
                polarity_n[key] = polarity_n.get(key, 0) + 1

    rows: list[dict] = []
    for (source, ticker), cnt in counter.items():
        n = polarity_n.get((source, ticker), 0)
        mean = (polarity_sum[(source, ticker)] / n) if n > 0 else float("nan")
        rows.append({
            "date": today,
            "ticker": ticker,
            "source": source,
            "mention_count": cnt,
            "polarity_mean": mean,
            "polarity_n": n,
        })

    df = pd.DataFrame(
        rows,
        columns=["date", "ticker", "source", "mention_count", "polarity_mean", "polarity_n"],
    )
    if df.empty:
        log.warning("Aggregator produced 0 rows for %s — collection may have failed", today)
    else:
        scored_share = (df["polarity_n"].sum() / df["mention_count"].sum()) if df["mention_count"].sum() else 0.0
        log.info("Aggregated %d (source, ticker) rows for %s — %.0f%% of mentions scored",
                 len(df), today, scored_share * 100)
    return df


def upsert_parquet(df: pd.DataFrame, path: Path = PARQUET_PATH) -> Path:
    """Merge `df` into the cumulative parquet, replacing same-day rows.

    Schema-evolves an older parquet that lacks polarity_mean / polarity_n
    by filling NaN / 0 — this is what older daily commits look like."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        if "polarity_mean" not in existing.columns:
            existing["polarity_mean"] = np.nan
        if "polarity_n" not in existing.columns:
            existing["polarity_n"] = 0
        merge_keys = ["date", "ticker", "source"]
        merged = pd.concat(
            [existing, df], ignore_index=True
        ).drop_duplicates(subset=merge_keys, keep="last")
    else:
        merged = df
    merged["date"] = pd.to_datetime(merged["date"]).dt.date
    merged["polarity_n"] = merged["polarity_n"].fillna(0).astype("int64")
    merged = merged.sort_values(["date", "ticker", "source"]).reset_index(drop=True)
    merged.to_parquet(path, index=False)
    log.info("Upserted parquet → %s  (rows=%d)", path, len(merged))
    return path


def empty_day_row(today: date) -> pd.DataFrame:
    """Sentinel row so that gaps (e.g., source down) are explicit in the parquet."""
    return pd.DataFrame([{
        "date": today,
        "ticker": TRACKED_TICKERS[0],
        "source": "__empty__",
        "mention_count": 0,
        "polarity_mean": float("nan"),
        "polarity_n": 0,
    }])
