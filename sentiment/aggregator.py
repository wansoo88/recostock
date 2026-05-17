"""Aggregate raw documents → daily (date, ticker, source, mention_count) rows.

Why per-source rows instead of one combined count?
  - Lets us check IC per source separately before mixing.
  - Keeps the parquet long-format trivially queryable & extensible (adding
    Reddit or EDGAR later doesn't change the schema).
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

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


def aggregate(docs: list[dict], today: date | None = None) -> pd.DataFrame:
    """Return long-format DataFrame labeled with the run's `today` (UTC date).

    Documents published within the last LOOKBACK_DAYS are all counted into
    `today` — daily collection snapshots the recent mention volume rather
    than partitioning by article date. This is the form the model will
    consume (a per-day sentiment intensity feature)."""
    if today is None:
        today = datetime.now(timezone.utc).date()
    earliest = today - timedelta(days=LOOKBACK_DAYS)

    rows: list[dict] = []
    # Per (source, ticker) counter so a single article doesn't double-count.
    counter: dict[tuple[str, str], int] = {}
    seen_keys: set[tuple[str, str, str]] = set()  # de-dup per source+title+ticker

    for d in docs:
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
            counter[(source, t)] = counter.get((source, t), 0) + 1

    for (source, ticker), cnt in counter.items():
        rows.append({
            "date": today,
            "ticker": ticker,
            "source": source,
            "mention_count": cnt,
        })

    df = pd.DataFrame(rows, columns=["date", "ticker", "source", "mention_count"])
    if df.empty:
        log.warning("Aggregator produced 0 rows for %s — collection may have failed", today)
    else:
        log.info("Aggregated %d (source, ticker) rows for %s", len(df), today)
    return df


def upsert_parquet(df: pd.DataFrame, path: Path = PARQUET_PATH) -> Path:
    """Merge `df` into the cumulative parquet, replacing same-day rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        # Drop any prior row for the same (date, ticker, source) — today wins.
        merge_keys = ["date", "ticker", "source"]
        merged = pd.concat(
            [existing, df], ignore_index=True
        ).drop_duplicates(subset=merge_keys, keep="last")
    else:
        merged = df
    merged["date"] = pd.to_datetime(merged["date"]).dt.date
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
    }])
