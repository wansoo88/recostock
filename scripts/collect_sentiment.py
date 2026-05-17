"""Daily sentiment collection entrypoint.

Runs from GitHub Actions cron (13:00 UTC, Mon–Fri) — before nightly retrain
(14:00 UTC) so that the next retrain *could* read sentiment features once
they are wired into the model. Until then, this only writes the parquet.

Pipeline:
  1. Yahoo Finance per-ticker RSS  (17 ETFs)
  2. HackerNews Algolia            (each ticker + ETF name alias)
  3. Aggregate ticker mention counts → upsert data/raw/sentiment_daily.parquet

Failure policy:
  - One source down → continue with the other; mark missing source absent.
  - Both down → exit 1, no parquet change.

Flags:
  --dry-run   skip parquet writes (still hits network)
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Repo root on PYTHONPATH (mirrors run_daily.py / nightly_retrain.py).
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from sentiment.aggregator import aggregate, upsert_parquet, PARQUET_PATH
from sentiment.sources import yahoo_rss, hackernews
from sentiment.ticker_extract import TRACKED_TICKERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("collect_sentiment")


# HackerNews queries — ticker cashtag is too noisy on HN; use plain ticker
# plus the most recognisable ETF alias. Keep queries narrow to avoid swamp.
HN_QUERIES = list(TRACKED_TICKERS) + [
    "S&P 500 ETF", "Nasdaq 100 ETF", "biotech ETF", "VIX futures",
]


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    today_utc = datetime.now(timezone.utc).date()
    log.info("Collecting sentiment for %s (dry_run=%s)", today_utc, dry_run)

    all_docs: list[dict] = []
    source_ok: dict[str, bool] = {}

    try:
        docs = yahoo_rss.fetch(TRACKED_TICKERS)
        all_docs.extend(docs)
        source_ok["yahoo_rss"] = bool(docs)
    except Exception as exc:
        log.exception("yahoo_rss source crashed: %s", exc)
        source_ok["yahoo_rss"] = False

    try:
        docs = hackernews.fetch(HN_QUERIES)
        all_docs.extend(docs)
        source_ok["hackernews"] = bool(docs)
    except Exception as exc:
        log.exception("hackernews source crashed: %s", exc)
        source_ok["hackernews"] = False

    log.info("Source health: %s", source_ok)
    if not any(source_ok.values()):
        log.error("All sources failed — no parquet change")
        return 1

    df = aggregate(all_docs, today=today_utc)
    if df.empty:
        log.warning("No matching ticker mentions in %d docs — writing empty marker", len(all_docs))

    if dry_run:
        log.info("[--dry-run] rows that would be written:\n%s",
                 df.head(40).to_string(index=False) if not df.empty else "<none>")
        return 0

    upsert_parquet(df, PARQUET_PATH)

    # Quick summary
    if not df.empty:
        top = (df.groupby("ticker")["mention_count"].sum()
                 .sort_values(ascending=False).head(8))
        log.info("Top tickers today:\n%s", top.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
