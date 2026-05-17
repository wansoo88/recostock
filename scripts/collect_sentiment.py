"""Daily sentiment collection entrypoint.

Runs from GitHub Actions cron (13:00 UTC, Mon–Fri) — before nightly retrain
(14:00 UTC) so that the next retrain *could* read sentiment features once
they are wired into the model. Until then, this only writes the parquet.

Pipeline:
  1. Yahoo Finance per-ticker RSS  (17 ETFs)
  2. HackerNews Algolia            (each ticker + ETF name alias)
  3. SEC EDGAR full-text search    (formal ETF name → 8-K / 10-Q / N-PORT / 485BPOS)
  4. StockTwits per-symbol stream  (retail trader sentiment, Bull/Bear-tagged)
  5. Aggregate ticker mention counts → upsert data/sentiment/sentiment_daily.parquet

Failure policy:
  - One source down → continue with the other; mark missing source absent.
  - All sources down → exit 1, no parquet change.

FinBERT polarity:
  - Computed if transformers + torch are installed (workflow installs them).
  - Local runs without those packages just write polarity_n=0 and the
    aggregator carries on. Forecast pipeline tolerates either.

Flags:
  --dry-run   skip parquet writes (still hits network).
  --no-score  skip FinBERT even if available (used to test schema locally).
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
from sentiment.sources import yahoo_rss, hackernews, edgar, stocktwits
from sentiment.ticker_extract import TRACKED_TICKERS
from sentiment import scorer

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
    no_score = "--no-score" in sys.argv
    today_utc = datetime.now(timezone.utc).date()
    log.info("Collecting sentiment for %s (dry_run=%s, no_score=%s)",
             today_utc, dry_run, no_score)

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

    try:
        # Fetch 5 days so the aggregator's 2-day publish window has slack and
        # late-filed forms still get attributed to the correct day.
        docs = edgar.fetch(lookback_days=5)
        all_docs.extend(docs)
        # EDGAR can legitimately return 0 hits on quiet days; treat empty as OK.
        source_ok["edgar"] = True
    except Exception as exc:
        log.exception("edgar source crashed: %s", exc)
        source_ok["edgar"] = False

    try:
        docs = stocktwits.fetch(TRACKED_TICKERS)
        all_docs.extend(docs)
        source_ok["stocktwits"] = bool(docs)
    except Exception as exc:
        log.exception("stocktwits source crashed: %s", exc)
        source_ok["stocktwits"] = False

    log.info("Source health: %s", source_ok)
    if not any(source_ok.values()):
        log.error("All sources failed — no parquet change")
        return 1

    # Optional polarity scoring.
    if no_score:
        polarities = [None] * len(all_docs)
        log.info("[--no-score] FinBERT skipped; polarity_n will be 0")
    elif scorer.is_available():
        log.info("Scoring %d docs with FinBERT…", len(all_docs))
        polarities = scorer.score_documents(all_docs)
        scored = sum(1 for p in polarities if p is not None)
        log.info("FinBERT scored %d / %d docs", scored, len(all_docs))
    else:
        polarities = [None] * len(all_docs)
        log.info("FinBERT unavailable (transformers/torch missing) — polarity_n will be 0")

    # Source-supplied polarity (StockTwits Bull/Bear tags) overrides FinBERT.
    # User-labeled trader sentiment is a better signal than FinBERT's reading
    # of a 140-char shout — see sentiment/sources/stocktwits.py.
    overrides = 0
    for i, d in enumerate(all_docs):
        ov = d.get("polarity_override")
        if ov is not None:
            polarities[i] = float(ov)
            overrides += 1
    if overrides:
        log.info("Applied %d source-supplied polarity overrides (StockTwits tags)", overrides)

    df = aggregate(all_docs, today=today_utc, polarities=polarities)
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
