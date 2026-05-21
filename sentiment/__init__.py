"""Sentiment data pipeline (Phase A).

Collects daily ticker mention counts from free public sources, stored as
long-format parquet at data/sentiment/sentiment_daily.parquet:

    columns = [date, ticker, source, mention_count, polarity_mean, polarity_n]

Sources currently wired:
  - Yahoo Finance per-ticker headline RSS  (sources.yahoo_rss)
  - HackerNews Algolia search              (sources.hackernews)
  - SEC EDGAR full-text filing search      (sources.edgar)
  - StockTwits per-symbol stream           (sources.stocktwits)

Reddit is intentionally deferred — StockTwits covers the same retail
sentiment niche with better signal/noise and no auth gating.
"""
