"""Sentiment data pipeline (Phase A).

Collects daily ticker mention counts from free public sources, stored as
long-format parquet at data/raw/sentiment_daily.parquet:

    columns = [date (date), ticker (str), source (str), mention_count (int)]

Sources currently wired:
  - Yahoo Finance per-ticker headline RSS  (sources.yahoo_rss)
  - HackerNews Algolia search              (sources.hackernews)

Reddit / SEC EDGAR are intentionally deferred until user provides API keys
or a confirmed need is established.
"""
