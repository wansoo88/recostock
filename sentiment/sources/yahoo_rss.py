"""Yahoo Finance per-ticker headline RSS — no API key required.

URL pattern:  https://finance.yahoo.com/rss/headline?s=SPY
Returns ≤ 20 most recent items. Yahoo throttles aggressively if hit fast,
so we serialize requests with a small delay.
"""
from __future__ import annotations

import logging
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable

log = logging.getLogger(__name__)

URL = "https://finance.yahoo.com/rss/headline?s={ticker}"
UA = "Mozilla/5.0 (compatible; recostock-sentiment/1.0)"
DELAY_S = 0.4
TIMEOUT_S = 10


def _parse_pub_date(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = parsedate_to_datetime(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _fetch_one(ticker: str) -> list[dict]:
    req = urllib.request.Request(URL.format(ticker=ticker), headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            xml_bytes = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        log.warning("Yahoo RSS %s failed: %s", ticker, exc)
        return []

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        log.warning("Yahoo RSS %s parse error: %s", ticker, exc)
        return []

    items: list[dict] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        desc = (item.findtext("description") or "").strip()
        pub = _parse_pub_date(item.findtext("pubDate"))
        items.append({
            "source": "yahoo_rss",
            "query_ticker": ticker,
            "title": title,
            "body": desc,
            "published": pub,
        })
    return items


def fetch(tickers: Iterable[str]) -> list[dict]:
    """Fetch headlines for every ticker, with throttling."""
    out: list[dict] = []
    for t in tickers:
        out.extend(_fetch_one(t))
        time.sleep(DELAY_S)
    log.info("yahoo_rss: fetched %d items across %d tickers", len(out), len(list(tickers)) if hasattr(tickers, "__len__") else -1)
    return out
