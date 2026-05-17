"""StockTwits per-symbol stream — no API key, ~200 req/hour.

Endpoint:  https://api.stocktwits.com/api/2/streams/symbol/{SYMBOL}.json
Returns the most recent ~30 messages per symbol.

Why StockTwits over Reddit
--------------------------
- Finance-focused community; signal/noise > Reddit for ETF threads.
- Users explicitly tag messages Bullish/Bearish — pre-labeled polarity.
- No auth, no account-age gating; usable from CI immediately.

Polarity handling
-----------------
The aggregator pipeline computes polarity via FinBERT on title+body.
StockTwits messages carry a user-supplied Bullish/Bearish tag that is a
*better* label than FinBERT's reading of a 140-char trader shout, so
each doc carries an optional `polarity_override`:

    Bullish  → +1.0
    Bearish  → -1.0
    untagged → None  (FinBERT will score body normally)

The collector applies the override after FinBERT runs.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Iterable

log = logging.getLogger(__name__)

URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
UA = "recostock-sentiment/1.0"
DELAY_S = 0.5
TIMEOUT_S = 15


def _parse_created(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _polarity_from_sentiment(entities: dict | None) -> float | None:
    if not entities:
        return None
    s = entities.get("sentiment")
    if not s:
        return None
    basic = (s.get("basic") or "").lower()
    if basic == "bullish":
        return 1.0
    if basic == "bearish":
        return -1.0
    return None


def _fetch_one(ticker: str) -> list[dict]:
    url = URL.format(ticker=ticker)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        log.warning("StockTwits %s failed: %s", ticker, exc)
        return []
    except json.JSONDecodeError:
        return []

    status = data.get("response", {}).get("status")
    if status != 200:
        log.warning("StockTwits %s non-200 response: status=%s", ticker, status)
        return []

    items: list[dict] = []
    for m in data.get("messages", []):
        body_text = (m.get("body") or "").strip()
        if not body_text:
            continue
        pub = _parse_created(m.get("created_at"))
        polarity = _polarity_from_sentiment(m.get("entities"))
        # Prepend cashtag so the aggregator's ticker_extract attributes
        # the row to this ticker without depending on the user including it.
        body = f"${ticker} {body_text}"
        items.append({
            "source": "stocktwits",
            "query_ticker": ticker,
            "title": body_text[:200],
            "body": body,
            "published": pub,
            "polarity_override": polarity,
        })
    return items


def fetch(tickers: Iterable[str]) -> list[dict]:
    """Fetch recent messages for every ticker. Respects ~200 req/hr cap."""
    tickers = list(tickers)
    out: list[dict] = []
    for t in tickers:
        out.extend(_fetch_one(t))
        time.sleep(DELAY_S)
    tagged = sum(1 for d in out if d.get("polarity_override") is not None)
    log.info(
        "stocktwits: fetched %d messages across %d tickers (%d user-tagged Bull/Bear)",
        len(out), len(tickers), tagged,
    )
    return out
