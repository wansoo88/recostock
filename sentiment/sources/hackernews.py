"""HackerNews Algolia search API — no key required.

Endpoint:
    https://hn.algolia.com/api/v1/search?
        query=<ticker or alias>&tags=story&numericFilters=created_at_i>{epoch}
Returns up to 1000 hits per query. We restrict to stories within the
look-back window (default 36h so daily runs overlap and catch late posts).
"""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from typing import Iterable

log = logging.getLogger(__name__)

BASE = "https://hn.algolia.com/api/v1/search"
UA = "recostock-sentiment/1.0"
DELAY_S = 0.3
TIMEOUT_S = 10
HITS_PER_PAGE = 50


def _fetch_query(query: str, since_epoch: int) -> list[dict]:
    params = {
        "query": query,
        "tags": "story",
        "numericFilters": f"created_at_i>{since_epoch}",
        "hitsPerPage": HITS_PER_PAGE,
    }
    url = f"{BASE}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        log.warning("HN query %r failed: %s", query, exc)
        return []
    except json.JSONDecodeError:
        return []

    items: list[dict] = []
    for hit in data.get("hits", []):
        items.append({
            "source": "hackernews",
            "query_ticker": query,
            "title": hit.get("title") or "",
            "body": hit.get("story_text") or hit.get("comment_text") or "",
            "published": datetime.fromtimestamp(
                hit.get("created_at_i", 0), tz=timezone.utc
            ),
        })
    return items


def fetch(queries: Iterable[str], lookback_hours: int = 36) -> list[dict]:
    """Search HN for each query string and merge results."""
    since = int((datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).timestamp())
    out: list[dict] = []
    for q in queries:
        out.extend(_fetch_query(q, since))
        time.sleep(DELAY_S)
    log.info("hackernews: fetched %d items across %d queries", len(out), len(list(queries)) if hasattr(queries, "__len__") else -1)
    return out
