"""StockTwits adapter — offline unit tests + opt-in live smoke test.

The unit tests stub urllib.request.urlopen with canned JSON to verify the
adapter contract (schema, polarity mapping, cashtag embedding). They do
NOT hit the network and run in CI by default.

The live test is gated behind RECOSTOCK_LIVE_NETWORK=1 so it only runs
when explicitly requested:

    RECOSTOCK_LIVE_NETWORK=1 pytest tests/sentiment/test_stocktwits.py -v -k live
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sentiment.sources import stocktwits


def _fake_response(payload: dict):
    """Construct a context-manager mock matching urlopen's return shape."""
    class _Resp:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *a):
            return False

        def read(self_inner):
            return json.dumps(payload).encode("utf-8")

    return _Resp()


SAMPLE_PAYLOAD = {
    "response": {"status": 200},
    "messages": [
        {
            "id": 1,
            "body": "$SPY breaking out, long calls into close",
            "created_at": "2026-05-17T15:30:00Z",
            "entities": {"sentiment": {"basic": "Bullish"}},
        },
        {
            "id": 2,
            "body": "rolling over hard, puts loading",
            "created_at": "2026-05-17T15:31:00Z",
            "entities": {"sentiment": {"basic": "Bearish"}},
        },
        {
            "id": 3,
            "body": "watching for direction",
            "created_at": "2026-05-17T15:32:00Z",
            "entities": {"sentiment": None},
        },
        {
            "id": 4,
            "body": "",  # empty body — should be skipped
            "created_at": "2026-05-17T15:33:00Z",
            "entities": None,
        },
    ],
}


def test_fetch_one_returns_expected_schema():
    with patch.object(stocktwits.urllib.request, "urlopen", return_value=_fake_response(SAMPLE_PAYLOAD)):
        items = stocktwits._fetch_one("SPY")

    # Empty-body message is filtered out.
    assert len(items) == 3
    expected_keys = {"source", "query_ticker", "title", "body", "published", "polarity_override"}
    for it in items:
        assert expected_keys.issubset(it.keys())
        assert it["source"] == "stocktwits"
        assert it["query_ticker"] == "SPY"
        # Body must start with the cashtag so ticker_extract attributes the row.
        assert it["body"].startswith("$SPY")
        assert it["published"] is not None


def test_polarity_mapping():
    with patch.object(stocktwits.urllib.request, "urlopen", return_value=_fake_response(SAMPLE_PAYLOAD)):
        items = stocktwits._fetch_one("SPY")

    polarities = [it["polarity_override"] for it in items]
    assert polarities == [1.0, -1.0, None]


def test_non_200_response_returns_empty():
    bad = {"response": {"status": 429}, "messages": []}
    with patch.object(stocktwits.urllib.request, "urlopen", return_value=_fake_response(bad)):
        items = stocktwits._fetch_one("SPY")
    assert items == []


def test_fetch_iterates_tickers_and_throttles(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(stocktwits.time, "sleep", lambda s: sleep_calls.append(s))
    with patch.object(stocktwits.urllib.request, "urlopen", return_value=_fake_response(SAMPLE_PAYLOAD)):
        out = stocktwits.fetch(["SPY", "QQQ"])
    # 3 valid messages per ticker × 2 tickers.
    assert len(out) == 6
    assert sleep_calls == [stocktwits.DELAY_S, stocktwits.DELAY_S]


@pytest.mark.skipif(
    os.environ.get("RECOSTOCK_LIVE_NETWORK") != "1",
    reason="set RECOSTOCK_LIVE_NETWORK=1 to run the live API smoke test",
)
def test_live_smoke_spy():
    """One real call to StockTwits to confirm the adapter still parses prod JSON."""
    items = stocktwits._fetch_one("SPY")
    # SPY has continuous traffic — even off-hours, messages persist for days.
    assert len(items) > 0
    for it in items:
        assert it["source"] == "stocktwits"
        assert it["body"].startswith("$SPY")
        assert it["published"] is not None
