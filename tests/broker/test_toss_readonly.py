"""Toss read-only client: no trading surface + schema-tolerant normalization.

The read-only constraint is a GATE requirement (Tier-2, ~2026-08-29), so the
first test is structural: the client must expose nothing that smells like
order placement. Network is never touched — auth flow runs against a fake
session.
"""
import types

import pytest

import broker.toss as toss
from broker.toss import (
    TossReadOnlyClient,
    build_snapshot,
    holdings_to_weights,
    normalize_positions,
)


def test_client_exposes_no_trading_surface():
    banned = ("order", "buy", "sell", "cancel", "modify", "trade", "execute")
    public = [m for m in dir(TossReadOnlyClient) if not m.startswith("_")]
    offenders = [m for m in public if any(b in m.lower() for b in banned)]
    assert not offenders, f"trading-like methods on read-only client: {offenders}"


def test_module_never_posts_outside_token_auth():
    # The ONLY POST allowed is the OAuth token request — anything else would
    # be a state-changing call on a read-only client.
    import inspect
    src = inspect.getsource(toss)
    assert src.count(".post(") == 1, "unexpected POST call added to read-only client"


# ── normalization across plausible response schemas ───────────────────────────

@pytest.mark.parametrize("raw", [
    {"positions": [{"ticker": "SPY", "quantity": 3, "evaluationAmount": 1800.0}]},
    {"data": [{"symbol": "spy", "qty": "3", "marketValue": "1800"}]},
    {"result": [{"stockCode": "SPY", "holdingQuantity": 3, "value": 1800}]},
    [{"code": "SPY", "balanceQty": 3, "amount": 1800.0}],
])
def test_normalize_positions_schema_tolerant(raw):
    out = normalize_positions(raw)
    assert out == [{"ticker": "SPY", "qty": 3.0, "value": 1800.0}]


def test_normalize_skips_bad_rows():
    raw = {"positions": [
        {"ticker": "SPY", "evaluationAmount": 1800.0},          # qty optional
        {"evaluationAmount": 500.0},                            # no ticker
        {"ticker": "QQQ", "evaluationAmount": "n/a"},           # bad value
        "garbage",
    ]}
    out = normalize_positions(raw)
    assert [p["ticker"] for p in out] == ["SPY"]
    assert out[0]["qty"] is None


def test_holdings_to_weights_sums_to_one_with_cash():
    pos = [{"ticker": "SPY", "qty": 1, "value": 700.0},
           {"ticker": "QQQ", "qty": 1, "value": 200.0}]
    w, cash_w = holdings_to_weights(pos, cash_value=100.0)
    assert w == {"SPY": 0.7, "QQQ": 0.2}
    assert cash_w == 0.1
    assert abs(sum(w.values()) + cash_w - 1.0) < 1e-9


def test_holdings_to_weights_empty_account():
    assert holdings_to_weights([], 0.0) == ({}, 0.0)


def test_build_snapshot_is_sanitized():
    pos = [{"ticker": "SPY", "qty": 12.34, "value": 7000.0}]
    snap = build_snapshot(pos, cash_value=3000.0, as_of="2026-06-12")
    assert snap == {"asOf": "2026-06-12", "source": "toss-openapi",
                    "weights": {"SPY": 0.7}, "cashWeight": 0.3}
    # absolute sizes must never leave the server
    flat = str(snap)
    assert "7000" not in flat and "3000" not in flat and "12.34" not in flat


# ── auth flow against a fake session (no network) ─────────────────────────────

class _Resp:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


class _FakeSession:
    def __init__(self):
        self.posts, self.gets = [], []

    def post(self, url, data=None, timeout=None):
        self.posts.append((url, data))
        return _Resp({"access_token": "tok-1", "expires_in": 600})

    def get(self, url, params=None, headers=None, timeout=None):
        self.gets.append((url, headers))
        return _Resp({"positions": [{"ticker": "SPY", "quantity": 1,
                                     "evaluationAmount": 100.0}]})


def test_token_cached_and_bearer_sent():
    s = _FakeSession()
    c = TossReadOnlyClient("k", "s", base_url="https://api.test", session=s)
    assert c.positions() == [{"ticker": "SPY", "qty": 1.0, "value": 100.0}]
    c.positions()
    assert len(s.posts) == 1, "token must be cached across calls"
    assert s.posts[0][0] == "https://api.test" + c.token_path
    assert all(h["Authorization"] == "Bearer tok-1" for _, h in s.gets)


def test_missing_credentials_raise():
    with pytest.raises(ValueError):
        TossReadOnlyClient("", "")
