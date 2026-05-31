"""Regression tests for the telegram daily-signal message construction.

Guards the 2026-05-31 bug: the old '🎯 TOP PICK' line referenced pick['tp'] and
pick['sl'], but the candidate dict (rebuilt 2026-05-29) has no tp/sl keys — so a
passing candidate would raise KeyError and the whole telegram push would fail.
These tests build the message with the real candidate contract and assert no crash.
"""
import asyncio
import types

import pytest

import bot.notifier as nb


class _FakeBot:
    """Captures the message instead of hitting the network."""
    sent = []

    def __init__(self, token=None):
        pass

    async def send_message(self, chat_id=None, text=None):
        _FakeBot.sent.append(text)


@pytest.fixture(autouse=True)
def _fake_telegram(monkeypatch):
    _FakeBot.sent = []
    fake_mod = types.SimpleNamespace(Bot=_FakeBot)
    # send_daily_signal does `import telegram` inside the function
    monkeypatch.setitem(__import__("sys").modules, "telegram", fake_mod)


def _candidate(ticker, rsi, passed=False):
    # Mirror the real candidate dict from scripts/run_daily.py (no tp/sl keys!).
    return {
        "ticker": ticker, "name": ticker, "confidence": 0.66,
        "calWin": 0.57, "rs": 0.05, "rsi": rsi,
        "above50": True, "above200": True,
        "entry": 100.0, "hi": 103.0, "lo": 97.0, "bandPct": 0.03,
        "estEv": 0.01, "passed": passed,
    }


def _run(regime):
    from datetime import date
    asyncio.run(nb.send_daily_signal("tok", "chat", [], regime, "", date(2026, 5, 31)))
    return _FakeBot.sent[-1] if _FakeBot.sent else ""


def test_passing_candidate_does_not_crash():
    # The exact old-bug trigger: a candidate with passed=True.
    regime = {"label": "normal", "exposure": 1.0,
              "candidates": [_candidate("XLK", 72, passed=True),
                             _candidate("XLV", 68)]}
    msg = _run(regime)  # must not raise KeyError
    assert "XLK" in msg
    assert "TP" not in msg  # old fixed-TP/SL framing is gone
    assert "RSI순" in msg   # new RSI watchlist present


def test_no_candidates_section_when_empty():
    msg = _run({"label": "normal", "exposure": 1.0, "candidates": []})
    assert "RSI순" not in msg


def test_rsi_watchlist_lists_top_names():
    regime = {"label": "normal", "exposure": 1.0,
              "candidates": [_candidate("XLK", 72), _candidate("XLE", 60),
                             _candidate("XLF", 54)]}
    msg = _run(regime)
    assert "XLK 72" in msg and "XLE 60" in msg


def test_stale_warning_surfaces():
    regime = {"label": "normal", "exposure": 1.0, "candidates": [],
              "stale": True, "dataAsOf": "2026-05-15", "staleDays": 16}
    msg = _run(regime)
    assert "데이터 지연" in msg and "2026-05-15" in msg
