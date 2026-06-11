"""Regression tests for the telegram daily message (decision-first, 2026-06-11).

The message contract: lead with the single decision (오늘 할 일), then the target
portfolio + stops, then why-bullets. The old scattered surfaces (RSI watchlist,
duplicate satellite suggestion, regime/expectancy header, flat-57% noise) must
NOT reappear. build_daily_message is pure, so most tests need no event loop.
"""
import asyncio
import types
from datetime import date

import pytest

import bot.notifier as nb

_DATE = date(2026, 6, 10)


def _decision(stance="hold", trades=None):
    return {
        "stance": stance,
        "headline": {"hold": "오늘 할 일: 없음 — 그대로 보유",
                     "rebalance": f"오늘 할 일: 리밸런스 {len(trades or [])}건",
                     "all_cash": "오늘 할 일: 전량 현금화 (추세 OFF) — 매도 후 BIL/SGOV 파킹"}[stance],
        "trades": trades or [],
        "nTrades": len(trades or []),
        "targetWeights": {"SPY": 0.3995, "QQQ": 0.425, "SPXL": 0.0255, "XLV": 0.075, "XLK": 0.075},
        "cashWeight": 0.0,
        "effExposure": 1.05,
        "why": ["추세 ON — SPY·QQQ 모두 200일선 위 (SPY 200일선 대비 +8.4%)",
                "VIX 18.9 — 평시 구성 (SPXL 기본 5%)",
                "섹터 슬리브 15%: XLV·XLK — RSI-14 상위 (2026-06-05 금요일 종가 선정), 주 1회 교체"],
        "prevDate": "2026-06-09",
    }


def _regime(**over):
    base = {
        "label": "normal", "exposure": 1.0, "vix": 18.9,
        "dataAsOf": "2026-06-09", "staleDays": 1, "stale": False,
        "decision": _decision(),
        "trendCore": {"coreOn": True, "exec": {
            "spy": {"price": 739.22, "stop": 682.10},
            "qqq": {"price": 716.07, "stop": 621.61},
            "spxl": {"price": 264.53}, "tiltDaysLeft": None}},
        "portfolio": {"weights": {"SPY": 0.3995, "QQQ": 0.425, "SPXL": 0.0255,
                                  "XLV": 0.075, "XLK": 0.075},
                      "cashWeight": 0.0, "effExposure": 1.05, "enabled": True,
                      "coreWeight": 0.85, "sleeveWeight": 0.15},
        "sectorSatellite": {"ranked": [{"ticker": "XLV", "rsi": 67.5}], "pick": ["XLV", "XLK"]},
        "portfolioPaper": {"nDays": 8, "months": 0.36, "totalReturn": -0.0309,
                           "annSharpe": -4.374, "targetSharpe": 1.23, "status": "warming up"},
        "candidates": [{"ticker": "XLV", "rsi": 67.5, "calWin": 0.57, "passed": False}],
    }
    base.update(over)
    return base


def test_hold_message_leads_with_decision():
    msg = nb.build_daily_message([], _regime(), "https://x/r.html", _DATE)
    lines = msg.splitlines()
    assert lines[0].startswith("📊 recostock 데일리")
    assert any("✅ 오늘 할 일: 없음" in l for l in lines)
    # decision comes before the portfolio section
    assert msg.index("오늘 할 일") < msg.index("목표 포트폴리오")


def test_rebalance_trades_are_listed():
    trades = [{"ticker": "XLE", "action": "전량 매도", "fromPct": 7.5, "toPct": 0.0, "deltaPct": -7.5},
              {"ticker": "XLV", "action": "신규 매수", "fromPct": 0.0, "toPct": 7.5, "deltaPct": 7.5}]
    msg = nb.build_daily_message([], _regime(decision=_decision("rebalance", trades)), "", _DATE)
    assert "🔄 오늘 할 일: 리밸런스 2건" in msg
    assert "XLE  7.5% → 0%  (전량 매도)" in msg
    assert "XLV  0% → 7.5%  (신규 매수)" in msg


def test_target_portfolio_and_stops():
    msg = nb.build_daily_message([], _regime(), "", _DATE)
    assert "📐 목표 포트폴리오 — 시장노출 ≈1.05x" in msg
    assert "QQQ 42.5%" in msg and "SPXL 2.5% (3x)" in msg
    assert "SPY 종가 < $682.10 → SPY·SPXL 청산" in msg
    assert "QQQ 종가 < $621.61 → QQQ 청산" in msg


def test_why_bullets_present():
    msg = nb.build_daily_message([], _regime(), "", _DATE)
    assert "💡 근거" in msg
    assert "추세 ON" in msg and "VIX 18.9" in msg and "XLV·XLK" in msg


def test_noise_surfaces_removed():
    msg = nb.build_daily_message([], _regime(), "", _DATE)
    assert "RSI순" not in msg                      # old watchlist line
    assert "RSI 섹터 로테이션(선택)" not in msg     # duplicate satellite suggestion
    assert "종합 기대값" not in msg                 # expectancy header
    assert "노출도" not in msg                      # VIX-regime exposure (conflicted with 1.05x)
    assert "57%" not in msg                         # flat calibrated-probability noise


def test_stale_warning_leads():
    msg = nb.build_daily_message([], _regime(stale=True, dataAsOf="2026-05-15", staleDays=16), "", _DATE)
    lines = msg.splitlines()
    assert "데이터 지연" in lines[1] and "2026-05-15" in lines[1]


def test_conviction_signal_renders_as_reference():
    sig = types.SimpleNamespace(ticker="XLK", entry=184.18, tp=189.71, sl=182.34,
                                winrate=0.7368, sample_n=19, direction="long", leverage=1)
    msg = nb.build_daily_message([sig], _regime(), "", _DATE)
    assert "⚡ 참고 — conviction 신호: XLK" in msg
    assert "n=19" in msg and "별개" in msg


def test_paper_validation_one_liner():
    msg = nb.build_daily_message([], _regime(), "", _DATE)
    assert "🧪 페이퍼 검증(실자본 아님) 8일/3개월" in msg
    assert "목표 1.23" in msg and "검증 초기" in msg


def test_fallback_without_decision_still_shows_target():
    msg = nb.build_daily_message([], _regime(decision=None), "", _DATE)
    assert "📐 목표 포트폴리오" in msg and "QQQ 4" in msg


def test_footer_link_and_disclaimer():
    msg = nb.build_daily_message([], _regime(), "https://pages/2026-06-10.html", _DATE)
    assert "🔗 상세 리포트: https://pages/2026-06-10.html" in msg
    assert "수동 실행" in msg


class _FakeBot:
    sent = []

    def __init__(self, token=None):
        pass

    async def send_message(self, chat_id=None, text=None):
        _FakeBot.sent.append(text)


def test_send_daily_signal_smoke(monkeypatch):
    _FakeBot.sent = []
    monkeypatch.setitem(__import__("sys").modules, "telegram",
                        types.SimpleNamespace(Bot=_FakeBot))
    asyncio.run(nb.send_daily_signal("tok", "chat", [], _regime(), "", _DATE))
    assert _FakeBot.sent and "오늘 할 일" in _FakeBot.sent[-1]
