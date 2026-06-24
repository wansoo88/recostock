"""Tests for the HTML report builder + the docs/index.html latest-redirect."""
from datetime import date

import pytest

import report.builder as rb


@pytest.fixture
def _tmp_docs(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    monkeypatch.setattr(rb, "OUTPUT_DIR", docs)
    return docs


def test_write_index_redirects_to_latest(_tmp_docs):
    for name in ("2026-06-01.html", "2026-06-10.html", "2026-06-05.html",
                 "intraday-2026-06-01.html", "index.html"):
        (_tmp_docs / name).write_text("x", encoding="utf-8")
    out = rb.write_index(_tmp_docs)
    html = out.read_text(encoding="utf-8")
    assert 'url=2026-06-10.html' in html                 # newest dated report
    assert "2026-06-01" in html and "2026-06-05" in html  # recent list
    assert "intraday" not in html                         # non-daily files excluded


def test_write_index_empty_dir_is_noop(_tmp_docs):
    assert rb.write_index(_tmp_docs) is None
    assert not (_tmp_docs / "index.html").exists()


def test_build_report_writes_report_and_index(_tmp_docs):
    regime = {"label": "normal", "vix": 18.9,
              "decision": {"stance": "hold", "headline": "오늘 할 일: 없음 — 그대로 보유",
                           "trades": [], "why": ["추세 ON"]}}
    out = rb.build_report([], regime, date(2026, 6, 10))
    assert out.name == "2026-06-10.html"
    body = out.read_text(encoding="utf-8")
    assert "const REPORT = {" in body
    assert "그대로 보유" in body                          # decision serialized in
    idx = (_tmp_docs / "index.html").read_text(encoding="utf-8")
    assert "url=2026-06-10.html" in idx


def test_build_report_carries_best_pick_satellite(_tmp_docs):
    """The best-pick satellite (report-only) must serialize into REPORT.regime and
    the template must contain its panel + renderer."""
    regime = {
        "label": "normal", "vix": 15.0,
        "bestPick": {
            "disciplined": {
                "mode": "disciplined", "pick": "XLK", "name": "Technology",
                "leverage": 1, "entry": 191.02, "tp": 196.75, "stop": 145.13,
                "distStopPct": 31.6, "pickAsOf": "2026-06-12",
                "ranked": [{"ticker": "XLK", "leverage": 1, "score": 15.4,
                            "above200": True, "distStopPct": 23.2}],
                "backtest": {"pool": "6 섹터", "rank": "RSI×60d", "fullRet": 182,
                             "fullSharpe": 1.15, "fullMdd": -19, "holdRet": 72,
                             "holdSharpe": 1.41, "holdMdd": -17, "wkMeanPct": 0.4,
                             "wkHit3Pct": 14, "wkWorstPct": -7.5,
                             "yearsPositive": "6/6", "bear2022Pct": 4.8, "gate": "PASS"},
            },
            "longshot": {"mode": "longshot", "pick": None, "note": "현금",
                         "ranked": [{"ticker": "XLE", "leverage": 1, "score": 1.0,
                                     "above200": False, "distStopPct": -2.0}],
                         "backtest": {"pool": "6섹터+3x", "rank": "RSI-14", "fullRet": 217,
                                      "fullSharpe": 0.86, "fullMdd": -25, "holdRet": 165,
                                      "holdSharpe": 1.41, "holdMdd": -22, "wkMeanPct": 0.47,
                                      "wkHit3Pct": 14, "wkWorstPct": -18.2,
                                      "yearsPositive": "4/6", "bear2022Pct": -11.6, "gate": "PASS"}},
        },
    }
    out = rb.build_report([], regime, date(2026, 6, 23))
    body = out.read_text(encoding="utf-8")
    assert '"bestPick"' in body                       # data carried through
    assert '"pick": "XLK"' in body
    assert 'id="bestpick-panel"' in body              # render target present
    assert "reg.bestPick || {}" in body               # renderer present
