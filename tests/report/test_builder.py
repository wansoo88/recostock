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
