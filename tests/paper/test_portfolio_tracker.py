"""Tests for the portfolio NAV paper tracker (3-month blend validation)."""
import json

import numpy as np
import pandas as pd
import pytest

import paper.portfolio_tracker as pt


@pytest.fixture(autouse=True)
def _tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "PATH", tmp_path / "portfolio_nav.parquet")
    # Force cash yield to 0 so tests are deterministic regardless of repo data.
    monkeypatch.setattr(pt, "_cash_daily_yield", lambda: 0.0)


def _close(prices: dict, dates):
    """prices: {ticker: [p0, p1, ...]} aligned to dates."""
    return pd.DataFrame(prices, index=pd.to_datetime(dates))


def test_first_record_is_seed_nav_one():
    close = _close({"SPY": [100.0]}, ["2026-06-01"])
    pf = {"weights": {"SPY": 1.0}, "cashWeight": 0.0, "effExposure": 1.0}
    out = pt.update(close, pf, today="2026-06-01")
    assert len(out) == 1
    assert out.iloc[0]["nav"] == pytest.approx(1.0)
    assert out.iloc[0]["net_ret"] == pytest.approx(0.0)


def test_realized_return_from_prior_weights():
    close = _close({"SPY": [100.0, 110.0]}, ["2026-06-01", "2026-06-02"])
    pf = {"weights": {"SPY": 1.0}, "cashWeight": 0.0, "effExposure": 1.0}
    pt.update(close.iloc[:1], pf, today="2026-06-01")          # seed
    out = pt.update(close, pf, today="2026-06-02")             # +10% move, no turnover
    assert len(out) == 2
    assert out.iloc[1]["gross_ret"] == pytest.approx(0.10, abs=1e-9)
    assert out.iloc[1]["cost"] == pytest.approx(0.0, abs=1e-12)
    assert out.iloc[1]["nav"] == pytest.approx(1.10, abs=1e-9)


def test_turnover_cost_on_rebalance():
    close = _close({"SPY": [100.0, 100.0], "QQQ": [50.0, 50.0]},
                   ["2026-06-01", "2026-06-02"])
    pt.update(close.iloc[:1], {"weights": {"SPY": 1.0}, "cashWeight": 0.0}, today="2026-06-01")
    out = pt.update(close, {"weights": {"QQQ": 1.0}, "cashWeight": 0.0}, today="2026-06-02")
    # full switch SPY->QQQ: turnover 2.0 * (0.0025/2) = 0.0025
    assert out.iloc[1]["cost"] == pytest.approx(0.0025, abs=1e-9)
    assert out.iloc[1]["net_ret"] == pytest.approx(-0.0025, abs=1e-9)


def test_idempotent_same_day_rerun():
    close = _close({"SPY": [100.0]}, ["2026-06-01"])
    pf = {"weights": {"SPY": 1.0}, "cashWeight": 0.0}
    pt.update(close, pf, today="2026-06-01")
    out = pt.update(close, pf, today="2026-06-01")  # re-run
    assert len(out) == 1  # no duplicate row


def test_last_weights_before_returns_prior_holdings():
    close = _close({"SPY": [100.0, 101.0], "QQQ": [50.0, 50.0]},
                   ["2026-06-01", "2026-06-02"])
    pt.update(close.iloc[:1], {"weights": {"SPY": 1.0}, "cashWeight": 0.0}, today="2026-06-01")
    pt.update(close, {"weights": {"QQQ": 1.0}, "cashWeight": 0.0}, today="2026-06-02")
    prev = pt.last_weights_before("2026-06-02")     # strictly before -> day 1
    assert prev["date"] == "2026-06-01"
    assert prev["weights"] == {"SPY": 1.0}
    # same-day re-run must never diff against its own (overwritten) record
    assert pt.last_weights_before("2026-06-01") is None


def test_last_weights_before_empty_store():
    assert pt.last_weights_before("2026-06-01") is None


def test_nav_history_chronological():
    close = _close({"SPY": [100.0, 110.0]}, ["2026-06-01", "2026-06-02"])
    pf = {"weights": {"SPY": 1.0}, "cashWeight": 0.0}
    pt.update(close.iloc[:1], pf, today="2026-06-01")
    pt.update(close, pf, today="2026-06-02")
    h = pt.nav_history()
    assert [r["date"] for r in h] == ["2026-06-01", "2026-06-02"]
    assert h[0]["nav"] == pytest.approx(1.0)
    assert h[1]["nav"] == pytest.approx(1.10, abs=1e-9)


def test_attribution_splits_engine_and_sleeve():
    # Day1 -> Day2: SPY +10% at 50%w (engine +5%p), XLK -10% at 50%w (sleeve -5%p)
    close = _close({"SPY": [100.0, 110.0], "XLK": [50.0, 45.0]},
                   ["2026-06-01", "2026-06-02"])
    pf = {"weights": {"SPY": 0.5, "XLK": 0.5}, "cashWeight": 0.0}
    pt.update(close.iloc[:1], pf, today="2026-06-01")
    pt.update(close, pf, today="2026-06-02")
    at = pt.attribution(close)
    assert at["engine"] == pytest.approx(0.05, abs=1e-6)
    assert at["sleeve"] == pytest.approx(-0.05, abs=1e-6)
    assert at["cost"] == pytest.approx(0.0, abs=1e-9)   # no turnover
    assert at["approx"] is True and at["days"] == 1


def test_attribution_none_with_single_record():
    close = _close({"SPY": [100.0]}, ["2026-06-01"])
    pt.update(close, {"weights": {"SPY": 1.0}, "cashWeight": 0.0}, today="2026-06-01")
    assert pt.attribution(close) is None


def test_metrics_exposes_chart_pace_constants():
    m = pt.metrics()                       # works even on the empty store
    assert m["paceDaily"] > 0
    assert m["sigmaDaily"] > m["paceDaily"]  # daily vol dwarfs daily drift


def test_blended_two_asset_return():
    close = _close({"SPY": [100.0, 110.0], "QQQ": [100.0, 90.0]},
                   ["2026-06-01", "2026-06-02"])
    pf = {"weights": {"SPY": 0.5, "QQQ": 0.5}, "cashWeight": 0.0}
    pt.update(close.iloc[:1], pf, today="2026-06-01")
    out = pt.update(close, pf, today="2026-06-02")
    # 0.5*(+10%) + 0.5*(-10%) = 0
    assert out.iloc[1]["gross_ret"] == pytest.approx(0.0, abs=1e-9)


def test_metrics_warming_up_then_fields():
    close = _close({"SPY": [100.0, 101.0, 102.0]},
                   ["2026-06-01", "2026-06-02", "2026-06-03"])
    pf = {"weights": {"SPY": 1.0}, "cashWeight": 0.0}
    for i, d in enumerate(["2026-06-01", "2026-06-02", "2026-06-03"]):
        pt.update(close.iloc[:i + 1], pf, today=d)
    m = pt.metrics()
    assert m["nDays"] == 3
    assert m["status"] == "warming up"          # < 3 months
    assert m["monthsOk"] is False
    assert m["passed"] is False
    assert m["targetSharpe"] == pytest.approx(1.23, abs=0.01)
    # NAV rose ~2% over the two real steps
    assert m["totalReturn"] > 0


def test_metrics_empty():
    m = pt.metrics()
    assert m["nDays"] == 0 and m["status"] == "no data" and m["passed"] is False
