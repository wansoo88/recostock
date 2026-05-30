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
