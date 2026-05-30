"""Tests for the data-freshness / stale-data guard."""
import pandas as pd

from data.collector import data_freshness


def test_weekend_gap_is_fresh():
    # Friday close, Sunday run — 2 days, normal
    r = data_freshness("2026-05-29", "2026-05-31")
    assert r["dataAsOf"] == "2026-05-29"
    assert r["staleDays"] == 2
    assert r["stale"] is False


def test_friday_to_monday_ok():
    r = data_freshness("2026-05-29", "2026-06-01")  # 3 days
    assert r["staleDays"] == 3 and r["stale"] is False


def test_boundary_four_days_ok():
    r = data_freshness("2026-05-29", "2026-06-02")  # 4 days = max, not stale
    assert r["staleDays"] == 4 and r["stale"] is False


def test_five_days_is_stale():
    r = data_freshness("2026-05-29", "2026-06-03")  # 5 days
    assert r["staleDays"] == 5 and r["stale"] is True


def test_very_stale():
    r = data_freshness("2026-05-15", "2026-05-31")
    assert r["staleDays"] == 16 and r["stale"] is True


def test_accepts_timestamp_inputs():
    r = data_freshness(pd.Timestamp("2026-05-29 16:00"), pd.Timestamp("2026-05-31"))
    assert r["dataAsOf"] == "2026-05-29" and r["stale"] is False


def test_custom_max_days():
    r = data_freshness("2026-05-29", "2026-06-01", max_days=2)
    assert r["staleDays"] == 3 and r["stale"] is True
