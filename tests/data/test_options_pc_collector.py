"""Tests for data/options_pc_collector.py."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data import options_pc_collector as pc


@pytest.fixture
def tmp_history(monkeypatch, tmp_path):
    """Redirect PC_PATH to a temp file so the test doesn't touch real data."""
    target = tmp_path / "spy_pc_daily.csv"
    monkeypatch.setattr(pc, "PC_PATH", target)
    return target


def _mock_chain(call_vol, call_oi, put_vol, put_oi):
    """Build a yfinance-style chain mock."""
    return MagicMock(
        calls=pd.DataFrame({
            "volume": [call_vol],
            "openInterest": [call_oi],
        }),
        puts=pd.DataFrame({
            "volume": [put_vol],
            "openInterest": [put_oi],
        }),
    )


def test_compute_spy_pc_sums_across_expirations():
    """P/C aggregates volume + OI across all sampled expirations."""
    fake = MagicMock()
    fake.options = ["2026-05-22", "2026-05-29"]
    # Expiration 1: calls vol=100 OI=200, puts vol=150 OI=300
    # Expiration 2: calls vol=200 OI=400, puts vol=250 OI=500
    fake.option_chain.side_effect = [
        _mock_chain(100, 200, 150, 300),
        _mock_chain(200, 400, 250, 500),
    ]
    fake.history.return_value = pd.DataFrame({"Close": [500.0]})

    with patch.dict(sys.modules, {"yfinance": MagicMock(Ticker=lambda _: fake)}):
        result = pc.compute_spy_pc(n_expirations=2)

    assert result is not None
    # Aggregate: calls vol 300, OI 600; puts vol 400, OI 800
    assert result["vol_pc"] == round(400 / 300, 4)
    assert result["oi_pc"] == round(800 / 600, 4)
    assert result["n_expirations"] == 2
    assert result["underlying_px"] == 500.00


def test_compute_returns_none_when_chain_empty():
    fake = MagicMock()
    fake.options = []
    with patch.dict(sys.modules, {"yfinance": MagicMock(Ticker=lambda _: fake)}):
        assert pc.compute_spy_pc() is None


def test_compute_returns_none_when_all_call_volumes_zero():
    fake = MagicMock()
    fake.options = ["2026-05-22"]
    fake.option_chain.return_value = _mock_chain(0, 0, 100, 100)
    fake.history.return_value = pd.DataFrame({"Close": [500.0]})
    with patch.dict(sys.modules, {"yfinance": MagicMock(Ticker=lambda _: fake)}):
        assert pc.compute_spy_pc() is None


def test_append_today_writes_parquet(tmp_history):
    """First append creates parquet with one row."""
    fake = MagicMock()
    fake.options = ["2026-05-22"]
    fake.option_chain.return_value = _mock_chain(100, 200, 150, 300)
    fake.history.return_value = pd.DataFrame({"Close": [500.0]})
    with patch.dict(sys.modules, {"yfinance": MagicMock(Ticker=lambda _: fake)}):
        row = pc.append_today(today=date(2026, 5, 18))
    assert row is not None
    df = pc.load_history()
    assert len(df) == 1
    assert df["date"].iloc[0] == date(2026, 5, 18)


def test_append_today_overwrites_same_date(tmp_history):
    """Re-running on the same day overwrites — no duplicate rows."""
    fake = MagicMock()
    fake.options = ["2026-05-22"]
    fake.history.return_value = pd.DataFrame({"Close": [500.0]})
    # First run with one set of volumes
    fake.option_chain.return_value = _mock_chain(100, 200, 150, 300)
    with patch.dict(sys.modules, {"yfinance": MagicMock(Ticker=lambda _: fake)}):
        pc.append_today(today=date(2026, 5, 18))
    # Second run same day with different volumes
    fake.option_chain.return_value = _mock_chain(100, 200, 250, 400)
    with patch.dict(sys.modules, {"yfinance": MagicMock(Ticker=lambda _: fake)}):
        pc.append_today(today=date(2026, 5, 18))
    df = pc.load_history()
    assert len(df) == 1
    # Should reflect the SECOND run's data (overwritten)
    assert df["vol_pc"].iloc[0] == round(250 / 100, 4)


def test_append_keeps_distinct_dates(tmp_history):
    """Different dates accumulate, do not overwrite."""
    fake = MagicMock()
    fake.options = ["2026-05-22"]
    fake.option_chain.return_value = _mock_chain(100, 200, 150, 300)
    fake.history.return_value = pd.DataFrame({"Close": [500.0]})
    with patch.dict(sys.modules, {"yfinance": MagicMock(Ticker=lambda _: fake)}):
        pc.append_today(today=date(2026, 5, 18))
        pc.append_today(today=date(2026, 5, 19))
        pc.append_today(today=date(2026, 5, 20))
    df = pc.load_history()
    assert len(df) == 3
    assert df["date"].tolist() == [date(2026, 5, 18), date(2026, 5, 19), date(2026, 5, 20)]


def test_status_summary_handles_empty(tmp_history):
    assert "empty" in pc.status_summary().lower()


def test_status_summary_handles_short_history(tmp_history):
    fake = MagicMock()
    fake.options = ["2026-05-22"]
    fake.option_chain.return_value = _mock_chain(100, 200, 150, 300)
    fake.history.return_value = pd.DataFrame({"Close": [500.0]})
    with patch.dict(sys.modules, {"yfinance": MagicMock(Ticker=lambda _: fake)}):
        for d in range(18, 22):
            pc.append_today(today=date(2026, 5, d))
    msg = pc.status_summary()
    assert "4 rows" in msg
    assert "need 56" in msg   # 60 - 4
