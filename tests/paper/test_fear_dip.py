"""Tests for the experimental fear-dip paper tracker open/close cycle."""
import pandas as pd
import pytest

import paper.fear_dip_tracker as fdt


@pytest.fixture
def close_df():
    # 30 trading days, SPY rising 1%/day from 100
    dates = pd.bdate_range("2026-01-01", periods=30)
    spy = pd.Series([100 * (1.01 ** i) for i in range(30)], index=dates)
    return pd.DataFrame({"SPY": spy})


@pytest.fixture(autouse=True)
def tmp_path_store(tmp_path, monkeypatch):
    monkeypatch.setattr(fdt, "PATH", tmp_path / "fear_dip_paper.parquet")


def _sig(close_df, pos, is_entry):
    d = close_df.index[pos]
    return {"date": d, "score": 1.5, "threshold": 1.0, "percentile": 0.9,
            "is_entry": is_entry, "entry_price": float(close_df["SPY"].iloc[pos])}


def test_opens_on_entry(close_df):
    sub = close_df.iloc[:6]
    t = fdt.update(sub, _sig(close_df, 5, True), close_df.index[5])
    assert (t["status"] == "open").sum() == 1
    assert t.iloc[0]["target_hold"] == fdt.FEAR_DIP_HOLD


def test_no_open_when_not_entry(close_df):
    sub = close_df.iloc[:6]
    t = fdt.update(sub, _sig(close_df, 5, False), close_df.index[5])
    assert t.empty or (t["status"] == "open").sum() == 0


def test_single_position_no_pyramiding(close_df):
    fdt.update(close_df.iloc[:6], _sig(close_df, 5, True), close_df.index[5])
    t = fdt.update(close_df.iloc[:7], _sig(close_df, 6, True), close_df.index[6])
    assert (t["status"] == "open").sum() == 1  # still only one


def test_closes_after_hold_with_profit(close_df):
    # enter at pos 5, hold 10 -> exit at pos 15. Rising market => profit.
    fdt.update(close_df.iloc[:6], _sig(close_df, 5, True), close_df.index[5])
    t = fdt.update(close_df.iloc[:16], _sig(close_df, 15, False), close_df.index[15])
    closed = t[t["status"] == "closed"]
    assert len(closed) == 1
    assert closed.iloc[0]["pnl_pct"] > 0  # ~ (1.01^10 - 1) - cost > 0


def test_metrics_after_close(close_df):
    fdt.update(close_df.iloc[:6], _sig(close_df, 5, True), close_df.index[5])
    fdt.update(close_df.iloc[:16], _sig(close_df, 15, False), close_df.index[15])
    m = fdt.metrics(fdt.load())
    assert m["n"] == 1
    assert m["winrate"] == 1.0
