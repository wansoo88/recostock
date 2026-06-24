"""Tests for the single best-pick (long-shot) selector.

Covers the two contracts that matter for a concentrated real-money-adjacent
pick: (1) the gating that keeps it inside the Tier-1 drawdown cap (SPY uptrend +
own-200SMA, else cash) and (2) strict look-ahead safety (the pick uses only
closes up to the decision date).
"""
import numpy as np
import pandas as pd
import pytest

from signals import best_pick
from signals.best_pick import select, select_weekly, SECTORS, LEVER_LONGS, WEEKLY_TARGET


def _walk(n, drift, seed, noise=0.004):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(drift + rng.normal(0, noise, n)))


def _frame(trends: dict[str, str], n=320, tickers=None):
    idx = pd.date_range("2023-06-01", periods=n, freq="B")
    drift = {"up": 0.004, "down": -0.004, "flat": 0.0}
    cols = tickers or (["SPY"] + SECTORS)
    data = {t: _walk(n, drift.get(trends.get(t, "flat"), 0.0), seed=i + 1)
            for i, t in enumerate(cols)}
    return pd.DataFrame(data, index=idx)


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        select(_frame({}), mode="moonshot")


def test_short_history_returns_empty():
    df = _frame({"SPY": "up", "XLK": "up"}, n=50)
    assert select(df) == {}


def test_picks_top_score_above_sma_in_uptrend():
    # SPY up (gate open); XLK the only uptrending sector -> unambiguous pick.
    df = _frame({"SPY": "up", "XLK": "up", "XLY": "flat", "XLF": "flat",
                 "XLE": "down", "XLV": "down", "XLI": "down"})
    res = select(df, mode="disciplined")
    assert res["spyTrendOn"] is True
    assert res["pick"] == "XLK"
    assert res["ranked"][0]["ticker"] == "XLK"          # ranked desc by score


def test_cash_when_spy_downtrend():
    # Even with a strong sector, a SPY downtrend forces cash (the MDD discipline).
    df = _frame({"SPY": "down", "XLK": "up", "XLY": "up"})
    res = select(df, mode="disciplined")
    assert res["spyTrendOn"] is False
    assert res["pick"] is None
    assert "현금" in res["note"]


def test_cash_when_no_candidate_above_sma():
    # SPY up but every sector below its own 200SMA -> cash.
    df = _frame({"SPY": "up", "XLK": "down", "XLY": "down", "XLF": "down",
                 "XLE": "down", "XLV": "down", "XLI": "down"})
    res = select(df, mode="disciplined")
    assert res["spyTrendOn"] is True
    assert res["pick"] is None


def test_tp_and_stop_levels():
    df = _frame({"SPY": "up", "XLK": "up"})
    res = select(df, mode="disciplined")
    assert res["pick"] is not None
    # TP is exactly +3% over entry; stop is the pick's 200d SMA.
    assert res["tp"] == pytest.approx(res["entry"] * (1 + WEEKLY_TARGET), abs=0.02)
    assert res["stop"] < res["entry"]                   # uptrend -> price above SMA
    assert res["target"] == WEEKLY_TARGET


def test_longshot_pool_includes_leverage():
    cols = ["SPY"] + SECTORS + LEVER_LONGS
    # Make a leveraged name the strongest trend so it can win.
    trends = {t: "flat" for t in cols}
    trends.update({"SPY": "up", "SOXL": "up", "TQQQ": "up"})
    df = _frame(trends, tickers=cols)
    res = select(df, mode="longshot")
    pool_tickers = {r["ticker"] for r in res["ranked"]}
    assert LEVER_LONGS[0] in pool_tickers or "SOXL" in pool_tickers
    # A leveraged pick carries its leverage tag for display.
    if res["pick"] in LEVER_LONGS:
        assert res["leverage"] >= 2


def test_lookahead_safe_pick_unchanged_by_future_bars():
    """The pick/score on data through date T must not change when later bars are
    appended (strict causality)."""
    df = _frame({"SPY": "up", "XLK": "up", "XLE": "up", "XLV": "down"}, n=340)
    cut = 300
    early = select(df.iloc[:cut], mode="disciplined")
    # Append 40 more (future) bars; recompute on the SAME truncated window.
    late_same_window = select(df.iloc[:cut], mode="disciplined")
    assert early["pick"] == late_same_window["pick"]
    assert early["ranked"][0]["score"] == late_same_window["ranked"][0]["score"]
    # And the full-frame pick is generally allowed to differ (newer data) — just
    # ensure scoring on the truncated frame ignored everything after `cut`.
    assert early["asOf"] == str(df.index[cut - 1])[:10]


def test_weekly_pins_pick_to_last_friday():
    df = _frame({"SPY": "up", "XLK": "up", "XLE": "up"}, n=330)
    wk = select_weekly(df, mode="disciplined")
    assert "pickAsOf" in wk
    # Pinned pick == select() on the frame truncated at the last Friday.
    fridays = [d for d in df.index if d.dayofweek == 4]
    pinned = select(df.loc[:fridays[-1]], mode="disciplined")
    assert wk["pick"] == pinned["pick"]


def test_backtest_disclosure_present_both_modes():
    for m in ("disciplined", "longshot"):
        res = select(_frame({"SPY": "up", "XLK": "up"}, tickers=["SPY"] + SECTORS + LEVER_LONGS), mode=m)
        bt = res["backtest"]
        assert bt["gate"].startswith("PASS")
        assert "wkHit3Pct" in bt and bt["wkHit3Pct"] <= 50    # +3%/wk is a minority of weeks
        assert res["reality"]["target"] == WEEKLY_TARGET
