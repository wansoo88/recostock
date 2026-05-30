"""Tests for the RSI-14 sector rotation satellite."""
import numpy as np
import pandas as pd

from signals.sector_rotation import compute_rsi, evaluate, SECTORS, TOP_K


def _walk(n, drift, seed, noise=0.004):
    """Realistic price path: geometric random walk with a daily drift.

    Pure linear ramps create degenerate RSI (constant diffs -> div-by-zero), so
    tests use noisy drifted walks like real prices. Drift is kept large vs noise
    so the short-term (14d) RSI reflects the intended trend.
    """
    rng = np.random.default_rng(seed)
    rets = drift + rng.normal(0, noise, n)
    return 100 * np.exp(np.cumsum(rets))


def _frame(trends: dict[str, str], n=300):
    """Build a close frame; 'up' = positive drift (high RSI), 'down' = negative."""
    idx = pd.date_range("2023-06-01", periods=n, freq="B")
    drift_map = {"up": 0.004, "down": -0.004, "flat": 0.0}
    data = {}
    for i, t in enumerate(SECTORS):
        data[t] = _walk(n, drift_map[trends.get(t, "flat")], seed=i + 1)
    return pd.DataFrame(data, index=idx)


def test_rsi_basic_bounds():
    s = pd.Series(_walk(200, 0.004, seed=7))
    r = compute_rsi(s).dropna()
    assert (r >= 0).all() and (r <= 100).all()
    # uptrending series -> RSI above the neutral 50 line
    assert r.iloc[-1] > 55


def test_evaluate_ranks_by_rsi_desc():
    # XLK strongest uptrend, XLE strongest downtrend
    df = _frame({"XLK": "up", "XLY": "up", "XLF": "flat",
                 "XLV": "flat", "XLI": "down", "XLE": "down"})
    res = evaluate(df)
    assert res, "should produce a result with enough history"
    rsis = [r["rsi"] for r in res["ranked"]]
    assert rsis == sorted(rsis, reverse=True), "ranked must be RSI-descending"
    # the rising names should outrank the falling ones
    order = [r["ticker"] for r in res["ranked"]]
    assert order.index("XLK") < order.index("XLE")


def test_pick_requires_above_200sma():
    # All uptrends -> top-2 are above 200SMA -> both picked
    df = _frame({t: "up" for t in SECTORS})
    res = evaluate(df)
    assert len(res["pick"]) == TOP_K
    assert res["cashHalf"] == 0
    for tk in res["pick"]:
        row = next(r for r in res["ranked"] if r["ticker"] == tk)
        assert row["above200"] is True


def test_downtrend_top_parks_in_cash():
    # Everything falling -> even the highest-RSI name is below its 200SMA -> cash
    df = _frame({t: "down" for t in SECTORS})
    res = evaluate(df)
    assert res["pick"] == []
    assert res["cashHalf"] == TOP_K


def test_insufficient_history_returns_empty():
    short = pd.DataFrame({t: np.arange(50.0) for t in SECTORS})
    assert evaluate(short) == {}


def test_backtest_disclosure_present():
    df = _frame({t: "up" for t in SECTORS})
    res = evaluate(df)
    bt = res["backtest"]
    assert bt["ic"] == 0.035 and bt["icT"] == 3.5
    assert "blendFull" in bt and "standaloneFull" in bt
