"""Tests for volatility-targeted position sizing."""
import numpy as np
import pandas as pd

from signals.sizing import position_size_pct, MAX_SIZE, MIN_SIZE, TARGET_ANN_VOL


def _series(daily_vol, n=60):
    rng = np.random.default_rng(0)
    rets = rng.normal(0, daily_vol, n)
    return pd.Series(100 * np.cumprod(1 + rets))


def test_insufficient_history():
    out = position_size_pct(pd.Series([100, 101, 102]))
    assert out["sizePct"] is None


def test_low_vol_gets_capped_high():
    # very low vol -> target/vol large -> capped at MAX
    out = position_size_pct(_series(0.002))
    assert out["sizePct"] == MAX_SIZE


def test_high_vol_shrinks():
    # high vol (leveraged-like) -> small size
    low = position_size_pct(_series(0.008))["sizePct"]
    high = position_size_pct(_series(0.04))["sizePct"]
    assert high < low
    assert MIN_SIZE <= high <= MAX_SIZE


def test_size_within_bounds():
    for dv in [0.005, 0.01, 0.02, 0.05]:
        s = position_size_pct(_series(dv))["sizePct"]
        assert MIN_SIZE <= s <= MAX_SIZE
