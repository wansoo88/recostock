"""Tests for the trend-core + fear-dip tilt exposure logic."""
import numpy as np
import pandas as pd

from signals.trend_core import evaluate, effective_exposure, TILT_SPXL_WEIGHT, SMA_WINDOW


def _close(trend="up", n=260):
    # up: rising series (price > SMA200); down: falling
    if trend == "up":
        spy = np.linspace(100, 200, n)
    else:
        spy = np.linspace(200, 100, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"SPY": pd.Series(spy, index=idx)})


def test_uptrend_no_fear_is_full_spy():
    r = evaluate(_close("up"), fear_dip_active=False)
    assert r["coreOn"] is True
    assert r["spyWeight"] == 1.0 and r["spxlWeight"] == 0.0
    assert r["effExposure"] == 1.0


def test_uptrend_with_fear_tilts_into_spxl():
    r = evaluate(_close("up"), fear_dip_active=True)
    assert r["spxlWeight"] == round(TILT_SPXL_WEIGHT, 2)
    assert r["effExposure"] > 1.0
    assert r["regime"] == "uptrend_panic"


def test_downtrend_is_cash():
    r = evaluate(_close("down"), fear_dip_active=False)
    assert r["coreOn"] is False
    assert r["spyWeight"] == 0.0 and r["spxlWeight"] == 0.0
    assert r["regime"] == "cash"


def test_downtrend_fear_reentry_no_leverage():
    r = evaluate(_close("down"), fear_dip_active=True)
    assert r["spyWeight"] == 1.0 and r["spxlWeight"] == 0.0  # no leverage in downtrend
    assert r["regime"] == "downtrend_panic"


def test_effective_exposure_formula():
    assert effective_exposure(0.0) == 1.0
    assert abs(effective_exposure(0.15) - 1.30) < 1e-9   # 1.3x
    assert abs(effective_exposure(0.25) - 1.50) < 1e-9   # 1.5x


def test_insufficient_history():
    short = pd.DataFrame({"SPY": pd.Series(np.arange(50.0))})
    assert evaluate(short, False)["coreOn"] is None
