"""Tests for the dual-asset vol-adaptive trend-core + fear-dip tilt."""
import numpy as np
import pandas as pd

from signals.trend_core import (evaluate, effective_exposure, TILT_SPXL_WEIGHT,
                                 SPY_SLEEVE_WEIGHT, QQQ_SLEEVE_WEIGHT,
                                 VIX_REGIME_THRESHOLD)


def _close(spy_trend="up", qqq_trend="up", n=260):
    def series(trend):
        return np.linspace(100, 200, n) if trend == "up" else np.linspace(200, 100, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "SPY": pd.Series(series(spy_trend), index=idx),
        "QQQ": pd.Series(series(qqq_trend), index=idx),
    })


def test_dual_uptrend_no_fear():
    r = evaluate(_close("up", "up"), fear_dip_active=False, vix_latest=15.0)
    assert r["spyWeight"] == SPY_SLEEVE_WEIGHT
    assert r["qqqWeight"] == QQQ_SLEEVE_WEIGHT
    assert r["spxlWeight"] == 0.0
    assert r["regime"] == "dual_uptrend"


def test_dual_uptrend_with_fear_tilts_spy_sleeve_only():
    r = evaluate(_close("up", "up"), fear_dip_active=True, vix_latest=15.0)
    expected_spxl = round(SPY_SLEEVE_WEIGHT * TILT_SPXL_WEIGHT, 2)
    expected_spy = round(SPY_SLEEVE_WEIGHT * (1 - TILT_SPXL_WEIGHT), 2)
    assert r["spxlWeight"] == expected_spxl
    assert r["spyWeight"] == expected_spy
    assert r["qqqWeight"] == QQQ_SLEEVE_WEIGHT
    assert r["regime"] == "dual_uptrend_panic"


def test_spy_only_uptrend():
    r = evaluate(_close("up", "down"), fear_dip_active=False, vix_latest=15.0)
    assert r["spyWeight"] == SPY_SLEEVE_WEIGHT
    assert r["qqqWeight"] == 0.0
    assert r["regime"] == "spy_only"


def test_qqq_only_uptrend():
    r = evaluate(_close("down", "up"), fear_dip_active=False, vix_latest=15.0)
    assert r["spyWeight"] == 0.0
    assert r["qqqWeight"] == QQQ_SLEEVE_WEIGHT
    assert r["regime"] == "qqq_only"


def test_both_downtrend_is_cash():
    r = evaluate(_close("down", "down"), fear_dip_active=False, vix_latest=15.0)
    assert r["spyWeight"] == 0.0 and r["qqqWeight"] == 0.0
    assert r["cashWeight"] == 1.0
    assert r["regime"] == "cash"


def test_downtrend_fear_reentry_no_leverage():
    r = evaluate(_close("down", "down"), fear_dip_active=True, vix_latest=15.0)
    assert r["spyWeight"] == SPY_SLEEVE_WEIGHT  # SPY sleeve goes to SPY 100%
    assert r["spxlWeight"] == 0.0               # no leverage in downtrend
    assert r["regime"] == "downtrend_panic"


def test_vol_adaptive_switches_filter_label():
    low = evaluate(_close("up", "up"), False, vix_latest=15.0)
    high = evaluate(_close("up", "up"), False, vix_latest=25.0)
    assert "200SMA" in low["trendFilter"]
    assert "골든크로스" in high["trendFilter"]


def test_effective_exposure_formula():
    assert effective_exposure(0.0) == 1.0
    assert abs(effective_exposure(0.15) - 1.30) < 1e-9
    assert abs(effective_exposure(0.25) - 1.50) < 1e-9


def test_insufficient_history():
    short = pd.DataFrame({"SPY": pd.Series(np.arange(50.0)), "QQQ": pd.Series(np.arange(50.0))})
    assert evaluate(short, False, 15.0)["coreOn"] is None
