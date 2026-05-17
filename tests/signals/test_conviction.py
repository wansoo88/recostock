"""Tests for signals/conviction.py — conviction strategy v1."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import pytest

import config
from signals.conviction import regime_ok, select_conviction_signals


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_score_result(ema_map: dict[str, float]) -> dict[str, dict]:
    return {t: {"raw_proba": v, "ema_proba": v, "signal": 1 if v >= 0.5 else 0}
            for t, v in ema_map.items()}


def _make_closes(price_map: dict[str, float]) -> pd.Series:
    return pd.Series(price_map)


# ── regime_ok ─────────────────────────────────────────────────────────────────

def test_regime_ok_passes_when_vix_low_and_spy_above_sma():
    ok, _ = regime_ok(vix_latest=15.0, spy_close=500.0, spy_sma200=480.0)
    assert ok


def test_regime_ok_fails_on_high_vix():
    ok, reason = regime_ok(vix_latest=25.0, spy_close=500.0, spy_sma200=480.0)
    assert not ok
    assert "VIX" in reason


def test_regime_ok_fails_on_spy_below_sma():
    ok, reason = regime_ok(vix_latest=15.0, spy_close=470.0, spy_sma200=480.0)
    assert not ok
    assert "SPY" in reason


def test_regime_ok_fails_on_missing_vix():
    ok, _ = regime_ok(vix_latest=None, spy_close=500.0, spy_sma200=480.0)
    assert not ok


def test_regime_ok_fails_on_missing_spy_when_required():
    # CONVICTION_REQUIRE_SPY_UPTREND defaults to True
    ok, _ = regime_ok(vix_latest=15.0, spy_close=None, spy_sma200=480.0)
    assert not ok


# ── select_conviction_signals ────────────────────────────────────────────────

ACTIVE = {"SPY", "QQQ", "DIA", "XLK", "XLF", "XLE", "SH", "PSQ", "VXX"}


def test_no_signal_when_regime_fails():
    sr = _make_score_result({"SPY": 0.70})
    sigs = select_conviction_signals(
        score_result=sr,
        latest_close=_make_closes({"SPY": 500}),
        vix_latest=30.0,  # too high
        spy_close=500, spy_sma200=480,
        active_tickers=ACTIVE,
    )
    assert sigs == []


def test_no_signal_when_no_candidate_above_threshold():
    sr = _make_score_result({"SPY": 0.60, "QQQ": 0.55})  # both below 0.65
    sigs = select_conviction_signals(
        score_result=sr,
        latest_close=_make_closes({"SPY": 500, "QQQ": 450}),
        vix_latest=15.0, spy_close=500, spy_sma200=480,
        active_tickers=ACTIVE,
    )
    assert sigs == []


def test_picks_top_1_by_ema_proba():
    sr = _make_score_result({"SPY": 0.70, "QQQ": 0.80, "XLK": 0.66})
    sigs = select_conviction_signals(
        score_result=sr,
        latest_close=_make_closes({"SPY": 500, "QQQ": 450, "XLK": 200}),
        vix_latest=15.0, spy_close=500, spy_sma200=480,
        active_tickers=ACTIVE,
    )
    assert len(sigs) == config.CONVICTION_TOP_K == 1
    assert sigs[0].ticker == "QQQ"  # highest ema_proba


def test_excludes_inverse_and_volatility():
    """Long-only universe: SH/PSQ/VXX must not be picked even with high proba."""
    sr = _make_score_result({"SH": 0.90, "PSQ": 0.85, "VXX": 0.80, "SPY": 0.70})
    sigs = select_conviction_signals(
        score_result=sr,
        latest_close=_make_closes({"SH": 30, "PSQ": 12, "VXX": 50, "SPY": 500}),
        vix_latest=15.0, spy_close=500, spy_sma200=480,
        active_tickers=ACTIVE,
    )
    assert len(sigs) == 1
    assert sigs[0].ticker == "SPY"


def test_fixed_tp_sl_prices():
    sr = _make_score_result({"SPY": 0.70})
    sigs = select_conviction_signals(
        score_result=sr,
        latest_close=_make_closes({"SPY": 500.0}),
        vix_latest=15.0, spy_close=500.0, spy_sma200=480,
        active_tickers=ACTIVE,
    )
    assert len(sigs) == 1
    s = sigs[0]
    # TP = entry * (1 + TP_PCT), SL = entry * (1 - SL_PCT)
    assert s.tp == round(500.0 * (1 + config.CONVICTION_TP_PCT), 4)  # 515.0
    assert s.sl == round(500.0 * (1 - config.CONVICTION_SL_PCT), 4)  # 495.0


def test_signal_passes_is_valid():
    """The conviction signal must satisfy WR>0, payoff>=MIN_PAYOFF, E>0."""
    sr = _make_score_result({"SPY": 0.70})
    sigs = select_conviction_signals(
        score_result=sr,
        latest_close=_make_closes({"SPY": 500.0}),
        vix_latest=15.0, spy_close=500, spy_sma200=480,
        active_tickers=ACTIVE,
    )
    assert len(sigs) == 1
    assert sigs[0].is_valid(), "Conviction signal must pass is_valid()"


def test_skips_inactive_tickers():
    sr = _make_score_result({"SPY": 0.70, "QQQ": 0.85})
    # QQQ not in active set (e.g., phase restriction)
    sigs = select_conviction_signals(
        score_result=sr,
        latest_close=_make_closes({"SPY": 500, "QQQ": 450}),
        vix_latest=15.0, spy_close=500, spy_sma200=480,
        active_tickers={"SPY", "DIA"},
    )
    assert len(sigs) == 1
    assert sigs[0].ticker == "SPY"


def test_skips_when_entry_price_unavailable():
    sr = _make_score_result({"SPY": 0.70})
    sigs = select_conviction_signals(
        score_result=sr,
        latest_close=_make_closes({"QQQ": 450}),  # SPY missing
        vix_latest=15.0, spy_close=500, spy_sma200=480,
        active_tickers=ACTIVE,
    )
    assert sigs == []
