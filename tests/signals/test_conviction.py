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

def _make_score_result(ema_map: dict[str, float],
                       ema3_map: dict[str, float] | None = None,
                       ema7_map: dict[str, float] | None = None) -> dict[str, dict]:
    """ema_map = EMA-5. If ema3/ema7 not provided, mirror ema_map for v2 confirmation."""
    return {t: {
        "raw_proba": v,
        "ema_proba": v,
        "ema_proba_3": (ema3_map or ema_map).get(t, v),
        "ema_proba_7": (ema7_map or ema_map).get(t, v),
        "signal": 1 if v >= 0.5 else 0,
    } for t, v in ema_map.items()}


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


# ── v2 Multi-EMA confirmation (added 2026-05-17) ──────────────────────────────

def test_v2_multi_ema_rejects_when_ema3_below_threshold():
    """v2: EMA-5 ≥ 0.65 alone is not enough — EMA-3 must also confirm."""
    sr = _make_score_result(
        ema_map={"SPY": 0.70},               # EMA-5 passes
        ema3_map={"SPY": 0.55},              # EMA-3 fails (below 0.65)
        ema7_map={"SPY": 0.68},              # EMA-7 passes
    )
    sigs = select_conviction_signals(
        score_result=sr,
        latest_close=_make_closes({"SPY": 500.0}),
        vix_latest=15.0, spy_close=500, spy_sma200=480,
        active_tickers=ACTIVE,
    )
    # CONVICTION_MULTI_EMA_CONFIRM defaults to True → reject SPY
    assert sigs == []


def test_v2_multi_ema_rejects_when_ema7_below_threshold():
    """EMA-7 must also confirm."""
    sr = _make_score_result(
        ema_map={"SPY": 0.70},
        ema3_map={"SPY": 0.68},
        ema7_map={"SPY": 0.60},              # EMA-7 below threshold
    )
    sigs = select_conviction_signals(
        score_result=sr,
        latest_close=_make_closes({"SPY": 500.0}),
        vix_latest=15.0, spy_close=500, spy_sma200=480,
        active_tickers=ACTIVE,
    )
    assert sigs == []


def test_v2_multi_ema_accepts_when_all_three_confirm():
    """All three EMAs ≥ 0.65 → signal fires."""
    sr = _make_score_result(
        ema_map={"SPY": 0.70},
        ema3_map={"SPY": 0.66},
        ema7_map={"SPY": 0.67},
    )
    sigs = select_conviction_signals(
        score_result=sr,
        latest_close=_make_closes({"SPY": 500.0}),
        vix_latest=15.0, spy_close=500, spy_sma200=480,
        active_tickers=ACTIVE,
    )
    assert len(sigs) == 1
    assert sigs[0].ticker == "SPY"


def test_v2_falls_back_when_ema3_or_ema7_missing(caplog):
    """Legacy callers without ema_proba_3/7 should NOT crash — warn + accept."""
    # Build a score_result without the multi-EMA fields
    sr = {"SPY": {"raw_proba": 0.70, "ema_proba": 0.70, "signal": 1}}
    with caplog.at_level("WARNING"):
        sigs = select_conviction_signals(
            score_result=sr,
            latest_close=_make_closes({"SPY": 500.0}),
            vix_latest=15.0, spy_close=500, spy_sma200=480,
            active_tickers=ACTIVE,
        )
    # Falls back to v1 behavior: signal fires
    assert len(sigs) == 1
    assert any("missing ema_proba_3/7" in rec.message for rec in caplog.records)


# ── v3 Options-regime overlays (added 2026-05-17) ─────────────────────────────

def test_v3_rejects_when_vix_term_backwardation():
    """VIX9D/VIX >= 1.0 means term structure inverted — short-term stress."""
    sr = _make_score_result({"SPY": 0.70})
    sigs = select_conviction_signals(
        score_result=sr,
        latest_close=_make_closes({"SPY": 500.0}),
        vix_latest=15.0, spy_close=500, spy_sma200=480,
        active_tickers=ACTIVE,
        vix9d_latest=16.0,    # 16/15 = 1.067, above threshold 1.0
        skew_z=0.5,
    )
    assert sigs == []


def test_v3_rejects_when_skew_elevated():
    """SKEW z-score >= 1.0 means tail risk priced higher than usual."""
    sr = _make_score_result({"SPY": 0.70})
    sigs = select_conviction_signals(
        score_result=sr,
        latest_close=_make_closes({"SPY": 500.0}),
        vix_latest=15.0, spy_close=500, spy_sma200=480,
        active_tickers=ACTIVE,
        vix9d_latest=13.0,    # 13/15 = 0.867, OK
        skew_z=1.5,           # >= 1.0, elevated tail risk
    )
    assert sigs == []


def test_v3_accepts_when_all_gates_pass():
    """All v3 gates pass → signal fires with backtested expected stats."""
    sr = _make_score_result({"SPY": 0.70})
    sigs = select_conviction_signals(
        score_result=sr,
        latest_close=_make_closes({"SPY": 500.0}),
        vix_latest=15.0, spy_close=500, spy_sma200=480,
        active_tickers=ACTIVE,
        vix9d_latest=13.0,    # contango
        skew_z=0.5,           # tail risk normal
    )
    assert len(sigs) == 1
    s = sigs[0]
    # v3 expected stats (n=20 holdout):
    assert s.winrate == round(config.CONVICTION_EXPECTED_WINRATE, 4)  # 0.70
    assert s.payoff == round(config.CONVICTION_EXPECTED_PAYOFF, 3)     # 1.25
    assert s.is_valid()


def test_v3_degrades_to_v2_when_options_data_missing(caplog):
    """If VIX9D / SKEW unavailable, log warning and skip those gates."""
    sr = _make_score_result({"SPY": 0.70})
    with caplog.at_level("WARNING"):
        sigs = select_conviction_signals(
            score_result=sr,
            latest_close=_make_closes({"SPY": 500.0}),
            vix_latest=15.0, spy_close=500, spy_sma200=480,
            active_tickers=ACTIVE,
            vix9d_latest=None,
            skew_z=None,
        )
    # v3 gates skipped → behaves like v2 → signal fires (Multi-EMA passes since all == 0.70)
    assert len(sigs) == 1
    assert any("VIX9D unavailable" in rec.message for rec in caplog.records)
    assert any("SKEW z-score unavailable" in rec.message for rec in caplog.records)


# ── v4 Bond-vol regime overlay (added 2026-05-17) ─────────────────────────────

def test_v4_rejects_when_move_elevated():
    """MOVE z-score >= 1.0 means bond market vol stress."""
    sr = _make_score_result({"SPY": 0.70})
    sigs = select_conviction_signals(
        score_result=sr,
        latest_close=_make_closes({"SPY": 500.0}),
        vix_latest=15.0, spy_close=500, spy_sma200=480,
        active_tickers=ACTIVE,
        vix9d_latest=13.0,    # contango
        skew_z=0.5,           # tail risk normal
        move_z=1.5,           # bond vol elevated → reject
    )
    assert sigs == []


def test_v4_accepts_when_all_four_gates_pass():
    """v4 = VIX + SPY trend + VIX9D term + SKEW + MOVE — all must pass."""
    sr = _make_score_result({"SPY": 0.70})
    sigs = select_conviction_signals(
        score_result=sr,
        latest_close=_make_closes({"SPY": 500.0}),
        vix_latest=15.0, spy_close=500, spy_sma200=480,
        active_tickers=ACTIVE,
        vix9d_latest=13.0, skew_z=0.5, move_z=-0.3,  # all normal
    )
    assert len(sigs) == 1
    s = sigs[0]
    # v4 expected stats (n=19 holdout):
    assert s.winrate == round(config.CONVICTION_EXPECTED_WINRATE, 4)  # 0.737
    assert s.payoff == round(config.CONVICTION_EXPECTED_PAYOFF, 3)     # 1.25
    assert s.is_valid()


def test_v4_degrades_when_move_missing(caplog):
    """If MOVE unavailable, log warning and skip MOVE gate (degrades to v3)."""
    sr = _make_score_result({"SPY": 0.70})
    with caplog.at_level("WARNING"):
        sigs = select_conviction_signals(
            score_result=sr,
            latest_close=_make_closes({"SPY": 500.0}),
            vix_latest=15.0, spy_close=500, spy_sma200=480,
            active_tickers=ACTIVE,
            vix9d_latest=13.0, skew_z=0.5, move_z=None,
        )
    assert len(sigs) == 1   # v3 gates still pass → signal fires
    assert any("MOVE z-score unavailable" in rec.message for rec in caplog.records)
