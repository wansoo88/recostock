"""Tests for the portfolio composition layer (engine + RSI sector sleeve)."""
import pytest

from signals.portfolio import compose, LEVER_MULT


def _engine(spy=0.42, spxl=0.07, qqq=0.50):
    """A typical trend-core output (calm-uptrend day)."""
    return {"spyWeight": spy, "spxlWeight": spxl, "qqqWeight": qqq,
            "cashWeight": round(1 - spy - spxl - qqq, 4), "effExposure": spy + spxl*3 + qqq,
            "regime": "dual_uptrend_boost"}


def _sat(pick, top_k=2):
    return {"pick": pick, "topK": top_k, "ranked": [{"ticker": "XLK"}], "cashHalf": top_k - len(pick)}


def test_blend_splits_capital_75_25():
    eng = _engine()
    res = compose(eng, _sat(["XLK", "XLV"]), sleeve_weight=0.25)
    # engine leg scaled to 75%
    assert res["coreWeight"] == 0.75
    assert res["weights"]["SPY"] == pytest.approx(0.42 * 0.75, abs=1e-6)
    assert res["weights"]["QQQ"] == pytest.approx(0.50 * 0.75, abs=1e-6)
    # sleeve leg: 25% split across 2 picks = 12.5% each (plus XLK already? no — engine has no XLK)
    assert res["weights"]["XLK"] == pytest.approx(0.125, abs=1e-6)
    assert res["weights"]["XLV"] == pytest.approx(0.125, abs=1e-6)
    assert res["enabled"] is True


def test_total_capital_not_exceeded():
    eng = _engine()
    res = compose(eng, _sat(["XLK", "XLV"]), sleeve_weight=0.25)
    total = sum(res["weights"].values()) + res["cashWeight"]
    assert total == pytest.approx(1.0, abs=1e-6)


def test_empty_sleeve_slot_becomes_cash():
    # only 1 pick of top-2 -> the other 12.5% slot is cash, not redistributed
    eng = _engine()
    res = compose(eng, _sat(["XLK"], top_k=2), sleeve_weight=0.25)
    assert res["weights"].get("XLK") == pytest.approx(0.125, abs=1e-6)
    # sleeve invested only half of its 25%
    assert res["sleeveInvested"] == pytest.approx(0.125, abs=1e-6)
    total = sum(res["weights"].values()) + res["cashWeight"]
    assert total == pytest.approx(1.0, abs=1e-6)


def test_all_sectors_below_sma_all_sleeve_cash():
    eng = _engine()
    res = compose(eng, _sat([], top_k=2), sleeve_weight=0.25)
    # no sector positions; engine still at 75%
    assert "XLK" not in res["weights"]
    assert res["coreWeight"] == 0.75
    eng_cap = 0.42*0.75 + 0.07*0.75 + 0.50*0.75
    assert res["cashWeight"] == pytest.approx(1 - eng_cap, abs=1e-6)


def test_sleeve_disabled_returns_engine_only():
    eng = _engine()
    res = compose(eng, _sat(["XLK", "XLV"]), sleeve_weight=0.0)
    assert res["coreWeight"] == 1.0
    assert res["weights"]["SPY"] == pytest.approx(0.42, abs=1e-6)
    assert "XLK" not in res["weights"]
    assert res["enabled"] is False


def test_missing_satellite_returns_engine():
    eng = _engine()
    res = compose(eng, None, sleeve_weight=0.25)
    # no satellite data -> sleeve not applied, engine at full (1.0) so it's still investable
    assert res["enabled"] is False
    assert res["coreWeight"] == 1.0
    assert res["weights"]["SPY"] == pytest.approx(0.42, abs=1e-6)


def test_effective_exposure_counts_leverage():
    eng = _engine(spy=0.42, spxl=0.07, qqq=0.50)
    res = compose(eng, _sat(["XLK", "XLV"]), sleeve_weight=0.25)
    # SPXL counts 3x; sectors 1x
    expected = (0.42*0.75)*1 + (0.07*0.75)*LEVER_MULT + (0.50*0.75)*1 + 0.125 + 0.125
    assert res["effExposure"] == pytest.approx(round(expected, 2), abs=0.01)
