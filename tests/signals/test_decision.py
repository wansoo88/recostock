"""Tests for the single daily decision (오늘 할 일) builder."""
from signals.decision import REBALANCE_MIN_DELTA, build_decision, format_target_line


def _pf(weights, cash=0.0, eff=1.0, enabled=True, sleeve=0.15):
    return {"weights": weights, "cashWeight": cash, "effExposure": eff,
            "enabled": enabled, "sleeveWeight": sleeve}


def _tc(spy=True, qqq=True, calm=False, dist=5.0):
    return {"coreSpyOn": spy, "coreQqqOn": qqq, "calmBoost": calm, "distPct": dist}


_BLEND = {"SPY": 0.3995, "QQQ": 0.425, "SPXL": 0.0255, "XLV": 0.075, "XLK": 0.075}


def test_hold_when_unchanged():
    prev = {"date": "2026-06-09", "weights": dict(_BLEND), "cashWeight": 0.0}
    d = build_decision(_pf(_BLEND), _tc(), prev=prev, vix=18.9)
    assert d["stance"] == "hold"
    assert d["trades"] == []
    assert "없음" in d["headline"]
    assert d["prevDate"] == "2026-06-09"


def test_rebalance_lists_concrete_trades():
    prev_w = dict(_BLEND, XLV=0.0, XLE=0.075)   # held XLE, target says XLV
    prev = {"date": "2026-06-09", "weights": {k: v for k, v in prev_w.items() if v}, "cashWeight": 0.0}
    d = build_decision(_pf(_BLEND), _tc(), prev=prev)
    assert d["stance"] == "rebalance"
    by_tk = {t["ticker"]: t for t in d["trades"]}
    assert by_tk["XLE"]["action"] == "전량 매도" and by_tk["XLE"]["toPct"] == 0
    assert by_tk["XLV"]["action"] == "신규 매수" and by_tk["XLV"]["toPct"] == 7.5
    assert "2건" in d["headline"]
    # sorted by |delta| descending
    deltas = [abs(t["deltaPct"]) for t in d["trades"]]
    assert deltas == sorted(deltas, reverse=True)


def test_increase_and_decrease_labels():
    prev = {"date": "2026-06-09",
            "weights": {"SPY": 0.40, "QQQ": 0.425, "SPXL": 0.025, "XLV": 0.075, "XLK": 0.075}}
    target = {"SPY": 0.34, "QQQ": 0.425, "SPXL": 0.085, "XLV": 0.075, "XLK": 0.075}
    d = build_decision(_pf(target), _tc(calm=True), prev=prev, vix=14.0)
    by_tk = {t["ticker"]: t for t in d["trades"]}
    assert by_tk["SPXL"]["action"] == "증액"
    assert by_tk["SPY"]["action"] == "감액"


def test_all_cash_when_trend_off():
    prev = {"date": "2026-06-09", "weights": dict(_BLEND)}
    d = build_decision(_pf({}, cash=1.0, eff=0.0), _tc(spy=False, qqq=False), prev=prev, vix=28.0)
    assert d["stance"] == "all_cash"
    assert all(t["action"] == "전량 매도" for t in d["trades"])
    assert "현금화" in d["headline"]


def test_start_without_prior_record():
    d = build_decision(_pf(_BLEND), _tc(), prev=None)
    assert d["stance"] == "start"
    assert all(t["action"] == "신규 매수" for t in d["trades"])
    assert d["prevDate"] is None


def test_prev_source_tagged():
    # tracker record (no source key) → "tracker"; broker snapshot → "broker"
    prev = {"date": "2026-06-09", "weights": dict(_BLEND), "cashWeight": 0.0}
    assert build_decision(_pf(_BLEND), _tc(), prev=prev)["prevSource"] == "tracker"
    prev["source"] = "broker"
    assert build_decision(_pf(_BLEND), _tc(), prev=prev)["prevSource"] == "broker"
    assert build_decision(_pf(_BLEND), _tc(), prev=None)["prevSource"] is None


def test_sub_threshold_drift_is_hold():
    prev_w = dict(_BLEND)
    prev_w["SPY"] = _BLEND["SPY"] + REBALANCE_MIN_DELTA * 0.5
    d = build_decision(_pf(_BLEND), _tc(), prev={"date": "2026-06-09", "weights": prev_w})
    assert d["stance"] == "hold" and d["trades"] == []


def test_why_bullets_cover_trend_vix_sleeve():
    sat = {"pick": ["XLV", "XLK"], "pickAsOf": "2026-06-05"}
    prev = {"date": "2026-06-09", "weights": dict(_BLEND)}
    d = build_decision(_pf(_BLEND), _tc(dist=8.4), prev=prev, satellite=sat, vix=18.9)
    joined = " ".join(d["why"])
    assert "추세 ON" in joined and "+8.4%" in joined
    assert "VIX 18.9" in joined
    assert "XLV·XLK" in joined and "2026-06-05" in joined


def test_calm_boost_and_feardip_bullets():
    prev = {"date": "2026-06-09", "weights": dict(_BLEND)}
    d_calm = build_decision(_pf(_BLEND), _tc(calm=True), prev=prev, vix=14.2)
    assert any("캄-불" in w for w in d_calm["why"])
    d_fd = build_decision(_pf(_BLEND), _tc(), prev=prev, vix=24.0,
                          fear_dip={"isEntry": True, "paper": {"open": 0}})
    assert any("공포매수" in w for w in d_fd["why"])


def test_prices_passthrough_filters_to_target():
    prices = {"SPY": 739.22, "QQQ": 716.07, "SPXL": 264.53, "XLV": 152.65,
              "XLK": 184.18, "XLE": 58.33,           # XLE not in target -> dropped
              "BAD": float("nan")}
    d = build_decision(_pf(_BLEND), _tc(), prev=None, prices=prices)
    assert d["prices"]["SPY"] == 739.22
    assert "XLE" not in d["prices"] and "BAD" not in d["prices"]


def test_stop_proximity_alert_fires_inside_threshold():
    tc = _tc()
    tc["exec"] = {"spy": {"price": 700.0, "stop": 690.0},     # +1.4% room -> alert
                  "qqq": {"price": 716.0, "stop": 621.6}}     # +15% room  -> quiet
    d = build_decision(_pf(_BLEND), tc, prev=None)
    assert len(d["alerts"]) == 1
    assert "SPY" in d["alerts"][0] and "1.4%" in d["alerts"][0]


def test_no_alert_when_stops_far():
    tc = _tc()
    tc["exec"] = {"spy": {"price": 739.22, "stop": 682.10},
                  "qqq": {"price": 716.07, "stop": 621.61}}
    d = build_decision(_pf(_BLEND), tc, prev=None)
    assert d["alerts"] == []


def test_sleeve_rotation_preview_bullet():
    sat = {"pick": ["XLV", "XLK"], "pickAsOf": "2026-06-05", "topK": 2,
           "ranked": [{"ticker": "XLE", "above200": True},   # daily order changed
                      {"ticker": "XLV", "above200": True},
                      {"ticker": "XLK", "above200": True}]}
    d = build_decision(_pf(_BLEND), _tc(), prev=None, satellite=sat)
    assert any("XLE·XLV" in w and "교체 가능" in w for w in d["why"])
    # identical daily order -> no preview noise
    sat2 = dict(sat, ranked=[{"ticker": "XLV", "above200": True},
                             {"ticker": "XLK", "above200": True},
                             {"ticker": "XLE", "above200": True}])
    d2 = build_decision(_pf(_BLEND), _tc(), prev=None, satellite=sat2)
    assert not any("교체 가능" in w for w in d2["why"])


def test_format_target_line_sorted_with_leverage_tag():
    d = build_decision(_pf(_BLEND, cash=0.0), _tc(), prev=None)
    line = format_target_line(d)
    assert line.startswith("QQQ 42.5%")          # weight-descending
    assert "SPXL" in line and "(3x)" in line
    d_cash = build_decision(_pf({"SPY": 0.5}, cash=0.5), _tc(qqq=False), prev=None)
    assert "현금/BIL 50%" in format_target_line(d_cash)
