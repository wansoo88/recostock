"""Broker snapshot loading (freshness gate) + tracker drift report."""
import json

import broker.reconcile as rc


def _write(tmp_path, **over):
    snap = {"asOf": "2026-06-11", "source": "toss-openapi",
            "weights": {"SPY": 0.40, "QQQ": 0.42, "SPXL": 0.03,
                        "XLV": 0.075, "XLK": 0.075},
            "cashWeight": 0.0}
    snap.update(over)
    p = tmp_path / "holdings.json"
    p.write_text(json.dumps(snap), encoding="utf-8")
    return p


def test_load_holdings_fresh(tmp_path):
    p = _write(tmp_path)
    out = rc.load_holdings(p, today="2026-06-12")
    assert out["source"] == "broker"
    assert out["date"] == "2026-06-11"
    assert out["weights"]["SPY"] == 0.40
    assert out["cashWeight"] == 0.0


def test_load_holdings_stale_returns_none(tmp_path):
    p = _write(tmp_path, asOf="2026-06-01")
    assert rc.load_holdings(p, today="2026-06-12") is None


def test_load_holdings_missing_or_malformed(tmp_path):
    assert rc.load_holdings(tmp_path / "nope.json", today="2026-06-12") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert rc.load_holdings(bad, today="2026-06-12") is None


def test_load_holdings_derives_cash_weight(tmp_path):
    p = _write(tmp_path, weights={"SPY": 0.9}, cashWeight=None)
    snap = json.loads(p.read_text(encoding="utf-8"))
    del snap["cashWeight"]
    p.write_text(json.dumps(snap), encoding="utf-8")
    out = rc.load_holdings(p, today="2026-06-12")
    assert abs(out["cashWeight"] - 0.1) < 1e-9


def test_drift_flags_missed_execution():
    broker_prev = {"date": "2026-06-11", "weights": {"SPY": 0.40, "QQQ": 0.42},
                   "cashWeight": 0.18}
    tracker_prev = {"date": "2026-06-11",
                    "weights": {"SPY": 0.40, "QQQ": 0.42, "XLV": 0.075, "XLK": 0.075},
                    "cashWeight": 0.03}
    d = rc.drift(broker_prev, tracker_prev)
    assert d["ok"] is False
    assert d["maxDeltaPp"] == 15.0                      # cash 18% vs 3%
    tickers = {r["ticker"]: r["deltaPp"] for r in d["perTicker"]}
    assert tickers["CASH"] == 15.0
    assert tickers["XLV"] == -7.5 and tickers["XLK"] == -7.5


def test_drift_rounding_noise_is_ok():
    a = {"date": "d", "weights": {"SPY": 0.401, "QQQ": 0.419}, "cashWeight": 0.18}
    b = {"date": "d", "weights": {"SPY": 0.40, "QQQ": 0.42}, "cashWeight": 0.18}
    d = rc.drift(a, b)
    assert d["ok"] is True and d["maxDeltaPp"] <= rc.DRIFT_WARN_PP


def test_drift_none_when_either_side_missing():
    assert rc.drift(None, {"weights": {}}) is None
    assert rc.drift({"weights": {}}, None) is None
