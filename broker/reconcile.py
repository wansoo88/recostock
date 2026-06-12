"""Real-account holdings snapshot → decision input + tracker drift report.

The ubuntu server (which alone holds the Toss API keys) runs
scripts/sync_broker_holdings.py daily and commits the sanitized snapshot to
data/broker/holdings.json. This module is the Actions-side consumer:

    load_holdings()  — the snapshot in the same {date, weights, cashWeight}
                       shape as paper.portfolio_tracker.last_weights_before,
                       tagged source="broker"; None when missing or stale
                       (stale real holdings are worse than the tracker
                       assumption — silently diffing against a week-old
                       account would emit wrong trades).
    drift()          — broker vs tracker weight deltas, surfaced in the report
                       so a missed/partial manual execution is caught the next
                       morning instead of silently compounding.

Read-only by construction: this file only ever reads a JSON file.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

HOLDINGS_PATH = Path("data/broker/holdings.json")
# Calendar-day tolerance, matching the spirit of data.collector.data_freshness:
# a snapshot older than this no longer represents "what the user holds now".
MAX_AGE_DAYS = 4
# Display threshold: deltas under this are fractional-share rounding, not a
# missed execution.
DRIFT_WARN_PP = 2.0
_CASH = "CASH"


def load_holdings(path: Path | None = None, today=None,
                  max_age_days: int = MAX_AGE_DAYS) -> dict | None:
    """The sanitized broker snapshot, or None when absent/stale/malformed.

    Returns {date, weights, cashWeight, source="broker"} — drop-in for the
    `prev` argument of signals.decision.build_decision.
    """
    p = Path(path) if path is not None else HOLDINGS_PATH
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        as_of = str(raw["asOf"])
        weights = {str(k).upper(): float(v) for k, v in dict(raw["weights"]).items()}
    except Exception as exc:
        log.warning("Broker holdings unreadable (%s) — falling back to tracker: %s", p, exc)
        return None
    ref = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.utcnow().normalize()
    age = (ref - pd.Timestamp(as_of).normalize()).days
    if age > max_age_days:
        log.warning("Broker holdings stale (%s, %dd old > %dd) — falling back to tracker",
                    as_of, age, max_age_days)
        return None
    cash_w = raw.get("cashWeight")
    if cash_w is None:
        cash_w = max(0.0, 1.0 - sum(weights.values()))
    return {
        "date": as_of,
        "weights": weights,
        "cashWeight": float(cash_w),
        "source": "broker",
    }


def drift(broker_prev: dict | None, tracker_prev: dict | None) -> dict | None:
    """Weight deltas between real holdings and the tracker's assumption.

    None unless both sides exist. Cash is compared as a pseudo-ticker so a
    'forgot to park in BIL' miss shows up too. perTicker is |delta|-descending
    and pre-trimmed to the meaningful rows (>= 0.1%p).
    """
    if not broker_prev or not tracker_prev:
        return None
    bw = dict(broker_prev.get("weights") or {})
    tw = dict(tracker_prev.get("weights") or {})
    bw[_CASH] = float(broker_prev.get("cashWeight", 0.0) or 0.0)
    tw[_CASH] = float(tracker_prev.get("cashWeight", 0.0) or 0.0)
    rows = []
    for tk in sorted(set(bw) | set(tw)):
        b, t = bw.get(tk, 0.0), tw.get(tk, 0.0)
        delta_pp = (b - t) * 100
        if abs(delta_pp) < 0.1:
            continue
        rows.append({"ticker": tk, "brokerPct": round(b * 100, 1),
                     "trackerPct": round(t * 100, 1), "deltaPp": round(delta_pp, 1)})
    rows.sort(key=lambda r: -abs(r["deltaPp"]))
    max_pp = abs(rows[0]["deltaPp"]) if rows else 0.0
    return {
        "maxDeltaPp": round(max_pp, 1),
        "ok": max_pp <= DRIFT_WARN_PP,
        "perTicker": rows[:6],
        "brokerDate": broker_prev.get("date"),
        "trackerDate": tracker_prev.get("date"),
    }
