"""Portfolio NAV paper tracker — forward out-of-sample validation of the blend.

Unlike the trade-based trackers (paper/tracker.py conviction, paper/fear_dip_tracker.py),
the composed portfolio (signals/portfolio.py) is a DAILY ALLOCATION, not discrete
trades. So validation tracks a NAV curve: each run records the recommended weights
and the realized daily return from holding the PRIOR run's weights into the latest
close, charging turnover cost on the rebalance. After TIER2_PAPER_MONTHS_MIN months
the realized annualized Sharpe is checked against the backtest expectation and the
Tier-2 drift gate (|realized - backtest| / backtest <= TIER2_PAPER_BACKTEST_GAP_MAX).

STRICTLY FORWARD — starts the day tracking begins, NO backfill. Paper validation
must be genuinely out-of-sample (see memory feedback_validation_discipline). The
NAV row for each day is locked in as it actually happened (returns computed from
that run's prices), so later price revisions don't rewrite history.

Records to data/paper/portfolio_nav.parquet. Reports eligibility only — never
auto-trades; graduation to real capital is always a human decision.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

PATH = Path("data/paper/portfolio_nav.parquet")
_ONEWAY = config.TOTAL_COST_ROUNDTRIP / 2.0
_TRADING_DAYS = 252
_COLS = ["date", "weights", "cash_weight", "eff_exposure",
         "gross_ret", "cost", "net_ret", "nav"]


def _target_sharpe() -> float:
    """Backtest blend Sharpe to validate against (single source: sector_rotation)."""
    try:
        from signals.sector_rotation import BACKTEST
        return float(BACKTEST["blendFull"]["sharpe"])
    except Exception:
        return 1.23


def load() -> pd.DataFrame:
    if PATH.exists():
        return pd.read_parquet(PATH)
    return pd.DataFrame(columns=_COLS)


def save(df: pd.DataFrame) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PATH, index=False)


def _cash_daily_yield() -> float:
    """Latest 13-week T-bill (IRX) as a daily simple yield. 0 if unavailable."""
    p = Path("data/raw/macro/irx.parquet")
    if not p.exists():
        return 0.0
    try:
        s = pd.read_parquet(p).iloc[:, 0].dropna()
        return float(s.iloc[-1]) / 100.0 / _TRADING_DAYS if len(s) else 0.0
    except Exception:
        return 0.0


def update(close_df: pd.DataFrame, portfolio: dict,
           today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Record today's allocation and the realized return since the last record.

    close_df       : wide close-price frame (date index, ticker columns).
    portfolio      : output of signals.portfolio.compose (weights/cashWeight/...).
    today          : optional override; defaults to the latest close date so the
                     weights align with actual prices (robust to calendar/data lag).

    Idempotent on the record date: re-running the same day overwrites that row.
    """
    if not portfolio or not portfolio.get("weights"):
        # Nothing investable today — skip silently (e.g. trend off → all cash is
        # still a valid allocation, but compose returns weights then; empty means
        # no portfolio was produced at all).
        return load()

    close_df = close_df.loc[:, ~close_df.columns.duplicated()]
    record_date = pd.Timestamp(today).normalize() if today is not None else \
        pd.Timestamp(close_df.index[-1]).normalize()
    date_str = record_date.date().isoformat()

    w_today = {k: float(v) for k, v in portfolio.get("weights", {}).items() if v}
    cash_today = float(portfolio.get("cashWeight", max(0.0, 1.0 - sum(w_today.values()))))
    eff_today = float(portfolio.get("effExposure", sum(w_today.values())))

    hist = load()
    # Drop any existing row for this date (idempotent re-run).
    if not hist.empty:
        hist = hist[hist["date"] != date_str].reset_index(drop=True)

    gross = cost = net = 0.0
    prev_nav = 1.0
    if not hist.empty:
        prev = hist.iloc[-1]
        prev_date = pd.Timestamp(prev["date"]).normalize()
        prev_nav = float(prev["nav"])
        if prev_date < record_date:
            prev_w = json.loads(prev["weights"]) if isinstance(prev["weights"], str) else dict(prev["weights"])
            prev_cash = float(prev.get("cash_weight", 0.0) or 0.0)
            idx = close_df.index.normalize()
            have_prev = (idx == prev_date).any()
            have_now = (idx == record_date).any()
            if have_prev and have_now:
                p_row = close_df[idx == prev_date].iloc[-1]
                t_row = close_df[idx == record_date].iloc[-1]
                missing = []
                for tk, w in prev_w.items():
                    if tk in close_df.columns and pd.notna(p_row[tk]) and pd.notna(t_row[tk]) and p_row[tk] > 0:
                        gross += w * (float(t_row[tk]) / float(p_row[tk]) - 1.0)
                    elif w:
                        missing.append(tk)
                # cash leg earns the T-bill yield over the calendar gap
                gap_days = max(1, (record_date - prev_date).days)
                gross += prev_cash * (_cash_daily_yield() * gap_days)
                # turnover cost on rebalancing prev -> today (charged today)
                allk = set(prev_w) | set(w_today)
                turn = sum(abs(w_today.get(k, 0.0) - prev_w.get(k, 0.0)) for k in allk)
                cost = turn * _ONEWAY
                net = gross - cost
                if missing:
                    log.warning("Portfolio paper: missing prices for %s on %s/%s — "
                                "treated as 0 return", missing, prev_date.date(), date_str)
            else:
                log.warning("Portfolio paper: price rows for %s or %s unavailable — "
                            "no return recorded for this step", prev_date.date(), date_str)

    nav = prev_nav * (1.0 + net)
    row = {
        "date": date_str,
        "weights": json.dumps(w_today),
        "cash_weight": round(cash_today, 6),
        "eff_exposure": round(eff_today, 4),
        "gross_ret": round(gross, 8),
        "cost": round(cost, 8),
        "net_ret": round(net, 8),
        "nav": round(nav, 8),
    }
    out = pd.concat([hist, pd.DataFrame([row])], ignore_index=True) if not hist.empty \
        else pd.DataFrame([row])
    save(out)
    log.info("Portfolio paper: %s net=%.3f%% nav=%.4f (day %d)",
             date_str, net * 100, nav, len(out))
    return out


def metrics() -> dict:
    """Realized paper metrics + Tier-2 gate progress. Pure reporting."""
    hist = load()
    n = len(hist)
    target = _target_sharpe()
    base = {
        "nDays": n, "months": 0.0, "totalReturn": 0.0, "annReturn": 0.0,
        "annSharpe": 0.0, "mdd": 0.0, "start": None, "last": None,
        "targetSharpe": target, "gap": None,
        "monthsOk": False, "sharpeOk": False, "driftOk": False,
        "passed": False, "status": "no data",
    }
    if n == 0:
        return base
    hist = hist.sort_values("date").reset_index(drop=True)
    start, last = hist["date"].iloc[0], hist["date"].iloc[-1]
    months = (pd.Timestamp(last) - pd.Timestamp(start)).days / 30.4375
    nav = hist["nav"].astype(float)
    # daily net returns excluding the seed row (row 0 has no prior → net 0)
    rets = hist["net_ret"].astype(float).iloc[1:]
    total = float(nav.iloc[-1] - 1.0)
    ann_sharpe = float(rets.mean() / rets.std() * np.sqrt(_TRADING_DAYS)) if len(rets) > 1 and rets.std() > 0 else 0.0
    yrs = max(len(rets), 1) / _TRADING_DAYS
    ann_ret = float(nav.iloc[-1] ** (1 / yrs) - 1.0) if yrs > 0 and nav.iloc[-1] > 0 else 0.0
    eq = nav.values
    mdd = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min()) if len(eq) > 1 else 0.0
    # Two-sided gap BY DESIGN (cf. one-sided paper/tracker.py:tier2_gate_check).
    # This track validates the blend against a SPECIFIC backtest claim
    # (blendFull Sharpe 1.23): realized NAV should CONVERGE toward it, so drift
    # in EITHER direction is suspect — a large outperformance would signal the
    # claim was lucky/leaked, not a pass. The conviction track instead asks "is
    # the live model at least as good as backtest?", where beating it is fine.
    gap = abs(ann_sharpe - target) / target if target else None

    months_ok = months >= config.TIER2_PAPER_MONTHS_MIN
    sharpe_ok = ann_sharpe >= config.TIER2_PAPER_SHARPE_MIN
    drift_ok = (gap is not None and gap <= config.TIER2_PAPER_BACKTEST_GAP_MAX)
    passed = bool(months_ok and sharpe_ok and drift_ok)
    status = ("warming up" if (not months_ok or len(rets) < 20)
              else ("PASS" if passed else "review"))

    return {
        "nDays": n, "months": round(months, 2),
        "totalReturn": round(total, 4), "annReturn": round(ann_ret, 4),
        "annSharpe": round(ann_sharpe, 3), "mdd": round(mdd, 4),
        "start": start, "last": last,
        "targetSharpe": round(target, 3), "gap": round(gap, 3) if gap is not None else None,
        "monthsOk": months_ok, "sharpeOk": sharpe_ok, "driftOk": drift_ok,
        "passed": passed, "status": status,
    }
