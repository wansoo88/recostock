"""H2 definitive test: SH sleeve inside the PRODUCTION blend's cash bucket.

Follow-up to scripts/research_short_v2.py, whose H2 (inverse sleeve while
trend-OFF) beat pure T-bills in 23/26 OFF segments on a SIMPLIFIED regime
definition (SPY<200SMA). Before that result means anything for the live
system it must survive the real thing, so this replays the EXACT production
blend day by day (signals.trend_core/sector_rotation/portfolio via the same
loop as scripts/sweep_blend_goal.py) and, on each day, substitutes

    SH weight   = K * cashWeight        K ∈ {0 (baseline), 0.10, 0.20, 0.30}
    cash weight = (1-K) * cashWeight

so SH only ever lives inside whatever cash the production engine actually
holds that day (engine-off legs, sleeve cash, etc.). Real SH prices, turnover
cost on the SH leg included via the standard |Δw| charge.

PRE-REGISTERED DECISION RULE (same as the goal sweep): a K>0 cell is an
adoption CANDIDATE only if Full(2021+) AND Holdout(2024+) Sharpe >= baseline
AND MDD not worse on both. Even then, nothing changes live before the paper
window (~2026-08-29) — a candidate is queued for re-validation, period
(goal-ceiling-audit rule: knob changes during validation invalidate the
hardwired tracker target).

Run:  python scripts/research_short_v2_production.py
Outputs: console + data/logs/short_v2_production.csv
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

import config
from signals import portfolio, sector_rotation, trend_core
from scripts.sweep_blend_goal import (
    ALLOC_TICKERS,
    FULL_OOS_START,
    HOLDOUT_START,
    TRADING_DAYS,
    feardip_mask,
    load_real,
    perf,
    wf_positive_frac,
)

ONEWAY = config.TOTAL_COST_ROUNDTRIP / 2
K_GRID = [0.0, 0.10, 0.20, 0.30]
OUT = Path("data/logs/short_v2_production.csv")


def run_blend_with_sh(close, vix, cash_yield, fd_mask, k_sh: float) -> pd.Series:
    """sweep_blend_goal.run_blend with K of the daily cash bucket held in SH."""
    idx = close.index
    cash_daily = cash_yield.reindex(idx).ffill().fillna(0.0) / 100.0 / TRADING_DAYS
    avail = [t for t in ALLOC_TICKERS if t in close.columns] + ["SH"]
    fwd = close[avail].pct_change().shift(-1)

    start = trend_core.SMA_WINDOW + 1
    prev_w: dict[str, float] = {}
    last_sat: dict | None = None
    out_dates, out_rets = [], []

    for i in range(start, len(idx) - 1):
        d = idx[i]
        window = close.iloc[: i + 1]
        if last_sat is None or d.dayofweek == 4:
            last_sat = sector_rotation.evaluate(window)
        vix_d = float(vix.asof(d)) if len(vix) else None
        tc = trend_core.evaluate(window, bool(fd_mask.get(d, False)), vix_d)
        blend = portfolio.compose(tc, last_sat)            # shipped sleeve 0.15
        w = dict(blend.get("weights", {}))
        cash_w = float(blend.get("cashWeight", 0.0))
        if k_sh > 0 and cash_w > 0:
            w["SH"] = k_sh * cash_w
            cash_w = (1 - k_sh) * cash_w

        gross = sum(wt * float(fwd[tk].iloc[i]) for tk, wt in w.items()
                    if tk in fwd.columns and not np.isnan(fwd[tk].iloc[i]))
        gross += cash_w * float(cash_daily.iloc[i])
        names = set(prev_w) | set(w)
        turnover = sum(abs(w.get(tk, 0.0) - prev_w.get(tk, 0.0)) for tk in names)
        net = gross - turnover * ONEWAY

        out_dates.append(idx[i + 1])
        out_rets.append(net)
        prev_w = w

    return pd.Series(out_rets, index=pd.DatetimeIndex(out_dates))


def main() -> None:
    close, vix, cash_yield = load_real()
    if "SH" not in close.columns:
        raise RuntimeError("SH price column missing from etf_ohlcv.parquet")
    fd = feardip_mask(close.index, synthetic=False)

    print(f"\nH2 production replay · cost {config.TOTAL_COST_ROUNDTRIP:.2%}/rt · "
          f"span {str(close.index[0])[:10]}→{str(close.index[-1])[:10]}")
    print(f"{'K(SH/cash)':>11}{'Full ret':>10}{'Full Shp':>9}{'Full MDD':>9}"
          f"{'Hold ret':>10}{'Hold Shp':>9}{'Hold MDD':>9}{'WF+':>6}")

    rows, base = [], None
    for k in K_GRID:
        daily = run_blend_with_sh(close, vix, cash_yield, fd, k)
        full = perf(daily[daily.index >= FULL_OOS_START])
        hold = perf(daily[daily.index >= HOLDOUT_START])
        wf_frac, _ = wf_positive_frac(daily[daily.index >= FULL_OOS_START])
        row = {"k_sh": k,
               "full_ret": round(full["ret"], 4), "full_sharpe": round(full["sharpe"], 3),
               "full_mdd": round(full["mdd"], 4),
               "hold_ret": round(hold["ret"], 4), "hold_sharpe": round(hold["sharpe"], 3),
               "hold_mdd": round(hold["mdd"], 4), "wf_pos_frac": round(wf_frac, 3)}
        if k == 0.0:
            base = row
            row["candidate"] = "baseline"
        else:
            ok = (row["full_sharpe"] >= base["full_sharpe"]
                  and row["hold_sharpe"] >= base["hold_sharpe"]
                  and abs(row["full_mdd"]) <= abs(base["full_mdd"])
                  and abs(row["hold_mdd"]) <= abs(base["hold_mdd"]))
            row["candidate"] = "CANDIDATE(queue_post_paper)" if ok else "no"
        rows.append(row)
        print(f"{k:>11.2f}{full['ret']*100:>+9.0f}%{full['sharpe']:>9.2f}"
              f"{full['mdd']*100:>+8.0f}%{hold['ret']*100:>+9.0f}%"
              f"{hold['sharpe']:>9.2f}{hold['mdd']*100:>+8.0f}%"
              f"{wf_frac*100:>5.0f}%   {row['candidate']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    print("REMINDER: any CANDIDATE is queued for AFTER the paper window "
          "(~2026-08-29) and must re-validate then — no live change now.")


if __name__ == "__main__":
    main()
