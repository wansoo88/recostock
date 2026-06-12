"""Open-execution realism check — does the blend claim survive T+1-open fills?

Every backtest, the paper tracker, and the +124%/1.23 claim assume execution
AT THE SIGNAL-DAY CLOSE (T). Reality: the signal lands 22:00 KST (pre-open
ET) and the user fills at the NEXT OPEN (T+1). The overnight gap between
close T and open T+1 is systematically risky for a trend follower — gaps tend
to jump in the trend's direction before the fill.

This replays the EXACT production blend (same loop as sweep_blend_goal /
research_short_v2_production) ONCE, and from the same daily weight stream
realizes two return series:

  close-exec (baseline): w(T) earns close_T → close_{T+1}        (status quo)
  open-exec  (realistic): prev weights earn the overnight gap
                          close_T → open_{T+1}, then w(T) earns
                          open_{T+1} → close_{T+1}

On days with no rebalance the two are identical, so the entire difference is
implementation shortfall on trade days. Costs are identical (|Δw| × one-way),
charged on the realization day in both streams.

Interpretation rule (pre-registered): this is a MEASUREMENT, not a knob sweep.
  * shortfall small (Full Sharpe drop ≤ ~0.05): claim robust — record and move on.
  * shortfall material: the Tier-2 checkpoint (~2026-08-29) must judge the
    realized track against the OPEN-exec expectation, not the close-exec one.

Run:  python scripts/research_open_execution.py
Outputs: console + data/logs/open_execution_shortfall.csv
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
    perf,
    wf_positive_frac,
)

ONEWAY = config.TOTAL_COST_ROUNDTRIP / 2
OUT = Path("data/logs/open_execution_shortfall.csv")


def load_with_open():
    raw = pd.read_parquet("data/raw/etf_ohlcv.parquet")
    close, open_ = raw["Close"], raw["Open"]
    vix = pd.read_parquet("data/raw/macro/vix.parquet").iloc[:, 0].dropna()
    cash_yield = pd.read_parquet("data/raw/macro/yield_2y.parquet").iloc[:, 0].dropna()
    return close, open_, vix, cash_yield


def replay_both(close, open_, vix, cash_yield, fd_mask):
    """One production replay → (close_exec_daily, open_exec_daily, n_fallback)."""
    idx = close.index
    cash_daily = cash_yield.reindex(idx).ffill().fillna(0.0) / 100.0 / TRADING_DAYS
    avail = [t for t in ALLOC_TICKERS if t in close.columns]

    start = trend_core.SMA_WINDOW + 1
    prev_w: dict[str, float] = {}
    last_sat: dict | None = None
    dates, cc_rets, oc_rets = [], [], []
    n_fallback = 0

    for i in range(start, len(idx) - 1):
        d = idx[i]
        window = close.iloc[: i + 1]
        if last_sat is None or d.dayofweek == 4:
            last_sat = sector_rotation.evaluate(window)
        vix_d = float(vix.asof(d)) if len(vix) else None
        tc = trend_core.evaluate(window, bool(fd_mask.get(d, False)), vix_d)
        blend = portfolio.compose(tc, last_sat)
        w = blend.get("weights", {})
        cash_w = float(blend.get("cashWeight", 0.0))

        gross_cc = gross_oc = 0.0
        for tk in set(prev_w) | set(w):
            if tk not in avail:
                continue
            c0, c1 = float(close[tk].iloc[i]), float(close[tk].iloc[i + 1])
            o1 = float(open_[tk].iloc[i + 1]) if tk in open_.columns else np.nan
            if np.isnan(c0) or np.isnan(c1) or c0 <= 0:
                continue
            w_new, w_old = w.get(tk, 0.0), prev_w.get(tk, 0.0)
            gross_cc += w_new * (c1 / c0 - 1.0)
            if np.isnan(o1) or o1 <= 0:
                # no usable open — fall back to close-exec for this ticker
                gross_oc += w_new * (c1 / c0 - 1.0)
                if abs(w_new - w_old) > 1e-12:
                    n_fallback += 1
            else:
                # overnight on yesterday's holdings, intraday on today's target
                gross_oc += w_old * (o1 / c0 - 1.0) + w_new * (c1 / o1 - 1.0)
        gross_cc += cash_w * float(cash_daily.iloc[i])
        gross_oc += cash_w * float(cash_daily.iloc[i])

        turnover = sum(abs(w.get(tk, 0.0) - prev_w.get(tk, 0.0))
                       for tk in set(prev_w) | set(w))
        cost = turnover * ONEWAY
        dates.append(idx[i + 1])
        cc_rets.append(gross_cc - cost)
        oc_rets.append(gross_oc - cost)
        prev_w = w

    di = pd.DatetimeIndex(dates)
    return pd.Series(cc_rets, index=di), pd.Series(oc_rets, index=di), n_fallback


def main() -> None:
    close, open_, vix, cash_yield = load_with_open()
    fd = feardip_mask(close.index, synthetic=False)
    cc, oc, n_fb = replay_both(close, open_, vix, cash_yield, fd)

    print(f"\nOpen-execution shortfall · cost {config.TOTAL_COST_ROUNDTRIP:.2%}/rt · "
          f"span {str(close.index[0])[:10]}→{str(close.index[-1])[:10]} · "
          f"open-fallbacks {n_fb}")
    rows = []
    print(f"{'window':>9}{'exec':>7}{'ret':>9}{'Sharpe':>8}{'MDD':>7}{'WF+':>6}")
    for label, start in (("Full21+", FULL_OOS_START), ("Hold24+", HOLDOUT_START)):
        for name, s in (("close", cc), ("open", oc)):
            sl = s[s.index >= start]
            p = perf(sl)
            wf, _ = wf_positive_frac(sl)
            print(f"{label:>9}{name:>7}{p['ret']*100:>+8.0f}%{p['sharpe']:>8.2f}"
                  f"{p['mdd']*100:>+6.0f}%{wf*100:>5.0f}%")
            rows.append({"window": label, "exec": name,
                         "ret": round(p["ret"], 4), "sharpe": round(p["sharpe"], 3),
                         "mdd": round(p["mdd"], 4), "wf_pos": round(wf, 3)})
        d_sh = rows[-1]["sharpe"] - rows[-2]["sharpe"]
        d_rt = rows[-1]["ret"] - rows[-2]["ret"]
        print(f"{'':>9}{'Δ(open-close)':>14}: ret {d_rt*100:+.1f}%p · Sharpe {d_sh:+.3f}")

    # yearly shortfall — where does the gap drag concentrate?
    diff = oc - cc
    by_year = ((1 + diff).groupby(diff.index.year).prod() - 1) * 100
    print("\nyearly shortfall (open − close, %p):")
    print(by_year.round(2).to_string())
    rows.append({"window": "yearly_shortfall_pp", "exec": "open-close",
                 **{str(y): round(v, 3) for y, v in by_year.items()}})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
