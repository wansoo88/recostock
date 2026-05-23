#!/usr/bin/env python3
"""Live-flow simulation of the trend-core + fear-dip tilt at 1.3x vs 1.5x.

Each day computes the target exposure (SPY weight + SPXL weight), marks the
portfolio to market with real SPXL prices (decay included), charges turnover
cost on exposure changes, and logs every REBALANCE event. Side-by-side
output for 1.3x (w_spxl=0.15) and 1.5x (w_spxl=0.25).

Run: python scripts/simulate_trend_tilt.py
"""
from __future__ import annotations
import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import config
from scripts.simulate_trading import feardip_trades, close

ONEWAY = config.TOTAL_COST_ROUNDTRIP / 2
START = 10_000.0
# Cash sleeve: ^IRX (13-week T-bill, annualized %) — what BIL/SGOV/short-bond
# parking actually earns. Previously the sim assumed 0% on cash, badly
# understating the strategy (264/1352 days in cash, full OOS).
_IRX = pd.read_parquet("data/raw/macro/yield_2y.parquet").iloc[:, 0].dropna()


def _cash_daily_yield(d):
    v = _IRX.asof(d)
    return (v / 100.0 / 252.0) if pd.notna(v) else 0.0


def fd_mask(idx):
    m = pd.Series(False, index=idx); s = list(idx)
    for t in feardip_trades():
        if t["entry_date"] in idx:
            i = s.index(t["entry_date"])
            for h in range(10):
                if i + h < len(s):
                    m.iloc[i + h] = True
    return m


def target_exposure(core_on: bool, fd_on: bool, w_spxl: float) -> tuple[float, float, str]:
    if core_on and fd_on:
        return (1 - w_spxl, w_spxl, f"상승+공포 → SPY {(1-w_spxl)*100:.0f}% + SPXL {w_spxl*100:.0f}%")
    if core_on:
        return (1.0, 0.0, "상승 → SPY 100%")
    if fd_on:
        return (1.0, 0.0, "하락+공포 → SPY 100% (저점 재진입)")
    return (0.0, 0.0, "하락 → 현금")


def simulate(w_spxl: float, since: pd.Timestamp, label: str, log_events: bool = True):
    spy = close["SPY"].dropna(); spxl = close["SPXL"].dropna()
    idx = spy.index.intersection(spxl.index)
    spy, spxl = spy.reindex(idx), spxl.reindex(idx)
    sret, xret = spy.pct_change(), spxl.pct_change()
    sma = spy.rolling(200).mean()
    core_on = (spy.shift(1) > sma.shift(1)).fillna(False)
    fd = fd_mask(idx)

    idx_s = idx[idx >= since]
    cap = START; peak = cap; mdd = 0.0
    prev_spy = prev_spxl = 0.0
    daily_rets = []
    events = []  # (date, action, cap)
    days_in_regime = {"상승+공포(틸트)": 0, "상승(코어)": 0, "하락+공포(재진입)": 0, "현금": 0}

    for d in idx_s:
        c_on = bool(core_on.loc[d]); f_on = bool(fd.loc[d])
        spy_w, spxl_w, _ = target_exposure(c_on, f_on, w_spxl)
        # turnover cost on change
        turn = abs(spy_w - prev_spy) + abs(spxl_w - prev_spxl)
        # daily P&L: SPY + SPXL legs + cash sleeve (1 - SPY - SPXL) earning IRX
        cash_w = max(0.0, 1.0 - prev_spy - prev_spxl)
        daily = (prev_spy * (sret.get(d, 0) or 0)
                 + prev_spxl * (xret.get(d, 0) or 0)
                 + cash_w * _cash_daily_yield(d)
                 - turn * ONEWAY)
        cap *= (1 + daily); daily_rets.append(daily)
        peak = max(peak, cap); mdd = min(mdd, cap / peak - 1)

        # regime label for stats
        if c_on and f_on: days_in_regime["상승+공포(틸트)"] += 1
        elif c_on:        days_in_regime["상승(코어)"] += 1
        elif f_on:        days_in_regime["하락+공포(재진입)"] += 1
        else:             days_in_regime["현금"] += 1

        # log only on EXPOSURE CHANGE
        if log_events and (spy_w != prev_spy or spxl_w != prev_spxl):
            _, _, note = target_exposure(c_on, f_on, w_spxl)
            events.append((d.date(), note, cap))
        prev_spy, prev_spxl = spy_w, spxl_w

    rets = pd.Series(daily_rets)
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    tot = cap / START - 1
    print(f"\n{'='*92}\n{label}  (시작 ${START:,.0f})\n{'='*92}")
    if log_events:
        print(f"{'날짜':>11}  {'리밸런스 액션':<46}  {'자본($)':>12}")
        for d, note, c in events:
            print(f"{d!s:>11}  {note:<46}  {c:>12,.0f}")
        print("-" * 92)
    print(f"리밸런스 {len(events)}회 · 최종자본 ${cap:,.0f} · 수익률 {tot*100:+.1f}% · "
          f"Sharpe {sharpe:+.2f} · MDD {mdd*100:+.1f}%")
    print("기간 비율:", "  ".join(f"{k} {v}" for k, v in days_in_regime.items()))
    return cap, tot, sharpe, mdd


if __name__ == "__main__":
    print("\n############ 추세코어 + 공포 틸트 실전 시뮬레이션 — 1.3x vs 1.5x ############")
    print("(자본 $10,000 시작 · 실제 SPXL 가격 decay 반영 · 턴오버 비용 반영 · 리밸런스 이벤트 단위 로그)")

    for since, tag in [(pd.Timestamp("2024-01-01"), "HOLDOUT 2024+ (강세장)")]:
        spy = close["SPY"].dropna()
        bh = (spy[spy.index >= since].iloc[-1] / spy[spy.index >= since].iloc[0] - 1) * 100
        print(f"\n##### {tag} #####  [벤치마크 SPY 바이앤홀드 {bh:+.1f}%]")
        c13, t13, s13, m13 = simulate(0.15, since, f"A) 1.3x 틸트 (SPXL 15%)")
        c15, t15, s15, m15 = simulate(0.25, since, f"B) 1.5x 틸트 (SPXL 25%)")

    print("\n##### FULL OOS 2021+ (약세 2022 포함) — 요약만 #####")
    since = pd.Timestamp("2021-01-01")
    spy = close["SPY"].dropna()
    bh = (spy[spy.index >= since].iloc[-1] / spy[spy.index >= since].iloc[0] - 1) * 100
    print(f"[벤치마크 SPY 바이앤홀드 {bh:+.1f}%]")
    simulate(0.15, since, "A) 1.3x 틸트 (Full OOS)", log_events=False)
    simulate(0.25, since, "B) 1.5x 틸트 (Full OOS)", log_events=False)
