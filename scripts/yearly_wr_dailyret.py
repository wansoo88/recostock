"""Yearly WR + daily-equivalent avg return for v1 (paper) and v2 (backtest)."""
from __future__ import annotations
import numpy as np
import pandas as pd

import config

# ── v1 paper-trades (실측) ────────────────────────────────────────────────────
paper = pd.read_parquet("data/paper/trades.parquet")
paper = paper[paper["close_date"].notna()].copy()
paper["close_date"] = pd.to_datetime(paper["close_date"])
paper["open_date"] = pd.to_datetime(paper["open_date"])
paper["pnl_pct"] = paper["pnl_pct"].astype(float)
paper["hold_days"] = (paper["close_date"] - paper["open_date"]).dt.days.clip(lower=1)
paper["year"] = paper["close_date"].dt.year
paper["daily_pnl"] = paper["pnl_pct"] / paper["hold_days"]
paper["win"] = (paper["pnl_pct"] > 0).astype(int)

print("=" * 80)
print("V1 (paper-tracker 실측) — 연도별")
print("=" * 80)
yr_v1 = paper.groupby("year").agg(
    n=("pnl_pct", "count"),
    wr=("win", "mean"),
    avg_trade_pct=("pnl_pct", "mean"),
    avg_daily_pct=("daily_pnl", "mean"),
    avg_hold=("hold_days", "mean"),
    total_pct=("pnl_pct", "sum"),
).round(5)
yr_v1["wr"] = yr_v1["wr"].apply(lambda x: f"{x:.1%}")
yr_v1["avg_trade_pct"] = yr_v1["avg_trade_pct"].apply(lambda x: f"{x*100:+.3f}%")
yr_v1["avg_daily_pct"] = yr_v1["avg_daily_pct"].apply(lambda x: f"{x*100:+.4f}%")
yr_v1["total_pct"] = yr_v1["total_pct"].apply(lambda x: f"{x*100:+.2f}%")
print(yr_v1.to_string())

# ── v2 (walk-forward backtest) — 같은 비용/같은 방식 ─────────────────────────
weekly = pd.read_csv("data/logs/phase3_v2_weekly_equity.csv", parse_dates=[0],
                     index_col=0)
weekly["year"] = weekly.index.year
weekly["daily_net"] = weekly["net"] / 5  # 5-day weekly hold → per-day
weekly["win"] = (weekly["net"] > 0).astype(int)

print("\n" + "=" * 80)
print("V2 (walk-forward 백테스트, 매크로 모델) — 연도별")
print("=" * 80)
# Only count weeks where signal was active (n_active > 0)
yr_v2 = weekly[weekly["n_active"] > 0].groupby("year").agg(
    n_weeks=("net", "count"),
    wr_weeks=("win", "mean"),
    avg_week_pct=("net", "mean"),
    avg_daily_pct=("daily_net", "mean"),
    n_active_avg=("n_active", "mean"),
    total_pct=("net", "sum"),
).round(5)
yr_v2["wr_weeks"] = yr_v2["wr_weeks"].apply(lambda x: f"{x:.1%}")
yr_v2["avg_week_pct"] = yr_v2["avg_week_pct"].apply(lambda x: f"{x*100:+.3f}%")
yr_v2["avg_daily_pct"] = yr_v2["avg_daily_pct"].apply(lambda x: f"{x*100:+.4f}%")
yr_v2["n_active_avg"] = yr_v2["n_active_avg"].apply(lambda x: f"{x:.1f}")
yr_v2["total_pct"] = yr_v2["total_pct"].apply(lambda x: f"{x*100:+.2f}%")
print(yr_v2.to_string())

# ── 가장 최근 표본 강조 ────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("가장 최근 — 세부 (v1 실측 vs v2 백테스트)")
print("=" * 80)

today = pd.Timestamp("2026-05-17")
windows = {
    "Recent 12m": today - pd.Timedelta(days=365),
    "Recent 6m":  today - pd.Timedelta(days=183),
    "Recent 3m":  today - pd.Timedelta(days=92),
    "Recent 30d": today - pd.Timedelta(days=30),
}

print(f"\n  {'window':<13} {'src':<5} {'n':>4} {'WR':>7} {'avg/trade':>11} "
      f"{'avg/day':>11} {'total':>9}")
print("-" * 70)

for label, cutoff in windows.items():
    p_sub = paper[paper["close_date"] >= cutoff]
    if not p_sub.empty:
        n_p = len(p_sub)
        wr_p = (p_sub["pnl_pct"] > 0).mean()
        avg_p = p_sub["pnl_pct"].mean()
        avg_d_p = p_sub["daily_pnl"].mean()
        tot_p = p_sub["pnl_pct"].sum()
        print(f"  {label:<13} v1    {n_p:>4} {wr_p:>6.1%}  "
              f"{avg_p*100:>+10.3f}% {avg_d_p*100:>+10.4f}% {tot_p*100:>+8.2f}%")

    v2_sub = weekly[(weekly.index >= cutoff) & (weekly["n_active"] > 0)]
    if not v2_sub.empty:
        n_v = len(v2_sub)
        wr_v = (v2_sub["net"] > 0).mean()
        avg_v = v2_sub["net"].mean()
        avg_d_v = avg_v / 5
        tot_v = ((1 + v2_sub["net"]).cumprod().iloc[-1] - 1) if n_v else 0
        print(f"  {label:<13} v2    {n_v:>4} {wr_v:>6.1%}  "
              f"{avg_v*100:>+10.3f}% {avg_d_v*100:>+10.4f}% {tot_v*100:>+8.2f}%")
    print()

# ── 동일 컬럼으로 한 줄 요약 ───────────────────────────────────────────────────
print("\n주: ")
print("  v1 = paper/trades.parquet (현재 운영 모델, 실제 backfill 시뮬)")
print("       avg/day = pnl_pct / 보유일수 (포지션 보유 기간 동안 일평균 수익)")
print("  v2 = 야간 학습한 신규 모델 walk-forward backtest")
print("       avg/day = weekly_net / 5 (Friday-to-Friday 5거래일 기준 환산)")
print(f"  비용: 0.25% 왕복 차감 후 (config.TOTAL_COST_ROUNDTRIP = {config.TOTAL_COST_ROUNDTRIP})")
