"""Analysis 1-3: confidence-PnL correlation, recent failure debug, per-ticker 12m.

Driven by user requirement: 최근 데이터 우선, WR 향상이 목표.
Produces concrete numbers to choose between A (cutoff), B (ticker exclusion),
C (retrain) — see paper/trades.parquet schema.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config
from paper.tracker import load_trades

trades = load_trades()
trades = trades[trades["close_date"].notna()].copy()
trades["open_date"] = pd.to_datetime(trades["open_date"])
trades["close_date"] = pd.to_datetime(trades["close_date"])
trades["pnl_pct"] = trades["pnl_pct"].astype(float)
trades["ema_proba"] = trades["ema_proba"].astype(float)
trades["win"] = (trades["pnl_pct"] > 0).astype(int)

CUTOFF_DATE = pd.Timestamp("2025-05-16")  # 최근 12개월 시작
recent = trades[trades["close_date"] >= CUTOFF_DATE].copy()

print("=" * 78)
print("ANALYSIS — 최근 12개월 중심 진단 (n={}, period from {})".format(
    len(recent), CUTOFF_DATE.date()))
print("=" * 78)


# ── ANALYSIS 1 — ema_proba × PnL 상관 ─────────────────────────────────────
print("\n[1] 신뢰도(ema_proba) vs 실제 PnL 상관")
print("-" * 78)


def correlations(df: pd.DataFrame, label: str) -> None:
    if len(df) < 5:
        print(f"  {label}: n={len(df)} (표본 부족)")
        return
    pearson = df["ema_proba"].corr(df["pnl_pct"])
    pearson_win = df["ema_proba"].corr(df["win"])
    spearman = df["ema_proba"].corr(df["pnl_pct"], method="spearman")
    print(f"  {label}  n={len(df):>3}  "
          f"pearson(proba,pnl)={pearson:+.3f}  "
          f"pearson(proba,win)={pearson_win:+.3f}  "
          f"spearman={spearman:+.3f}")


correlations(trades, "전체 5.8년     ")
correlations(recent, "최근 12개월    ")
correlations(trades[trades["close_date"] >= "2025-11-16"], "최근 6개월     ")
correlations(trades[trades["close_date"] < "2024-01-01"], "2020-2023      ")
correlations(trades[(trades["close_date"] >= "2024-01-01") & (trades["close_date"] < "2025-05-16")],
             "2024-2025 봄  ")

print("\n  → pearson(proba,pnl)이 양수면 신뢰도 컷오프 상향이 효과 있음.")
print("    0에 가까우면 신뢰도가 무용 → 모델이 자기 자신감과 결과를 못 맞춤.")


# ── ANALYSIS 2 — 신뢰도 분위수별 WR (최근 12개월) ─────────────────────────
print("\n[2] 최근 12개월 — ema_proba 분위수별 WR")
print("-" * 78)
if len(recent) >= 8:
    recent_sorted = recent.sort_values("ema_proba")
    quartiles = pd.qcut(recent_sorted["ema_proba"], q=4, labels=["Q1(낮음)", "Q2", "Q3", "Q4(높음)"],
                        duplicates="drop")
    by_q = recent_sorted.groupby(quartiles, observed=True).agg(
        n=("pnl_pct", "count"),
        proba_min=("ema_proba", "min"),
        proba_max=("ema_proba", "max"),
        wr=("win", "mean"),
        avg_pnl=("pnl_pct", "mean"),
    ).round(4)
    print(by_q.to_string())
    if "Q1(낮음)" in by_q.index and "Q4(높음)" in by_q.index:
        delta = by_q.loc["Q4(높음)", "wr"] - by_q.loc["Q1(낮음)", "wr"]
        print(f"\n  → WR 차이 Q4-Q1 = {delta:+.2%}  "
              f"(클수록 신뢰도 컷오프 효과 큼)")


# ── ANALYSIS 3 — 컷오프별 시뮬레이션 (A안 정량화) ──────────────────────────
print("\n[3] A안 — ema_proba 컷오프 시뮬레이션 (최근 12개월)")
print("-" * 78)
print(f"  {'cutoff':>8s} {'n':>5s} {'WR':>7s} {'avg_pnl':>9s} {'sharpe':>7s} "
      f"{'mdd%':>7s} {'tot_ret%':>9s}")


def sim_cutoff(df: pd.DataFrame, cutoff: float) -> dict:
    sub = df[df["ema_proba"] >= cutoff]
    if sub.empty:
        return {"n": 0, "wr": 0.0, "avg": 0.0, "sharpe": 0.0,
                "mdd": 0.0, "ret": 0.0}
    weekly = sub.groupby("close_date")["pnl_pct"].mean().sort_index()
    if len(weekly) < 2 or weekly.std() < 1e-10:
        sharpe = 0.0
    else:
        sharpe = weekly.mean() / weekly.std() * np.sqrt(52)
    equity = (1 + weekly).cumprod()
    peak = equity.cummax()
    mdd = float(((equity - peak) / peak).min()) * 100
    return {
        "n": len(sub),
        "wr": float((sub["pnl_pct"] > 0).mean()),
        "avg": float(sub["pnl_pct"].mean()),
        "sharpe": float(sharpe),
        "mdd": mdd,
        "ret": float((equity.iloc[-1] - 1) * 100),
    }


for cut in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    r = sim_cutoff(recent, cut)
    print(f"  {cut:>8.2f} {r['n']:>5d} {r['wr']:>7.2%} "
          f"{r['avg']:>+9.4%} {r['sharpe']:>+7.2f} "
          f"{r['mdd']:>+7.2f} {r['ret']:>+9.2f}")

# 동일하게 5.8년 전체로도 (over-tuning 점검)
print(f"\n  대조 — 5.8년 전체 (over-tuning 점검):")
print(f"  {'cutoff':>8s} {'n':>5s} {'WR':>7s} {'sharpe':>7s}")
for cut in [0.50, 0.55, 0.60, 0.65, 0.70]:
    r = sim_cutoff(trades, cut)
    print(f"  {cut:>8.2f} {r['n']:>5d} {r['wr']:>7.2%} {r['sharpe']:>+7.2f}")


# ── ANALYSIS 4 — 종목별 최근 12개월 (B안 정량화) ──────────────────────────
print("\n[4] B안 — 종목별 최근 12개월")
print("-" * 78)
by_t = recent.groupby("ticker").agg(
    n=("pnl_pct", "count"),
    wr=("win", "mean"),
    avg_pnl=("pnl_pct", "mean"),
    best=("pnl_pct", "max"),
    worst=("pnl_pct", "min"),
    avg_proba=("ema_proba", "mean"),
).round(4).sort_values("avg_pnl", ascending=False)
print(by_t.to_string())

# 종목 제외 시뮬레이션
print(f"\n  종목 제외 시뮬:")
all_pnl = recent.groupby("close_date")["pnl_pct"].mean()
base_sharpe = (all_pnl.mean() / all_pnl.std() * np.sqrt(52)) if all_pnl.std() > 0 else 0
base_wr = (recent["pnl_pct"] > 0).mean()
print(f"    baseline (전 종목)        n={len(recent):>3} WR={base_wr:.2%}  Sharpe={base_sharpe:+.2f}")

# 가장 나쁜 N개 제외
losers_ranked = by_t.sort_values("avg_pnl").index.tolist()
for k in range(1, min(5, len(losers_ranked))):
    excluded = losers_ranked[:k]
    sub = recent[~recent["ticker"].isin(excluded)]
    if sub.empty:
        continue
    weekly = sub.groupby("close_date")["pnl_pct"].mean()
    s = (weekly.mean() / weekly.std() * np.sqrt(52)) if weekly.std() > 0 else 0
    wr_s = (sub["pnl_pct"] > 0).mean()
    print(f"    제외 {','.join(excluded):<28s} n={len(sub):>3} "
          f"WR={wr_s:.2%}  Sharpe={s:+.2f}")


# ── ANALYSIS 5 — 손실 거래 패턴 ────────────────────────────────────────────
print("\n[5] 최근 12개월 손실 거래 디버그")
print("-" * 78)
losers = recent[recent["pnl_pct"] < 0].copy()
winners = recent[recent["pnl_pct"] > 0].copy()
print(f"  losers n={len(losers)},  winners n={len(winners)}")

if not losers.empty and not winners.empty:
    print(f"\n  진입 시점 ema_proba 비교:")
    print(f"    losers  mean={losers['ema_proba'].mean():.3f}  "
          f"median={losers['ema_proba'].median():.3f}  "
          f"min={losers['ema_proba'].min():.3f}")
    print(f"    winners mean={winners['ema_proba'].mean():.3f}  "
          f"median={winners['ema_proba'].median():.3f}  "
          f"min={winners['ema_proba'].min():.3f}")

    print(f"\n  진입 시점 backtest stats 비교 (signal generator가 보고한 wr/payoff/expectancy):")
    for col in ["winrate", "payoff", "expectancy", "sample_n"]:
        if col in recent.columns:
            l = pd.to_numeric(losers[col], errors="coerce").mean()
            w = pd.to_numeric(winners[col], errors="coerce").mean()
            print(f"    {col:12s}  losers={l:.4f}  winners={w:.4f}  diff={w-l:+.4f}")

# 최악 5개 거래
print(f"\n  최악의 5건:")
worst = recent.nsmallest(5, "pnl_pct")[
    ["open_date", "close_date", "ticker", "ema_proba", "pnl_pct"]
]
print(worst.to_string(index=False))

print(f"\n  최고의 5건:")
best = recent.nlargest(5, "pnl_pct")[
    ["open_date", "close_date", "ticker", "ema_proba", "pnl_pct"]
]
print(best.to_string(index=False))

# 보유기간 비교
print(f"\n  보유기간 (losers vs winners):")
losers["hold"] = (losers["close_date"] - losers["open_date"]).dt.days
winners["hold"] = (winners["close_date"] - winners["open_date"]).dt.days
print(f"    losers  mean={losers['hold'].mean():.1f}일  median={losers['hold'].median():.0f}일")
print(f"    winners mean={winners['hold'].mean():.1f}일  median={winners['hold'].median():.0f}일")
