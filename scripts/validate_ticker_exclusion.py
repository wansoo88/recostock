"""Validate B-plan ticker exclusion against earlier periods to rule out
data-snooping. The XLV/XLE/XLI/DIA exclusion was identified on recent-12m;
we now check if it would also have been the right call in 2023-2024 (a HOLDOUT
window the recommendation didn't see).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from paper.tracker import load_trades

trades = load_trades()
trades = trades[trades["close_date"].notna()].copy()
trades["close_date"] = pd.to_datetime(trades["close_date"])
trades["pnl_pct"] = trades["pnl_pct"].astype(float)
trades["win"] = (trades["pnl_pct"] > 0).astype(int)


def period_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df.groupby("ticker").agg(
        n=("pnl_pct", "count"),
        wr=("win", "mean"),
        avg=("pnl_pct", "mean"),
    ).round(4)


# Three non-overlapping windows
P_2024_2025 = trades[(trades["close_date"] >= "2024-01-01") &
                     (trades["close_date"] <  "2025-05-16")]
P_RECENT    = trades[trades["close_date"] >= "2025-05-16"]
P_PRE_2024  = trades[trades["close_date"] <  "2024-01-01"]

stats_pre   = period_stats(P_PRE_2024).rename(
    columns={"n": "n_pre", "wr": "wr_pre", "avg": "avg_pre"})
stats_24    = period_stats(P_2024_2025).rename(
    columns={"n": "n_24", "wr": "wr_24", "avg": "avg_24"})
stats_recent = period_stats(P_RECENT).rename(
    columns={"n": "n_R", "wr": "wr_R", "avg": "avg_R"})

joined = pd.concat([stats_pre, stats_24, stats_recent], axis=1)
joined = joined.sort_values("avg_R")
print("=" * 90)
print("티커별 3구간 비교 — exclusion 패턴이 진짜인지 over-tuning인지")
print("=" * 90)
print(f"  P_PRE_2024:   2020-2023  (n={len(P_PRE_2024)})")
print(f"  P_2024_2025:  2024-01 ~ 2025-05  (n={len(P_2024_2025)})")
print(f"  P_RECENT:     2025-05 ~ 현재     (n={len(P_RECENT)})")
print()
print(joined.to_string())

# B안 후보 4종목이 2024-2025 봄에도 약했는지
B_EXCLUDE = ["XLV", "XLE", "XLI", "DIA"]
print(f"\nB안 제외 후보 {B_EXCLUDE} — 구간별 검증:")
print(f"  {'ticker':<7} {'wr_pre':>8} {'wr_24':>8} {'wr_R':>8}  |  "
      f"{'avg_pre':>9} {'avg_24':>9} {'avg_R':>9}  |  verdict")
for t in B_EXCLUDE:
    if t not in joined.index:
        continue
    r = joined.loc[t]
    bad24 = r["wr_24"] < 0.45 if not pd.isna(r["wr_24"]) else False
    badR = r["wr_R"] < 0.45 if not pd.isna(r["wr_R"]) else False
    consistent = bad24 and badR
    verdict = "CONSISTENTLY WEAK (real pattern)" if consistent else \
              "RECENT ONLY (data snoop risk)" if badR else \
              "NOT WEAK"
    print(f"  {t:<7} {r['wr_pre']:>8.2%} {r['wr_24']:>8.2%} {r['wr_R']:>8.2%}  |  "
          f"{r['avg_pre']:>+9.4%} {r['avg_24']:>+9.4%} {r['avg_R']:>+9.4%}  |  {verdict}")


# Apply the 2024-2025 weakness pattern to predict 2025-recent (out-of-sample test)
print(f"\n" + "=" * 90)
print("HOLDOUT 검증 — 2024-2025 봄 데이터로만 약한 종목 선정 → 최근 12개월에 적용")
print("=" * 90)

# Step 1: Using ONLY P_2024_2025, identify weak tickers
weak_by_24 = stats_24[(stats_24["wr_24"] < 0.45) | (stats_24["avg_24"] < 0)].index.tolist()
print(f"  2024-2025 데이터에서 약하다고 판정된 종목 (wr<45% or avg<0): {weak_by_24}")

# Step 2: Apply to RECENT — does excluding them help?
recent = P_RECENT.copy()


def perf(df: pd.DataFrame) -> tuple[int, float, float]:
    if df.empty:
        return 0, 0.0, 0.0
    weekly = df.groupby("close_date")["pnl_pct"].mean()
    sh = (weekly.mean() / weekly.std() * np.sqrt(52)) if weekly.std() > 0 else 0
    return len(df), float((df["pnl_pct"] > 0).mean()), float(sh)


n_b, wr_b, sh_b = perf(recent)
print(f"\n  baseline (전 종목 in RECENT):      n={n_b:>3} WR={wr_b:.2%} Sharpe={sh_b:+.2f}")

n_x, wr_x, sh_x = perf(recent[~recent["ticker"].isin(weak_by_24)])
print(f"  2024-기반 weak 종목 제외:           n={n_x:>3} WR={wr_x:.2%} Sharpe={sh_x:+.2f}")
print(f"  → 진짜 패턴이라면 holdout(최근)에서도 개선되어야 함.")
print(f"  → Sharpe Δ = {sh_x - sh_b:+.2f}  (양수 = pattern is real)")

# Step 3: 더 엄격하게 — 2020-2023 데이터로 약한 종목 → 2024-2025와 최근에 적용 (이중 holdout)
print(f"\n  반대 — 2020-2023 데이터로 weak 선정 → 미래 적용 (의미 없음 점검):")
weak_by_pre = stats_pre[(stats_pre["wr_pre"] < 0.50) | (stats_pre["avg_pre"] < 0)].index.tolist()
print(f"    2020-2023 weak: {weak_by_pre}")
n_x2, wr_x2, sh_x2 = perf(recent[~recent["ticker"].isin(weak_by_pre)])
print(f"    최근에 적용:       n={n_x2:>3} WR={wr_x2:.2%} Sharpe={sh_x2:+.2f}  "
      f"(Δ {sh_x2-sh_b:+.2f})")


# Per-ticker stability (is weakness persistent or one-off?)
print(f"\n" + "=" * 90)
print("티커별 weakness 지속성 — 각 구간에서 WR<45%였던 횟수")
print("=" * 90)
def is_weak(wr) -> bool:
    return (not pd.isna(wr)) and wr < 0.45

stability = []
for t in joined.index:
    r = joined.loc[t]
    weak_count = sum(is_weak(r[c]) for c in ["wr_pre", "wr_24", "wr_R"])
    stability.append((t, r["wr_pre"], r["wr_24"], r["wr_R"], weak_count))
stab_df = pd.DataFrame(stability, columns=["ticker", "wr_pre", "wr_24", "wr_R", "weak_periods"])
stab_df = stab_df.sort_values("weak_periods", ascending=False)
print(stab_df.to_string(index=False))
print(f"\n  weak_periods=3: 항상 약함 → 안전한 제외")
print(f"  weak_periods=2: 2/3 약함 → 제외 권장")
print(f"  weak_periods=1: 단기 변동 → 신중")
