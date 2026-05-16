"""Reproduce the 1.538 Sharpe claim, compute MDD, and decompose recent weakness.

Uses paper/tracker.py compute_metrics methodology:
- group by close_date → equal-weight portfolio weekly return
- Sharpe = weekly mean / std * sqrt(52)
- MDD on cumulative equity curve
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from paper.tracker import compute_metrics, load_trades

trades = load_trades()
trades = trades[trades["close_date"].notna()].copy()
trades["open_date"] = pd.to_datetime(trades["open_date"])
trades["close_date"] = pd.to_datetime(trades["close_date"])
trades["pnl_pct"] = trades["pnl_pct"].astype(float)


def metrics_for(df: pd.DataFrame, label: str) -> dict:
    """Replicate paper.tracker.compute_metrics on a subset."""
    if df.empty:
        return {"label": label, "n": 0}
    weekly = df.groupby("close_date")["pnl_pct"].mean().sort_index()
    equity = (1 + weekly).cumprod()
    sharpe = (weekly.mean() / weekly.std() * np.sqrt(52)) if weekly.std() > 1e-10 else 0.0
    peak = equity.cummax()
    mdd = float(((equity - peak) / peak).min())
    pnl = df["pnl_pct"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    winrate = len(wins) / len(pnl)
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = abs(losses.mean()) if len(losses) else 0.0
    payoff = avg_win / avg_loss if avg_loss > 1e-10 else 0.0
    return {
        "label": label,
        "n": len(df),
        "n_weeks": len(weekly),
        "sharpe": round(sharpe, 4),
        "mdd_pct": round(mdd * 100, 2),
        "total_ret_pct": round((equity.iloc[-1] - 1) * 100, 2),
        "wr": round(winrate, 4),
        "avg_win": round(avg_win, 5),
        "avg_loss": round(avg_loss, 5),
        "payoff": round(payoff, 3),
    }


# ── 1. Sharpe 1.538 재현 ──────────────────────────────────────────────────
print("=" * 78)
print("PART 1 — Sharpe 1.538 재현 검증")
print("=" * 78)
all_trades = metrics_for(trades, "ALL (backfill + live, all years)")
print(f"  source mix:        {dict(trades['source'].value_counts())}")
for k, v in all_trades.items():
    print(f"  {k:18s} {v}")

# tracker.compute_metrics 결과와 대조
official = compute_metrics(trades, include_backfill=True)
print(f"\n  tracker.compute_metrics(): sharpe={official['sharpe']}, "
      f"mdd={official['mdd']}, n={official['n_trades']}, wr={official['winrate']}")

# ── 2. 시기별 분해 ─────────────────────────────────────────────────────────
print()
print("=" * 78)
print("PART 2 — 시기별 분해 (paper-tracker Sharpe + MDD)")
print("=" * 78)

rows = []
trades["year"] = trades["close_date"].dt.year
for year in sorted(trades["year"].unique()):
    rows.append(metrics_for(trades[trades["year"] == year], f"{year}"))

period_defs = [
    ("2020-2023 (강세 구간)", (trades["close_date"] >= "2020-01-01") &
                              (trades["close_date"] <  "2024-01-01")),
    ("2024-2026 (약세 구간)", (trades["close_date"] >= "2024-01-01")),
    ("최근 12개월",            (trades["close_date"] >= "2025-05-16")),
    ("최근 6개월",             (trades["close_date"] >= "2025-11-16")),
]
for label, mask in period_defs:
    rows.append(metrics_for(trades[mask], label))

cols = ["label", "n", "n_weeks", "sharpe", "mdd_pct", "total_ret_pct",
        "wr", "payoff"]
df_out = pd.DataFrame(rows)[cols]
print(df_out.to_string(index=False))

# ── 3. Tier 1 게이트 ───────────────────────────────────────────────────────
print()
print("=" * 78)
print("PART 3 — Tier 1 게이트 검증 (Sharpe > 0.7, MDD < 25%, n ≥ 120)")
print("=" * 78)

import config

m = all_trades
gates = [
    (f"Sharpe > {config.TIER1_SHARPE_MIN}",
     m["sharpe"] > config.TIER1_SHARPE_MIN, m["sharpe"]),
    (f"MDD < {config.TIER1_MDD_MAX:.0%}",
     abs(m["mdd_pct"] / 100) < config.TIER1_MDD_MAX,
     f"{abs(m['mdd_pct'])}%"),
    (f"n_trades >= {config.TIER1_MIN_TRADING_DAYS}",
     m["n"] >= config.TIER1_MIN_TRADING_DAYS, m["n"]),
]
for name, ok, val in gates:
    mark = "✓ PASS" if ok else "✗ FAIL"
    print(f"  {mark}  {name:30s} = {val}")
all_pass = all(ok for _, ok, _ in gates)
print(f"\n  TIER 1 OVERALL: {'✓ PASS' if all_pass else '✗ FAIL'}")

# ── 4. live vs backfill 분리 ────────────────────────────────────────────────
print()
print("=" * 78)
print("PART 4 — backfill 시뮬레이션 vs 실제 라이브 비교")
print("=" * 78)
for src in ["backfill", "live"]:
    sub = trades[trades["source"] == src]
    if sub.empty:
        print(f"  {src}: (없음)")
        continue
    res = metrics_for(sub, src)
    print(f"  {src:9s} n={res['n']:4d}  sharpe={res['sharpe']:>7.3f}  "
          f"mdd={res['mdd_pct']:>6.2f}%  wr={res['wr']:.2%}  "
          f"payoff={res['payoff']}")

# ── 5. 약화 진단 ────────────────────────────────────────────────────────────
print()
print("=" * 78)
print("PART 5 — 2025-2026 약화 원인 분리")
print("=" * 78)

# 5a. 종목별 변화: 2020-2023 vs 2024-
period_a = trades[trades["close_date"] < "2024-01-01"]
period_b = trades[trades["close_date"] >= "2024-01-01"]

ticker_breakdown = []
for ticker in sorted(trades["ticker"].unique()):
    a = period_a[period_a["ticker"] == ticker]
    b = period_b[period_b["ticker"] == ticker]
    if len(a) < 3 or len(b) < 3:
        continue
    ticker_breakdown.append({
        "ticker": ticker,
        "n_a": len(a), "wr_a": (a["pnl_pct"] > 0).mean(), "avg_a": a["pnl_pct"].mean(),
        "n_b": len(b), "wr_b": (b["pnl_pct"] > 0).mean(), "avg_b": b["pnl_pct"].mean(),
        "wr_diff": (b["pnl_pct"] > 0).mean() - (a["pnl_pct"] > 0).mean(),
        "avg_diff": b["pnl_pct"].mean() - a["pnl_pct"].mean(),
    })
tb = pd.DataFrame(ticker_breakdown).sort_values("avg_diff")
print("종목별 (2020-2023 vs 2024-):")
for col in ["wr_a", "wr_b", "avg_a", "avg_b", "wr_diff", "avg_diff"]:
    tb[col] = tb[col].round(4)
print(tb.to_string(index=False))

# 5b. 보유기간 변화
hold_a = (period_a["close_date"] - period_a["open_date"]).dt.days
hold_b = (period_b["close_date"] - period_b["open_date"]).dt.days
print(f"\n평균 보유일수:")
print(f"  2020-2023:  {hold_a.mean():.1f}일  (중앙값 {hold_a.median():.0f})")
print(f"  2024-2026:  {hold_b.mean():.1f}일  (중앙값 {hold_b.median():.0f})")

# 5c. 진입수 변화 (월별)
month_count = trades.groupby(pd.Grouper(key="open_date", freq="ME")).size()
print(f"\n월별 신규 진입 (마지막 12개월):")
for date, cnt in month_count.tail(12).items():
    print(f"  {date.strftime('%Y-%m'):8s} → {cnt}건")
