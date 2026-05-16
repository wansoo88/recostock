"""Summarize realized stats from the daily-system paper-trading backfill."""
from __future__ import annotations
import pandas as pd
import numpy as np

COST = 0.0025

df = pd.read_parquet("data/paper/trades.parquet")
df["open_date"] = pd.to_datetime(df["open_date"])
df["close_date"] = pd.to_datetime(df["close_date"])
df["net_pnl"] = df["pnl_pct"] - COST  # apply roundtrip cost

n = len(df)
wins = (df["net_pnl"] > 0).sum()
wr = wins / n
gross_wr = (df["pnl_pct"] > 0).mean()

avg_gross = df["pnl_pct"].mean()
avg_net = df["net_pnl"].mean()
avg_win = df.loc[df["net_pnl"] > 0, "net_pnl"].mean()
avg_loss = df.loc[df["net_pnl"] <= 0, "net_pnl"].mean()
payoff = abs(avg_win / avg_loss) if avg_loss else float("nan")

# Hold period
df["hold_days"] = (df["close_date"] - df["open_date"]).dt.days
avg_hold = df["hold_days"].mean()
median_hold = df["hold_days"].median()

# Sharpe (per trade → annualized assuming avg hold)
ret_std = df["net_pnl"].std()
trades_per_year = 252 / avg_hold
sharpe = (avg_net / ret_std) * np.sqrt(trades_per_year) if ret_std > 0 else float("nan")

# Date range
start = df["open_date"].min()
end = df["close_date"].max()
years = (end - start).days / 365.25

# Per direction
by_dir = df.groupby("direction").agg(
    n=("net_pnl", "count"),
    wr=("net_pnl", lambda s: (s > 0).mean()),
    avg=("net_pnl", "mean"),
    avg_gross=("pnl_pct", "mean"),
).round(4)

# Per ticker
by_ticker = df.groupby("ticker").agg(
    n=("net_pnl", "count"),
    wr=("net_pnl", lambda s: (s > 0).mean()),
    avg_net=("net_pnl", "mean"),
).round(4).sort_values("n", ascending=False)

# By year
df["year"] = df["open_date"].dt.year
by_year = df.groupby("year").agg(
    n=("net_pnl", "count"),
    wr=("net_pnl", lambda s: (s > 0).mean()),
    avg_net=("net_pnl", "mean"),
    sum_net=("net_pnl", "sum"),
).round(4)

print("=" * 70)
print("DAILY SYSTEM — REALIZED STATS (paper backfill)")
print("=" * 70)
print(f"Period:           {start.date()} → {end.date()}  ({years:.2f} years)")
print(f"Cost (roundtrip): {COST:.2%}")
print(f"Source mix:       {dict(df['source'].value_counts())}")
print()
print("─ TRADE-LEVEL (NET, after 0.25% cost) ──────────────────")
print(f"Total trades:     {n}")
print(f"Wins / Losses:    {wins} / {n - wins}")
print(f"Win rate (NET):   {wr:.2%}")
print(f"Win rate (gross): {gross_wr:.2%}  (before cost)")
print(f"Avg WIN  (net):   {avg_win:+.4%}")
print(f"Avg LOSS (net):   {avg_loss:+.4%}")
print(f"Payoff ratio:     {payoff:.3f}")
print(f"Avg gross/trade:  {avg_gross:+.4%}")
print(f"Avg NET  /trade:  {avg_net:+.4%}")
print(f"Avg hold:         {avg_hold:.1f} days  (median {median_hold:.0f})")
print(f"Sharpe (annual):  {sharpe:.3f}")
print()
print("─ BY DIRECTION ─────────────────────────────────────────")
print(by_dir)
print()
print("─ BY TICKER ────────────────────────────────────────────")
print(by_ticker)
print()
print("─ BY YEAR ──────────────────────────────────────────────")
print(by_year)
