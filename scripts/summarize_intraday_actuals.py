"""Summarize realized winrate / avg PnL from the live-rule intraday backtest.

Mirrors what the bot would have done if it followed `signals/intraday_generator.py`
on the last 60 days. Output replaces the unvalidated INTRADAY_WINRATE_EST.
"""
from __future__ import annotations

import pandas as pd

COST = 0.0025  # config.TOTAL_COST_ROUNDTRIP

df = pd.read_csv("data/intraday_backtest.csv", parse_dates=["entry_date"])
trades = len(df)
wins = (df["net_pnl"] > 0).sum()
losses = (df["net_pnl"] <= 0).sum()
wr = wins / trades
avg_gross = df["gross_pnl"].mean()
avg_net = df["net_pnl"].mean()
avg_win = df.loc[df["net_pnl"] > 0, "net_pnl"].mean()
avg_loss = df.loc[df["net_pnl"] <= 0, "net_pnl"].mean()
payoff = abs(avg_win / avg_loss) if avg_loss else float("nan")

# Daily PnL (sum of all ticker trades that day)
daily = df.groupby("entry_date")["net_pnl"].sum().sort_index()
daily_mean = daily.mean()
daily_median = daily.median()
daily_std = daily.std()
sessions = len(daily)
positive_days = (daily > 0).sum()
day_wr = positive_days / sessions

# Per-ticker breakdown
per_ticker = (
    df.groupby("ticker")
    .agg(
        n=("net_pnl", "count"),
        wr=("net_pnl", lambda s: (s > 0).mean()),
        avg_net=("net_pnl", "mean"),
        avg_gross=("gross_pnl", "mean"),
    )
    .round(4)
)

# Exit reason mix
exit_mix = df["exit_reason"].value_counts()
exit_wr = df.groupby("exit_reason")["net_pnl"].apply(lambda s: (s > 0).mean()).round(3)
exit_avg = df.groupby("exit_reason")["net_pnl"].mean().round(5)

# By direction
by_dir = df.groupby("direction").agg(
    n=("net_pnl", "count"),
    wr=("net_pnl", lambda s: (s > 0).mean()),
    avg=("net_pnl", "mean"),
).round(4)

print("=" * 70)
print("INTRADAY LIVE-RULE BACKTEST — REALIZED STATS (60d, 9 ETFs)")
print("=" * 70)
print(f"Cost assumption (roundtrip):  {COST:.4%}")
print()
print("─ TRADE-LEVEL ─────────────────────────────────────────────")
print(f"Total trades:        {trades}")
print(f"Wins / Losses:       {wins} / {losses}")
print(f"Win rate:            {wr:.2%}")
print(f"Avg WIN  (net):      {avg_win:+.4%}")
print(f"Avg LOSS (net):      {avg_loss:+.4%}")
print(f"Payoff ratio:        {payoff:.3f}")
print(f"Avg gross / trade:   {avg_gross:+.4%}")
print(f"Avg NET   / trade:   {avg_net:+.4%}")
print()
print("─ DAILY-LEVEL (sum across all 9 tickers per session) ──────")
print(f"Sessions:            {sessions}")
print(f"Mean daily PnL:      {daily_mean:+.4%}")
print(f"Median daily PnL:    {daily_median:+.4%}")
print(f"Daily PnL stdev:     {daily_std:.4%}")
print(f"Positive days:       {positive_days}/{sessions} = {day_wr:.1%}")
print(f"Worst day:           {daily.min():+.4%}")
print(f"Best  day:           {daily.max():+.4%}")
print()
print("─ BY DIRECTION ────────────────────────────────────────────")
print(by_dir)
print()
print("─ BY EXIT REASON ──────────────────────────────────────────")
print(pd.DataFrame({"n": exit_mix, "wr": exit_wr, "avg_net": exit_avg}))
print()
print("─ BY TICKER ───────────────────────────────────────────────")
print(per_ticker)
print()
print("─ RECOMMENDED REPLACEMENTS FOR signals/intraday_generator.py ─")
print(f"INTRADAY_WINRATE_EST  = {wr:.3f}    # was 0.54")
print(f"INTRADAY_AVG_WIN_EST  = {avg_win:.5f}  # was 0.010")
print(f"INTRADAY_AVG_LOSS_EST = {abs(avg_loss):.5f}  # was 0.004")
print(f"# Implied net EV:    {wr*avg_win + (1-wr)*avg_loss:+.4%}/trade  (was assumed +0.106%)")
