#!/usr/bin/env python3
"""Phase 4: Paper trading Tier 2 gate check.

Usage:
    python scripts/run_paper_check.py

Loads data/paper/trades.parquet, computes paper performance,
compares with Phase 3 backtest Sharpe, and prints Tier 2 gate result.
"""
from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from paper.tracker import compute_metrics, load_trades, tier2_gate_check

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PHASE3_REPORT = Path("data/logs/phase3_lgbm_report.csv")


def _load_backtest_sharpe() -> float:
    if not PHASE3_REPORT.exists():
        log.warning("phase3_lgbm_report.csv not found — using 1.03 from memory")
        return 1.03
    df = pd.read_csv(PHASE3_REPORT)
    row = df[df["section"] == "aggregate_ema_weekly"]
    if row.empty:
        return 1.03
    return float(row.iloc[0]["sharpe"])


def main() -> None:
    trades = load_trades()

    print("\n" + "=" * 65)
    print("  PHASE 4 — Paper Trading Report")
    print("=" * 65)

    if trades.empty:
        print("\n  No paper trades recorded yet.")
        print("  Paper trading starts automatically on the next Friday")
        print("  when the Phase 4 pipeline runs and valid signals exist.")
        print("=" * 65)
        return

    closed = trades[trades["close_date"].notna()]
    open_pos = trades[trades["close_date"].isna()]

    print(f"\n  Total trades:   {len(trades):>4}")
    print(f"  Closed trades:  {len(closed):>4}")
    print(f"  Open positions: {len(open_pos):>4}")

    if not closed.empty:
        print("\n── Closed Trades ──")
        print(f"{'Open':>10}  {'Ticker':>6}  {'Entry':>8}  {'Exit':>8}  {'Net PnL':>8}")
        print("-" * 50)
        for _, row in closed.sort_values("open_date").iterrows():
            mark = "+" if (row["pnl_pct"] or 0) > 0 else " "
            print(f"  {row['open_date']:>10}  {row['ticker']:>6}  "
                  f"{row['entry_price']:>8.2f}  {row['exit_price']:>8.2f}  "
                  f"{mark}{row['pnl_pct']:>7.2%}")

    if not open_pos.empty:
        print("\n── Open Positions ──")
        print(f"{'Open':>10}  {'Ticker':>6}  {'Entry':>8}  {'Conf':>6}  {'WR':>6}  {'Payoff':>7}")
        print("-" * 55)
        for _, row in open_pos.sort_values("open_date").iterrows():
            print(f"  {row['open_date']:>10}  {row['ticker']:>6}  "
                  f"{row['entry_price']:>8.2f}  {row['ema_proba']:>6.3f}  "
                  f"{row['winrate']:>6.1%}  {row['payoff']:>7.3f}")

    metrics = compute_metrics(trades)

    print("\n── Paper Performance ──")
    print(f"  Weeks elapsed:   {metrics['weeks_elapsed']:>4}")
    print(f"  Closed trades:   {metrics['n_trades']:>4}")
    print(f"  Weekly periods:  {metrics['n_weeks']:>4}")
    print(f"  Total return:    {metrics['total_return']:>+.2%}")
    print(f"  Sharpe (ann):    {metrics['sharpe']:>6.3f}")
    print(f"  MDD:             {metrics['mdd']:>+.2%}")
    print(f"  Winrate:         {metrics['winrate']:>6.1%}")
    print(f"  Avg win:         {metrics['avg_win']:>7.3%}")
    print(f"  Avg loss:        {metrics['avg_loss']:>7.3%}")
    print(f"  Payoff:          {metrics['payoff']:>6.3f}")

    backtest_sharpe = _load_backtest_sharpe()
    print(f"\n  Backtest Sharpe (Phase 3): {backtest_sharpe:.3f}")

    checks = tier2_gate_check(metrics, backtest_sharpe)
    print("\n── Tier 2 Gate Check ──")
    for label, passed in checks:
        print(f"  {'✅' if passed else '❌'}  {label}")

    n_fail = sum(1 for _, p in checks if not p)
    print("\n" + "=" * 65)
    if n_fail == 0:
        print("  VERDICT: ✅ TIER 2 PASS — Phase 5 실거래 진입 가능!")
    else:
        print(f"  VERDICT: ❌ TIER 2 미달 ({n_fail}개 조건 불충족)")
    print("=" * 65)


if __name__ == "__main__":
    main()
