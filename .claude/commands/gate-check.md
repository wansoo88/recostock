Check whether the current backtest results satisfy the phase gate criteria defined in `config.py`.

Usage: /gate-check [tier1|tier2|leverage]

$ARGUMENTS specifies which gate to check (default: tier1).

**Tier 1 (Paper trading entry):**
- Sharpe > `config.TIER1_SHARPE_MIN` (0.7)
- MDD < `config.TIER1_MDD_MAX` (25%)
- OOS Sharpe / IS Sharpe >= `config.TIER1_OOS_IS_RATIO_MIN` (40%)
- Walk-forward: majority of windows positive
- Sample >= `config.TIER1_MIN_TRADING_DAYS` (120 trading days)

**Tier 2 (Live trading entry):**
- Paper trading >= `config.TIER2_PAPER_MONTHS_MIN` (3 months)
- Realized vs backtest gap < `config.TIER2_PAPER_BACKTEST_GAP_MAX` (40%)
- Paper realized Sharpe > `config.TIER2_PAPER_SHARPE_MIN` (0.5)

**Leverage activation:**
- Tier 2 passed
- LEVERAGE_EDUCATION_DONE = true
- Highest-confidence bucket empirically validated (winrate + expectancy positive in paper/live)

Read `data/logs/` for any existing performance logs. Read `backtest/engine.py:BacktestResult.passes_tier1()` for the code implementation.

Report: PASS / FAIL per criterion with actual vs required values.
