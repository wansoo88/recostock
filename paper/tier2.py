"""Phase 4→5 gate: evaluate Tier 2 criteria and determine Phase 5 readiness.

Tier 2 requires:
    - Paper trading >= 3 months (live trades, not backfill)
    - Paper Sharpe > 0.5
    - Backtest/paper Sharpe gap < 40%
    - n_trades >= 120
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import config
from paper.tracker import compute_metrics, load_trades, tier2_gate_check

log = logging.getLogger(__name__)

PHASE3_REPORT = Path("data/logs/phase3_lgbm_report.csv")


def load_backtest_sharpe() -> float:
    if not PHASE3_REPORT.exists():
        return 1.03
    df = pd.read_csv(PHASE3_REPORT)
    row = df[df["section"] == "aggregate_ema_weekly"]
    if row.empty:
        return 1.03
    return float(row.iloc[0]["sharpe"])


def evaluate_tier2(include_backfill: bool = False) -> dict:
    """Evaluate Tier 2 gate. Returns dict with passed, checks, metrics.

    Now also verifies the Tier 1 MDD<25% gate — previously omitted, which
    let MDD-failing systems claim Tier 2 PASS.
    """
    trades = load_trades()
    metrics = compute_metrics(trades, include_backfill=include_backfill)
    backtest_sharpe = load_backtest_sharpe()
    checks = tier2_gate_check(metrics, backtest_sharpe)
    # Tier 1 MDD gate must also hold for Tier 2 to be valid
    mdd_abs = abs(metrics.get("mdd", 0.0))
    checks.append((
        f"Tier 1 carry-over: MDD < {config.TIER1_MDD_MAX:.0%}",
        mdd_abs < config.TIER1_MDD_MAX,
    ))
    passed = all(p for _, p in checks)
    return {
        "passed": passed,
        "checks": checks,
        "metrics": metrics,
        "backtest_sharpe": backtest_sharpe,
    }


def is_phase5_ready(leverage_education_done: bool = False) -> bool:
    """Returns True if all Phase 5 conditions are met."""
    if not leverage_education_done:
        log.debug("Phase 5 blocked: LEVERAGE_EDUCATION_DONE not set")
        return False
    result = evaluate_tier2(include_backfill=False)
    if not result["passed"]:
        log.debug("Phase 5 blocked: Tier 2 gate not passed")
    return result["passed"]
