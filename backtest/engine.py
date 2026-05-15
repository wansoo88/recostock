"""Backtesting engine.

Phase 2: vectorbt for factor/parameter search (fast, vectorized).
Phase 3: Qlib or Backtrader for precise walk-forward validation.

All backtest calls MUST pass TOTAL_COST_ROUNDTRIP from config — never override.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config


@dataclass
class BacktestResult:
    sharpe: float
    mdd: float             # Maximum drawdown as positive fraction
    total_return: float
    n_trades: int
    oos_is_ratio: float    # OOS Sharpe / IS Sharpe
    wf_positive_pct: float # Fraction of walk-forward windows with positive returns

    def passes_tier1(self) -> bool:
        return (
            self.sharpe > config.TIER1_SHARPE_MIN
            and self.mdd < config.TIER1_MDD_MAX
            and self.oos_is_ratio >= config.TIER1_OOS_IS_RATIO_MIN
            and self.wf_positive_pct >= 0.5
            and self.n_trades >= config.TIER1_MIN_TRADING_DAYS
        )


def run_walk_forward(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    n_splits: int = 5,
    cost: float = config.TOTAL_COST_ROUNDTRIP,
) -> BacktestResult:
    """Walk-forward backtest. Applies cost on every round-trip trade.

    Phase 0 stub — raises NotImplementedError until Phase 2.
    """
    raise NotImplementedError("Implement in Phase 2 using vectorbt")


def compute_sharpe(returns: pd.Series, annualization: int = 252) -> float:
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(annualization)


def compute_mdd(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return float(-drawdown.min())
