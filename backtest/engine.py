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
    """Expanding-window walk-forward backtest. Applies cost on every position change.

    signals : {-1, 0, +1} per ticker, indexed by date.
              signal[T] = position held from T's close to (T+1)'s close.
    returns : forward simple returns aligned to the same index as signals.
              returns[T] = (close[T+1] - close[T]) / close[T].
    """
    idx = signals.index.intersection(returns.index)
    sig = signals.loc[idx].astype(float).fillna(0)
    ret = returns.loc[idx].fillna(0)
    n = len(idx)

    # Require at least 2 years of IS before first OOS window
    min_is = min(504, n // 3)
    oos_size = (n - min_is) // n_splits
    if oos_size < 20:
        raise ValueError(f"Insufficient data for walk-forward: n={n}, n_splits={n_splits}")

    is_sharpes: list[float] = []
    oos_sharpes: list[float] = []
    oos_positive: list[float] = []
    oos_pnl_parts: list[pd.Series] = []
    total_trades = 0

    for k in range(n_splits):
        is_end = min_is + k * oos_size
        oos_start = is_end
        oos_end = oos_start + oos_size if k < n_splits - 1 else n

        is_pnl = _portfolio_pnl(sig.iloc[:is_end], ret.iloc[:is_end], cost)
        oos_pnl = _portfolio_pnl(sig.iloc[oos_start:oos_end], ret.iloc[oos_start:oos_end], cost)

        if is_pnl.std() > 1e-10:
            is_sharpes.append(compute_sharpe(is_pnl))
        if oos_pnl.std() > 1e-10:
            oos_sharpes.append(compute_sharpe(oos_pnl))
            oos_positive.append(1.0 if oos_pnl.sum() > 0 else 0.0)

        oos_pnl_parts.append(oos_pnl)
        total_trades += int((sig.iloc[oos_start:oos_end].diff().abs() > 0).values.sum())

    combined_oos = pd.concat(oos_pnl_parts) if oos_pnl_parts else pd.Series(dtype=float)
    equity = (1 + combined_oos).cumprod()
    sharpe = compute_sharpe(combined_oos) if combined_oos.std() > 1e-10 else 0.0
    mdd = compute_mdd(equity) if len(equity) > 1 else 0.0
    total_return = float(equity.iloc[-1] - 1) if len(equity) > 0 else 0.0

    avg_is = float(np.mean(is_sharpes)) if is_sharpes else 0.0
    avg_oos = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
    oos_is_ratio = avg_oos / avg_is if avg_is > 1e-6 else 0.0
    wf_positive_pct = float(np.mean(oos_positive)) if oos_positive else 0.0

    return BacktestResult(
        sharpe=sharpe,
        mdd=mdd,
        total_return=total_return,
        n_trades=total_trades,
        oos_is_ratio=oos_is_ratio,
        wf_positive_pct=wf_positive_pct,
    )


def _portfolio_pnl(
    signals: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    cost: float,
) -> pd.Series:
    """Equal-weight portfolio P&L. Cost charged on position changes (half-roundtrip each way).

    signals[T] = position entered at T's close; fwd_returns[T] = return T→T+1.
    """
    idx = signals.index.intersection(fwd_returns.index)
    sig = signals.loc[idx].astype(float).fillna(0)
    ret = fwd_returns.loc[idx].fillna(0)

    gross = (sig * ret).mean(axis=1)

    # Position delta: first day treated as entering from flat
    pos_delta = sig.diff()
    pos_delta.iloc[0] = sig.iloc[0].abs()
    trade_cost = (pos_delta.abs() * (cost / 2)).mean(axis=1)

    return gross - trade_cost


def compute_sharpe(returns: pd.Series, annualization: int = 252) -> float:
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(annualization)


def compute_mdd(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return float(-drawdown.min())
