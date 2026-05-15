"""Signal generation: compute entry/TP/SL levels and validate expectancy.

A signal is ONLY emitted when winrate, payoff, AND cost-adjusted expectancy are all positive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy.stats as stats

import config


@dataclass
class Signal:
    ticker: str
    name: str
    direction: Literal["long", "short"]
    leverage: int
    entry: float
    tp: float
    sl: float
    # ── Statistics from backtest bucket ──────────────────────────────────────
    winrate: float          # fraction of same-bucket trades that closed positive
    sample_n: int           # number of observations in that bucket
    ci_low: float           # 95% CI lower bound on winrate
    ci_high: float          # 95% CI upper bound on winrate
    payoff: float           # avg_win / avg_loss (loss expressed as positive magnitude)
    expectancy: float       # cost-adjusted: winrate*avg_win - (1-winrate)*avg_loss - costs
    confidence: float       # model output probability [0, 1]
    factors: dict[str, float] = field(default_factory=dict)  # factor -> contribution

    def is_valid(self) -> bool:
        """All three conditions must hold — winrate alone is meaningless."""
        return self.winrate > 0 and self.payoff > 1.0 and self.expectancy > 0


def compute_expectancy(winrate: float, avg_win: float, avg_loss: float) -> float:
    """Cost-adjusted expectancy. avg_loss is a positive magnitude."""
    gross = winrate * avg_win - (1 - winrate) * avg_loss
    return gross - config.TOTAL_COST_ROUNDTRIP


def compute_winrate_ci(wins: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for binomial winrate."""
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = wins / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def compute_levels(
    entry: float,
    direction: Literal["long", "short"],
    tp_pct: float,
    sl_pct: float,
) -> tuple[float, float]:
    """Compute TP and SL prices from entry and percentage distances."""
    if direction == "long":
        tp = entry * (1 + tp_pct)
        sl = entry * (1 - sl_pct)
    else:
        tp = entry * (1 - tp_pct)
        sl = entry * (1 + sl_pct)
    return round(tp, 4), round(sl, 4)
