"""Probability calibration for the model's EMA proba (DISPLAY honesty).

Audit finding (2026-05-22): the raw EMA proba is badly miscalibrated — a
displayed "confidence 0.74" corresponds to an actual ~53-60% directional
win rate, and proba magnitude has ~zero out-of-sample rank correlation with
wins (Spearman -0.02). So the number shown to the user overstates the odds.

This maps raw proba -> empirically-calibrated win probability via an isotonic
fit on the full OOS sample (n=12186 long-ETF daily observations, target =
P(5-day forward return > 0)). Used to DISPLAY an honest win estimate in the
watchlist. It does NOT change the firing threshold — conviction still fires on
raw EMA proba >= 0.65, whose backtest (Friday-K=1, n=19) was validated on raw
values. See scripts/conviction_winrate_buckets.py and the audit notes.

The map is nearly flat 0.45-0.75 (~0.55-0.60): in the actionable range the
model's stated confidence barely differentiates win odds. That is the point —
the honest number is "~57%", not "74%".
"""
from __future__ import annotations

import numpy as np

# (raw_ema_proba, calibrated_win_prob) — isotonic fit, full OOS 2021-2026.
_GRID: list[tuple[float, float]] = [
    (0.45, 0.5539), (0.50, 0.5577), (0.55, 0.5633), (0.60, 0.5697),
    (0.65, 0.5725), (0.70, 0.5958), (0.75, 0.5958), (0.80, 0.7182),
    (0.85, 0.7422),
]
_XS = np.array([g[0] for g in _GRID])
_YS = np.array([g[1] for g in _GRID])

BASE_WIN_RATE = 0.573  # unconditional OOS directional win rate


def calibrated_winrate(proba: float | None) -> float | None:
    """Map a raw EMA proba to its empirically-calibrated directional win prob.

    Piecewise-linear over the isotonic grid, clipped to the fitted range.
    Returns None for None input.
    """
    if proba is None:
        return None
    p = float(np.clip(proba, _XS[0], _XS[-1]))
    return round(float(np.interp(p, _XS, _YS)), 4)
