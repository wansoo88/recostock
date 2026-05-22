"""Volatility-targeted position sizing — tells the user HOW MUCH to buy.

Improvement #3. Calibration (signals/calibration.py) showed model proba has no
out-of-sample ranking power, so sizing by "confidence" would be noise. The
honest, standard approach is volatility targeting: size each position so its
risk contribution is roughly constant, which automatically shrinks volatile /
leveraged names (a 3x ETF gets ~1/3 the weight of SPY).

    size% = clip(TARGET_ANN_VOL / realized_ann_vol, MIN, MAX)

Output is a fraction of the capital the user allocates to this system — not of
their whole net worth. Capped well below 100% because the per-trade edge is
thin and samples are small (see audit notes).
"""
from __future__ import annotations

import pandas as pd

TARGET_ANN_VOL = 0.10   # target annualized volatility per position
MAX_SIZE = 0.50         # never put >50% of system capital in one name
MIN_SIZE = 0.05
VOL_WINDOW = 20
_TRADING_DAYS = 252


def position_size_pct(close: pd.Series) -> dict:
    """Return {sizePct, annVol} from the ticker's recent realized volatility.

    sizePct/annVol are None when there isn't enough history.
    """
    rets = close.pct_change().dropna().tail(VOL_WINDOW)
    if len(rets) < 10:
        return {"sizePct": None, "annVol": None}
    ann_vol = float(rets.std() * (_TRADING_DAYS ** 0.5))
    if ann_vol <= 0:
        return {"sizePct": MIN_SIZE, "annVol": round(ann_vol, 4)}
    size = max(MIN_SIZE, min(MAX_SIZE, TARGET_ANN_VOL / ann_vol))
    return {"sizePct": round(size, 3), "annVol": round(ann_vol, 4)}
