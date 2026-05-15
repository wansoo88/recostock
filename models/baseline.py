"""Phase 2 baseline: cross-sectional IC-weighted composite signal.

Weights are fixed Phase 1 IC values — no parameter fitting.
Each day, factor values are z-scored across tickers, then combined by IC weight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.factors import compute_factors

# IC weights: Phase 1 results, factors passing BOTH 1d and 5d horizons
# Negative weight = inverse signal (high vol → lower expected return)
IC_WEIGHTS: dict[str, float] = {
    "mom_63d":   0.045,
    "mom_21d":   0.035,
    "zscore_20": 0.033,
    "rsi_14":    0.030,
    "vol_5d":   -0.014,
    "vol_21d":  -0.010,
}


def build_signals(close_df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Cross-sectional IC-weighted signals for each ETF.

    Each day: factor values z-scored across tickers → IC-weighted composite score
    → threshold to {-1, 0, +1}.

    Parameters
    ----------
    close_df : columns=tickers, index=dates
    threshold : minimum absolute composite score to emit a signal (default 0 = always signal)

    Returns
    -------
    DataFrame of int8 {-1, 0, +1}, columns=tickers, index=dates (NaN rows dropped)
    """
    factor_cols: dict[str, dict[str, pd.Series]] = {}
    for ticker in close_df.columns:
        close = close_df[ticker].dropna()
        if len(close) < 100:
            continue
        f = compute_factors(close)
        for fname in IC_WEIGHTS:
            if fname in f.columns:
                factor_cols.setdefault(fname, {})[ticker] = f[fname]

    if not factor_cols:
        return pd.DataFrame()

    factor_dfs: dict[str, pd.DataFrame] = {
        fname: pd.DataFrame(cols) for fname, cols in factor_cols.items()
    }

    common_idx: pd.Index = None  # type: ignore[assignment]
    for df in factor_dfs.values():
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)

    tickers = sorted({t for df in factor_dfs.values() for t in df.columns})
    composite = pd.DataFrame(0.0, index=common_idx, columns=tickers)

    for fname, weight in IC_WEIGHTS.items():
        if fname not in factor_dfs:
            continue
        fmat = factor_dfs[fname].reindex(index=common_idx, columns=tickers)
        # Cross-sectional z-score: normalize across tickers each day
        mu = fmat.mean(axis=1)
        sigma = fmat.std(axis=1).replace(0, np.nan)
        normalized = fmat.sub(mu, axis=0).div(sigma, axis=0).fillna(0.0)
        composite = composite.add(normalized.mul(weight), fill_value=0.0)

    raw = composite.values
    sig_vals = np.where(raw > threshold, 1, np.where(raw < -threshold, -1, 0)).astype(np.int8)
    return pd.DataFrame(sig_vals, index=composite.index, columns=composite.columns)
