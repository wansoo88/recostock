"""Phase 1: Information Coefficient (IC) analysis.

IC = Spearman correlation between factor value at T and forward return at T+horizon.
ICIR = mean(IC) / std(IC)  — measures signal consistency.

Rejection rule: |mean IC| < config.IC_MIN_VIABLE (0.01) → factor is buried by costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute_forward_returns(close: pd.DataFrame, horizons: list[int] = [1, 5]) -> dict[int, pd.DataFrame]:
    """Compute forward log-returns for each horizon. No look-ahead: shift(-h) is intentional here
    because we're measuring predictive power, not building a live feature."""
    result = {}
    for h in horizons:
        result[h] = np.log(close / close.shift(1)).shift(-h)
    return result


def compute_ic_series(factor: pd.Series, fwd_return: pd.Series) -> pd.Series:
    """Rolling 63-day cross-ETF Spearman IC at each date.

    factor: MultiIndex (date, ticker) or aligned Series.
    fwd_return: same index as factor.
    Returns a Series indexed by date.
    """
    combined = pd.concat([factor.rename("f"), fwd_return.rename("r")], axis=1).dropna()
    if combined.empty:
        return pd.Series(dtype=float)

    # Group by date and compute cross-sectional Spearman correlation
    def _spearman(group: pd.DataFrame) -> float:
        if len(group) < 3:
            return np.nan
        rho, _ = stats.spearmanr(group["f"], group["r"])
        return rho

    return combined.groupby(level=0).apply(_spearman)


def ic_summary(ic_series: pd.Series, factor_name: str, horizon: int) -> dict:
    """Summarize IC series into key metrics."""
    clean = ic_series.dropna()
    if len(clean) < 20:
        return {"factor": factor_name, "horizon": horizon, "verdict": "INSUFFICIENT DATA"}

    mean_ic = clean.mean()
    std_ic = clean.std()
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    t_stat, p_value = stats.ttest_1samp(clean, 0)

    import config
    viable = abs(mean_ic) >= config.IC_MIN_VIABLE and p_value < 0.05

    return {
        "factor": factor_name,
        "horizon": horizon,
        "mean_ic": round(mean_ic, 4),
        "std_ic": round(std_ic, 4),
        "icir": round(icir, 3),
        "t_stat": round(t_stat, 2),
        "p_value": round(p_value, 4),
        "n_obs": len(clean),
        "verdict": "KEEP" if viable else "REJECT",
    }


def run_full_ic_analysis(
    close_df: pd.DataFrame,
    horizons: list[int] = [1, 5],
) -> pd.DataFrame:
    """Run IC analysis for all factors across all ETFs.

    close_df: columns = tickers, index = dates.
    Returns a summary DataFrame sorted by |mean_ic| descending.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from features.factors import compute_factors

    fwd_returns = compute_forward_returns(close_df, horizons)

    # Build stacked (date, ticker) factor DataFrame
    factor_frames: dict[str, pd.DataFrame] = {}
    for ticker in close_df.columns:
        close = close_df[ticker].dropna()
        if len(close) < 100:
            continue
        f = compute_factors(close)
        for col in f.columns:
            if col not in factor_frames:
                factor_frames[col] = {}
            factor_frames[col][ticker] = f[col]

    # Convert to stacked Series
    stacked_factors: dict[str, pd.Series] = {}
    for fname, ticker_dict in factor_frames.items():
        df = pd.DataFrame(ticker_dict)
        stacked_factors[fname] = df.stack()  # MultiIndex (date, ticker)

    rows = []
    for horizon, fwd_df in fwd_returns.items():
        stacked_fwd = fwd_df.stack()
        stacked_fwd.index.names = ["date", "ticker"]

        for fname, factor_stacked in stacked_factors.items():
            factor_stacked.index.names = ["date", "ticker"]
            ic_s = compute_ic_series(factor_stacked, stacked_fwd)
            rows.append(ic_summary(ic_s, fname, horizon))

    result = pd.DataFrame(rows)
    result["abs_mean_ic"] = result["mean_ic"].abs()
    return result.sort_values("abs_mean_ic", ascending=False).drop(columns="abs_mean_ic")
