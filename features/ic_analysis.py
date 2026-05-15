"""Phase 1: Information Coefficient (IC) analysis. numpy/pandas only — no scipy.

IC = Spearman correlation between factor value at T and forward return at T+horizon.
ICIR = mean(IC) / std(IC)  — measures signal consistency.

Rejection rule: |mean IC| < config.IC_MIN_VIABLE (0.01) → factor is buried by costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── p-value via normal approximation (valid for n > 30) ──────────────────────

def _normal_cdf(z: float) -> float:
    """Standard normal CDF using Abramowitz & Stegun erf approximation."""
    t = 1.0 / (1.0 + 0.3275911 * abs(z / np.sqrt(2)))
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf_val = 1.0 - poly * np.exp(-(z / np.sqrt(2)) ** 2)
    cdf = 0.5 * (1.0 + (erf_val if z >= 0 else -erf_val))
    return float(np.clip(cdf, 0.0, 1.0))


def _two_sided_pvalue(t_stat: float) -> float:
    """Two-sided p-value from t-statistic using normal approximation (n > 30)."""
    return 2.0 * (1.0 - _normal_cdf(abs(t_stat)))


# ── Core functions ────────────────────────────────────────────────────────────

def compute_forward_returns(close: pd.DataFrame, horizons: list[int] = [1, 5]) -> dict[int, pd.DataFrame]:
    """Forward log-returns at each horizon. shift(-h) is intentional: measuring predictability."""
    return {h: np.log(close / close.shift(1)).shift(-h) for h in horizons}


def compute_ic_series(factor: pd.Series, fwd_return: pd.Series) -> pd.Series:
    """Cross-sectional Spearman IC at each date (across ETFs).

    Both inputs: MultiIndex (date, ticker).
    Returns Series indexed by date.
    """
    combined = pd.concat([factor.rename("f"), fwd_return.rename("r")], axis=1).dropna()
    if combined.empty:
        return pd.Series(dtype=float)

    def _spearman(group: pd.DataFrame) -> float:
        if len(group) < 3:
            return np.nan
        return float(group["f"].corr(group["r"], method="spearman"))

    return combined.groupby(level=0).apply(_spearman)


def ic_summary(ic_series: pd.Series, factor_name: str, horizon: int) -> dict:
    """Summarize IC series into key metrics with KEEP/REJECT verdict."""
    clean = ic_series.dropna()
    if len(clean) < 20:
        return {
            "factor": factor_name, "horizon": horizon,
            "mean_ic": np.nan, "std_ic": np.nan, "icir": np.nan,
            "t_stat": np.nan, "p_value": np.nan, "n_obs": len(clean),
            "verdict": "INSUFFICIENT DATA",
        }

    n = len(clean)
    mean_ic = float(clean.mean())
    std_ic = float(clean.std())
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else 0.0
    p_value = _two_sided_pvalue(t_stat)

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
        "n_obs": n,
        "verdict": "KEEP" if viable else "REJECT",
    }


def run_full_ic_analysis(close_df: pd.DataFrame, horizons: list[int] = [1, 5]) -> pd.DataFrame:
    """Run IC analysis for all factors across all ETFs.

    close_df: columns = tickers, index = dates.
    Returns a summary DataFrame sorted by |mean_ic| descending.
    """
    from features.factors import compute_factors

    fwd_returns = compute_forward_returns(close_df, horizons)

    # Build per-factor stacked (date, ticker) Series
    factor_frames: dict[str, dict[str, pd.Series]] = {}
    for ticker in close_df.columns:
        close = close_df[ticker].dropna()
        if len(close) < 100:
            continue
        f = compute_factors(close)
        for col in f.columns:
            factor_frames.setdefault(col, {})[ticker] = f[col]

    stacked_factors: dict[str, pd.Series] = {}
    for fname, ticker_dict in factor_frames.items():
        stacked = pd.DataFrame(ticker_dict).stack()
        stacked.index.names = ["date", "ticker"]
        stacked_factors[fname] = stacked

    rows = []
    for horizon, fwd_df in fwd_returns.items():
        stacked_fwd = fwd_df.stack()
        stacked_fwd.index.names = ["date", "ticker"]
        for fname, factor_stacked in stacked_factors.items():
            ic_s = compute_ic_series(factor_stacked, stacked_fwd)
            rows.append(ic_summary(ic_s, fname, horizon))

    result = pd.DataFrame(rows)
    result["abs_mean_ic"] = result["mean_ic"].abs()
    return result.sort_values("abs_mean_ic", ascending=False).drop(columns="abs_mean_ic")
