"""Unit tests for IC analysis correctness."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import pytest

from features.ic_analysis import compute_forward_returns, compute_ic_series, ic_summary


@pytest.fixture
def synthetic_close() -> pd.DataFrame:
    """Two ETFs with 300 days of price data."""
    np.random.seed(0)
    dates = pd.date_range("2022-01-01", periods=300, freq="B")
    return pd.DataFrame({
        "SPY": 400 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, 300))),
        "QQQ": 300 * np.exp(np.cumsum(np.random.normal(0.0004, 0.012, 300))),
    }, index=dates)


def test_forward_returns_no_lookahead(synthetic_close):
    """Forward return at T uses price at T+h, not beyond."""
    fwd = compute_forward_returns(synthetic_close, horizons=[1])
    # The last row must be NaN (no future data for the final bar)
    assert fwd[1].iloc[-1].isna().all()


def test_ic_series_shape(synthetic_close):
    from features.factors import compute_factors
    fwd = compute_forward_returns(synthetic_close, horizons=[1])[1]

    factor_parts, fwd_parts = {}, {}
    for ticker in synthetic_close.columns:
        close = synthetic_close[ticker]
        f = compute_factors(close)
        factor_parts[ticker] = f["mom_5d"]
        fwd_parts[ticker] = fwd[ticker]

    factor_stacked = pd.DataFrame(factor_parts).stack()
    fwd_stacked = pd.DataFrame(fwd_parts).stack()
    factor_stacked.index.names = ["date", "ticker"]
    fwd_stacked.index.names = ["date", "ticker"]

    ic = compute_ic_series(factor_stacked, fwd_stacked)
    assert len(ic) > 0
    assert ic.between(-1, 1).all()


def test_ic_summary_verdict():
    """KEEP when |mean_ic| >= 0.01 and p < 0.05."""
    # Construct an IC series with strong positive signal
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    strong_ic = pd.Series(np.random.normal(0.05, 0.05, 200), index=dates)
    result = ic_summary(strong_ic, "mock_factor", horizon=1)
    assert result["verdict"] == "KEEP"

    # Weak IC — should reject
    weak_ic = pd.Series(np.random.normal(0.001, 0.10, 200), index=dates)
    result2 = ic_summary(weak_ic, "weak_factor", horizon=1)
    assert result2["verdict"] == "REJECT"
