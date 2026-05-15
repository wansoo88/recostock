"""Critical: verify no look-ahead bias in feature construction.

Run after every new factor is added:
    pytest tests/data/test_lookahead.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import pytest

from features.factors import compute_factors


@pytest.fixture
def sample_close() -> pd.Series:
    """200 days of synthetic close prices with a simple trend + noise."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=200, freq="B")
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, 200)))
    return pd.Series(prices, index=dates, name="TEST")


def test_no_lookahead_single_factor(sample_close):
    """Each factor value at date T must equal the value computed using only data up to T."""
    full_factors = compute_factors(sample_close)

    # Test every 10th row to keep the test fast
    test_dates = full_factors.index[::10]

    for t in test_dates:
        restricted = compute_factors(sample_close[sample_close.index <= t])
        if t not in restricted.index:
            continue
        for col in full_factors.columns:
            full_val = full_factors.loc[t, col]
            restricted_val = restricted.loc[t, col]
            if np.isnan(full_val) and np.isnan(restricted_val):
                continue
            assert abs(full_val - restricted_val) < 1e-8, (
                f"Look-ahead bias detected: column='{col}', date={t.date()}, "
                f"full={full_val:.6f}, restricted={restricted_val:.6f}"
            )


def test_factors_drop_nan_rows(sample_close):
    """Factor DataFrame must have no NaN values (dropna is applied internally)."""
    factors = compute_factors(sample_close)
    assert not factors.isnull().any().any(), "NaN found in factors — check dropna()"


def test_factors_index_subset_of_close(sample_close):
    """Factors index must be a subset of the close price index."""
    factors = compute_factors(sample_close)
    assert set(factors.index).issubset(set(sample_close.index))


def test_costs_constant_is_not_zero():
    """Trading cost constant must never be set to zero."""
    import config
    assert config.TOTAL_COST_ROUNDTRIP > 0, (
        "TOTAL_COST_ROUNDTRIP must be > 0. "
        "Toss Securities charges 0.1%/trade (0.2% roundtrip)."
    )
    assert config.COMMISSION_ROUNDTRIP >= 0.002, (
        "COMMISSION_ROUNDTRIP below Toss Securities actual rate. Do not lower assumptions."
    )
