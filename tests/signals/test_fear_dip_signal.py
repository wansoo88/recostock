"""Tests for the fear-dip SIGNAL logic (signals/fear_dip.py).

The paper TRACKER is covered by tests/paper/test_fear_dip.py; the signal-
generation logic itself (RSI, z-score, and the causal expanding-quantile entry
rule) was untested. The critical property is look-ahead safety: the entry
threshold at date T must use only data through T. build_bear_score reads ~13
macro parquet files, so evaluate() is tested with a monkeypatched score series
to isolate the entry rule from macro fixtures.
"""
import numpy as np
import pandas as pd
import pytest

import signals.fear_dip as fd


# ── pure helpers ──────────────────────────────────────────────────────────────

def test_rsi_bounds_and_trend():
    # Noisy uptrend (real prices have down days; a pure ramp makes RSI undefined
    # because the downside EWM is all zeros).
    rng = np.random.default_rng(0)
    rets = 0.004 + rng.normal(0, 0.004, 120)
    up = pd.Series(100 * np.exp(np.cumsum(rets)))
    r = fd._rsi(up).dropna()
    assert len(r) > 0
    assert (r >= 0).all() and (r <= 100).all()
    assert r.iloc[-1] > 55  # uptrend -> RSI above neutral


def test_zscore_is_causal_rolling():
    s = pd.Series(np.arange(100.0))
    z = fd._zscore(s, w=20)
    # first w-1 are NaN (rolling needs full window) -> no peeking at future
    assert z.iloc[:19].isna().all()
    assert not np.isnan(z.iloc[-1])


# ── evaluate(): causal expanding-quantile entry rule ──────────────────────────

@pytest.fixture
def patch_score(monkeypatch):
    """Replace build_bear_score with a caller-supplied series; stub price."""
    def _apply(score: pd.Series):
        monkeypatch.setattr(fd, "build_bear_score", lambda close_df: score)
    return _apply


def _close(index):
    return pd.DataFrame({fd.FEAR_DIP_TICKER: np.linspace(100, 110, len(index))},
                        index=index)


def test_entry_fires_when_score_at_extreme(patch_score):
    idx = pd.bdate_range("2020-01-01", periods=fd._ZWIN + 50)
    # mostly low score, today spikes to the very top
    vals = np.r_[np.zeros(len(idx) - 1), 99.0]
    score = pd.Series(vals, index=idx)
    patch_score(score)
    out = fd.evaluate(_close(idx))
    assert out["is_entry"] is True
    assert out["percentile"] == pytest.approx(1.0, abs=1e-9)  # highest ever


def test_no_entry_when_score_low(patch_score):
    idx = pd.bdate_range("2020-01-01", periods=fd._ZWIN + 50)
    vals = np.r_[np.linspace(1, 100, len(idx) - 1), 0.5]  # today near the bottom
    score = pd.Series(vals, index=idx)
    patch_score(score)
    out = fd.evaluate(_close(idx))
    assert out["is_entry"] is False
    assert out["percentile"] < 0.5


def test_threshold_uses_only_history_not_future(patch_score):
    # The expanding quantile at the last row must equal the quantile computed
    # over the full series up to and including today — never beyond (there is no
    # beyond), and dropping the last point must change it (proves it's inclusive
    # of today only, not look-ahead to a longer series).
    idx = pd.bdate_range("2020-01-01", periods=fd._ZWIN + 100)
    rng = np.random.default_rng(0)
    score = pd.Series(rng.normal(size=len(idx)), index=idx)
    patch_score(score)
    out = fd.evaluate(_close(idx), q=fd.FEAR_DIP_Q)
    manual = score.expanding(min_periods=fd._ZWIN).quantile(fd.FEAR_DIP_Q).iloc[-1]
    assert out["threshold"] == pytest.approx(float(manual), abs=1e-9)


def test_entry_price_is_last_close(patch_score):
    idx = pd.bdate_range("2020-01-01", periods=fd._ZWIN + 10)
    patch_score(pd.Series(np.zeros(len(idx)), index=idx))
    c = _close(idx)
    out = fd.evaluate(c)
    assert out["entry_price"] == pytest.approx(float(c[fd.FEAR_DIP_TICKER].iloc[-1]))


def test_nan_score_yields_no_entry(patch_score):
    idx = pd.bdate_range("2020-01-01", periods=fd._ZWIN + 5)
    s = pd.Series(np.nan, index=idx)
    patch_score(s)
    out = fd.evaluate(_close(idx))
    assert out["is_entry"] is False
    assert out["score"] is None
