"""Regression tests for signal generation gates (signals/generator.py).

This module computes the TP/SL/expectancy/Wilson-CI and the is_valid() gate that
decides whether ANY signal is emitted. It was previously untested despite running
on the live daily path — a silent bug here corrupts every signal. These lock the
contract: a signal is valid ONLY when winrate>0 AND payoff>=MIN_PAYOFF AND
cost-adjusted expectancy>0.
"""
import math

import pytest

import config
from signals.generator import (
    Signal, compute_expectancy, compute_winrate_ci, compute_levels,
)


def _sig(**kw):
    base = dict(ticker="SPY", name="SPDR", direction="long", leverage=1,
                entry=100.0, tp=103.0, sl=99.0, winrate=0.6, sample_n=50,
                ci_low=0.45, ci_high=0.73, payoff=1.5, expectancy=0.01,
                confidence=0.7)
    base.update(kw)
    return Signal(**base)


# ── is_valid: the three-condition gate ────────────────────────────────────────

def test_is_valid_all_conditions_met():
    assert _sig(winrate=0.6, payoff=1.5, expectancy=0.01).is_valid() is True


def test_is_valid_fails_on_zero_winrate():
    assert _sig(winrate=0.0).is_valid() is False


def test_is_valid_fails_on_low_payoff():
    # payoff just below the configured minimum must fail
    assert _sig(payoff=config.MIN_PAYOFF - 0.01).is_valid() is False
    assert _sig(payoff=config.MIN_PAYOFF).is_valid() is True  # boundary inclusive


def test_is_valid_fails_on_nonpositive_expectancy():
    assert _sig(expectancy=0.0).is_valid() is False
    assert _sig(expectancy=-0.001).is_valid() is False


# ── compute_expectancy: cost-adjusted ─────────────────────────────────────────

def test_expectancy_subtracts_roundtrip_cost():
    # gross = 0.6*0.03 - 0.4*0.01 = 0.018 - 0.004 = 0.014; minus cost
    e = compute_expectancy(0.6, 0.03, 0.01)
    assert e == pytest.approx(0.014 - config.TOTAL_COST_ROUNDTRIP, abs=1e-12)


def test_expectancy_can_go_negative_when_cost_dominates():
    # tiny edge fully eaten by cost
    e = compute_expectancy(0.5, 0.001, 0.001)  # gross 0 -> -cost
    assert e == pytest.approx(-config.TOTAL_COST_ROUNDTRIP, abs=1e-12)
    assert e < 0


# ── compute_winrate_ci: Wilson interval ───────────────────────────────────────

def test_ci_zero_sample_returns_zero_zero():
    assert compute_winrate_ci(0, 0) == (0.0, 0.0)


def test_ci_brackets_point_estimate():
    lo, hi = compute_winrate_ci(30, 50)  # p = 0.6
    assert 0.0 <= lo < 0.6 < hi <= 1.0


def test_ci_clamped_to_unit_interval():
    lo, hi = compute_winrate_ci(50, 50)  # p = 1.0
    assert lo >= 0.0 and hi <= 1.0
    lo0, hi0 = compute_winrate_ci(0, 50)  # p = 0.0
    assert lo0 >= 0.0 and hi0 <= 1.0


def test_ci_narrows_with_more_samples():
    _, hi_small = compute_winrate_ci(6, 10)
    lo_small, _ = compute_winrate_ci(6, 10)
    width_small = hi_small - lo_small
    lo_big, hi_big = compute_winrate_ci(600, 1000)  # same p, 100x n
    width_big = hi_big - lo_big
    assert width_big < width_small


# ── compute_levels: TP/SL geometry ────────────────────────────────────────────

def test_levels_long():
    tp, sl = compute_levels(100.0, "long", 0.03, 0.01)
    assert tp == 103.0 and sl == 99.0


def test_levels_short_inverts():
    tp, sl = compute_levels(100.0, "short", 0.03, 0.01)
    assert tp == 97.0 and sl == 101.0


def test_levels_rounding():
    tp, sl = compute_levels(123.456789, "long", 0.03, 0.01)
    # rounded to 4 dp
    assert tp == round(123.456789 * 1.03, 4)
    assert sl == round(123.456789 * 0.99, 4)
