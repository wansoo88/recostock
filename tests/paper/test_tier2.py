"""Tests for the Phase 4->5 Tier-2 gate (paper/tier2.py).

This gate stands between the system and LIVE LEVERAGE: is_phase5_ready must be
False unless education is done AND every Tier-2 check passes (3mo, paper Sharpe,
backtest drift, n_trades) AND the Tier-1 MDD<25% gate holds. Previously untested
despite running in the live pipeline. A bug here could wrongly enable leverage.
"""
import pytest

import config
import paper.tier2 as t2


def _metrics(sharpe=0.8, weeks=16, n_trades=130, mdd=-0.10):
    """A metrics dict that PASSES all gates by default."""
    return {"sharpe": sharpe, "weeks_elapsed": weeks, "n_trades": n_trades,
            "mdd": mdd, "total_return": 0.2, "winrate": 0.6, "payoff": 1.5}


@pytest.fixture
def patch(monkeypatch):
    """Helper to set metrics + backtest sharpe without real parquet/csv."""
    def _apply(metrics, backtest_sharpe=1.03):
        monkeypatch.setattr(t2, "load_trades", lambda: None)
        monkeypatch.setattr(t2, "compute_metrics", lambda trades, include_backfill=False: metrics)
        monkeypatch.setattr(t2, "load_backtest_sharpe", lambda: backtest_sharpe)
    return _apply


def test_all_pass(patch):
    patch(_metrics())
    r = t2.evaluate_tier2()
    assert r["passed"] is True
    assert all(p for _, p in r["checks"])


def test_fails_when_under_three_months(patch):
    patch(_metrics(weeks=8))  # ~1.8 months < 3
    assert t2.evaluate_tier2()["passed"] is False


def test_fails_on_low_sharpe(patch):
    patch(_metrics(sharpe=config.TIER2_PAPER_SHARPE_MIN - 0.01))
    assert t2.evaluate_tier2()["passed"] is False


def test_fails_on_insufficient_trades(patch):
    patch(_metrics(n_trades=config.TIER1_MIN_TRADING_DAYS - 1))
    assert t2.evaluate_tier2()["passed"] is False


def test_fails_on_mdd_breach(patch):
    # MDD worse than the Tier-1 cap must block Tier-2 even if everything else passes
    patch(_metrics(mdd=-(config.TIER1_MDD_MAX + 0.05)))
    r = t2.evaluate_tier2()
    assert r["passed"] is False
    # the MDD check specifically is the failing one
    mdd_checks = [ok for name, ok in r["checks"] if "MDD" in name]
    assert mdd_checks and not all(mdd_checks)


def test_backtest_drift_gap(patch):
    # paper sharpe far below backtest -> gap > 40% -> fail
    patch(_metrics(sharpe=0.55), backtest_sharpe=2.0)  # gap = (2.0-0.55)/2.0 = 72%
    assert t2.evaluate_tier2()["passed"] is False


def test_outperformance_does_not_fail_drift(patch):
    # paper sharpe ABOVE backtest -> one-sided gap = 0 -> drift check passes
    patch(_metrics(sharpe=1.8), backtest_sharpe=1.0)
    r = t2.evaluate_tier2()
    drift = [ok for name, ok in r["checks"] if "gap" in name]
    assert drift and all(drift)


# ── is_phase5_ready: the leverage safety switch ───────────────────────────────

def test_phase5_blocked_without_education(patch):
    patch(_metrics())  # metrics would pass
    assert t2.is_phase5_ready(leverage_education_done=False) is False


def test_phase5_ready_when_education_and_gates_pass(patch):
    patch(_metrics())
    assert t2.is_phase5_ready(leverage_education_done=True) is True


def test_phase5_blocked_when_gates_fail_even_with_education(patch):
    patch(_metrics(sharpe=0.1))  # fails sharpe
    assert t2.is_phase5_ready(leverage_education_done=True) is False
