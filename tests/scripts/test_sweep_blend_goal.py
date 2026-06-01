"""Network-free coverage for the blend-goal reproduction harness.

The real sweep needs Yahoo data; these tests exercise the plumbing on synthetic
prices (the same path as `--self-test`) so the harness can never silently rot,
and pin the config single-source-of-truth so the live 'goal' knobs can't drift
back into undocumented module defaults.
"""
import numpy as np
import pytest

import config
from scripts import sweep_blend_goal as sg
from signals import portfolio, trend_core


@pytest.fixture
def synth():
    return sg.make_synthetic(seed=3, n=900)


def test_run_blend_returns_finite_daily_series(synth):
    close, vix, irx = synth
    fd = sg.feardip_mask(close.index, synthetic=True)
    strong0 = trend_core.STRONG_SPXL
    try:
        daily = sg.run_blend(close, vix, irx, fd, sleeve_weight=0.15, strong_spxl=0.20)
    finally:
        trend_core.STRONG_SPXL = strong0  # harness mutates the module global
    assert len(daily) > 100
    assert np.isfinite(daily.to_numpy()).all()
    # realisation dates are shifted forward (d -> d+1), so start after warm-up
    assert daily.index[0] > close.index[trend_core.SMA_WINDOW]


def test_perf_and_gate_shape(synth):
    close, vix, irx = synth
    fd = sg.feardip_mask(close.index, synthetic=True)
    strong0 = trend_core.STRONG_SPXL
    try:
        daily = sg.run_blend(close, vix, irx, fd, 0.15, 0.20)
    finally:
        trend_core.STRONG_SPXL = strong0
    full = daily[daily.index >= sg.FULL_OOS_START]
    m = sg.perf(full)
    assert set(m) == {"ret", "sharpe", "mdd", "n"}
    gate = sg.gate_check(m, m, full, is_sharpe=1.0)
    assert isinstance(gate["pass"], bool)
    assert len(gate["checks"]) == 5  # the five Tier-1 sub-checks


def test_higher_sleeve_changes_result(synth):
    """A different sleeve weight must actually flow through compose() (not a no-op)."""
    close, vix, irx = synth
    fd = sg.feardip_mask(close.index, synthetic=True)
    strong0 = trend_core.STRONG_SPXL
    try:
        lo = sg.run_blend(close, vix, irx, fd, 0.15, 0.20)
        hi = sg.run_blend(close, vix, irx, fd, 0.25, 0.20)
    finally:
        trend_core.STRONG_SPXL = strong0
    assert not np.allclose(lo.to_numpy(), hi.to_numpy())


def test_config_is_single_source_of_truth():
    """Live 'goal' knobs must resolve from config, matching the shipped defaults."""
    assert portfolio.SECTOR_SLEEVE_WEIGHT == config.SECTOR_SLEEVE_WEIGHT == 0.15
    assert trend_core.STRONG_SPXL == config.TREND_CORE_STRONG_SPXL == 0.20
    assert trend_core.TILT_SPXL_WEIGHT == config.TREND_CORE_TILT_WEIGHT == 0.15
    assert trend_core.ALWAYS_ON_SPXL == config.TREND_CORE_ALWAYS_ON_SPXL == 0.05
    assert trend_core.SPY_SLEEVE_WEIGHT == config.TREND_CORE_SPY_WEIGHT == 0.5
    assert trend_core.CALM_VIX_MAX == config.TREND_CORE_CALM_VIX_MAX == 16.0
    assert trend_core.VIX_REGIME_THRESHOLD == config.TREND_CORE_VIX_THRESHOLD == 22.0
