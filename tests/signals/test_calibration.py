"""Tests for proba calibration display map."""
from signals.calibration import calibrated_winrate, BASE_WIN_RATE, _XS, _YS


def test_none_passthrough():
    assert calibrated_winrate(None) is None


def test_clips_below_and_above():
    assert calibrated_winrate(0.10) == calibrated_winrate(_XS[0])
    assert calibrated_winrate(0.99) == calibrated_winrate(_XS[-1])


def test_monotonic_nondecreasing():
    vals = [calibrated_winrate(p) for p in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))


def test_overstatement_is_corrected():
    # Raw confidence 0.74 should map to a much lower actual win prob (~0.59).
    cw = calibrated_winrate(0.74)
    assert cw < 0.65, f"calibrated win {cw} should be well below raw 0.74"
    assert 0.50 < cw < 0.65


def test_base_rate_sane():
    assert 0.5 < BASE_WIN_RATE < 0.65
