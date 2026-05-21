"""Look-ahead guard for the macro feature layer (build_global_macro).

The original test_lookahead.py covers only the technical factors
(compute_factors). The macro layer was added later (v3/v4/v5) without a
causality test, despite CLAUDE.md requiring one after every factor change.

Gold-standard test: a feature value at date T computed from the FULL history
must equal the value computed from history restricted to <= T. Any mismatch
means information from after T leaked into T (look-ahead).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import pytest

from features.macro_factors import build_global_macro

_KEYS = ["dxy", "yield_10y", "yield_2y", "oil", "gold", "hyg", "lqd", "tlt",
         "vvix", "vix", "vix9d", "vix3m", "skew", "move"]


@pytest.fixture
def macro() -> dict[str, pd.Series]:
    np.random.seed(7)
    dates = pd.date_range("2022-01-03", periods=400, freq="B")
    out = {}
    for i, k in enumerate(_KEYS):
        base = 20 + 5 * i
        series = base * np.exp(np.cumsum(np.random.normal(0, 0.01, len(dates))))
        out[k] = pd.Series(series, index=dates, name=k)
    return out


def test_macro_features_no_lookahead(macro):
    full_index = macro["vix"].index
    full = build_global_macro(full_index, macro)
    assert not full.empty

    # Check at several dates deep enough for the 63-day rolling windows.
    for t in full_index[120::40]:
        macro_r = {k: s[s.index <= t] for k, s in macro.items()}
        index_r = full_index[full_index <= t]
        restricted = build_global_macro(index_r, macro_r)
        if t not in restricted.index:
            continue
        for col in full.columns:
            fv, rv = full.loc[t, col], restricted.loc[t, col]
            if pd.isna(fv) and pd.isna(rv):
                continue
            assert abs(fv - rv) < 1e-8, (
                f"Look-ahead in macro feature '{col}' at {t.date()}: "
                f"full={fv:.8f} restricted={rv:.8f}"
            )


def test_macro_index_matches_input(macro):
    idx = macro["vix"].index
    out = build_global_macro(idx, macro)
    assert out.index.equals(idx)


def test_macro_empty_input_safe():
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    out = build_global_macro(idx, {})
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(idx)


def test_macro_asymmetric_series_no_lookahead(macro):
    """Audit flagged HY/IG: hyg ending earlier than lqd, ffill carrying stale
    hyg forward. ffill uses PAST values so it is causal — prove it: truncate
    hyg 30 business days early and confirm full==restricted at dates past the
    truncation (stale-but-causal, no leakage)."""
    full_index = macro["vix"].index
    cut = full_index[-30]
    macro2 = dict(macro)
    macro2["hyg"] = macro["hyg"][macro["hyg"].index < cut]  # hyg dies 30d early
    full = build_global_macro(full_index, macro2)

    for t in full_index[-20::5]:  # all past hyg's last date
        macro_r = {k: s[s.index <= t] for k, s in macro2.items()}
        index_r = full_index[full_index <= t]
        restricted = build_global_macro(index_r, macro_r)
        if t not in restricted.index:
            continue
        for col in ("hy_ig_logratio", "hy_ig_z"):
            if col not in full.columns:
                continue
            fv, rv = full.loc[t, col], restricted.loc[t, col]
            if pd.isna(fv) and pd.isna(rv):
                continue
            assert abs(fv - rv) < 1e-8, f"asymmetric ffill leak in '{col}' at {t.date()}"
