"""IC analysis for new macro/external features.

Two evaluation modes:
1. POOLED  — all (ticker, date) rows together, single spearman per feature
2. PER-TICKER — spearman per (feature, ticker) pair to find which tickers
                gain most from each feature

Look-ahead is enforced by build_global_macro: features at T use only data ≤ T,
target is forward return T → T+h (shift -h).

Cost-context note: Phase 0+ uses 5-day horizon (matches Friday-rebalance).
IC_MIN_VIABLE = 0.01 from config.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

import config
from data.collector import load_parquet
from features.macro_factors import build_global_macro, build_ticker_specific
from features.ic_analysis import _two_sided_pvalue

logging.basicConfig(level=logging.WARNING)


# ── Setup ──────────────────────────────────────────────────────────────────────

ohlcv = load_parquet("etf_ohlcv")
close = ohlcv["Close"].dropna(how="all")
tickers = [t for t in config.CORE_ETFS + config.SECTOR_ETFS if t in close.columns]
close = close[tickers].dropna(how="all")

print(f"Universe: {tickers}")
print(f"Date range: {close.index.min().date()} → {close.index.max().date()}")
print(f"Rows: {len(close)}")

HORIZONS = [1, 5, 21]

# Build global macro features (same for every ticker on a given date)
global_mac = build_global_macro(close.index)
print(f"\nGlobal macro features: {list(global_mac.columns)}")
print(f"  rows after dropna: {len(global_mac.dropna())}")

# Forward returns
fwd_returns: dict[int, pd.DataFrame] = {}
for h in HORIZONS:
    fwd_returns[h] = np.log(close / close.shift(1)).shift(-h)


# ── 1. POOLED IC for each global feature × horizon ─────────────────────────────

def pooled_ic(feature_series: pd.Series,
              fwd_df: pd.DataFrame,
              tickers_used: list[str]) -> tuple[float, int, float]:
    """Pool all (date, ticker) rows: feature value is broadcast to all tickers."""
    rows = []
    for t in tickers_used:
        fwd = fwd_df[t]
        common = feature_series.index.intersection(fwd.index)
        if len(common) < 30:
            continue
        rows.append(pd.DataFrame({"f": feature_series.loc[common],
                                  "r": fwd.loc[common]}))
    if not rows:
        return float("nan"), 0, float("nan")
    full = pd.concat(rows).dropna()
    if len(full) < 50:
        return float("nan"), len(full), float("nan")
    ic = float(full["f"].corr(full["r"], method="spearman"))
    # Fisher z transformation for SE
    n = len(full)
    se = 1.0 / np.sqrt(n - 3) if n > 3 else float("nan")
    z = 0.5 * np.log((1 + ic) / (1 - ic)) if abs(ic) < 0.999 else 0.0
    p = _two_sided_pvalue(z / se) if se and se > 0 else float("nan")
    return ic, n, p


pooled_rows = []
for feat_name in global_mac.columns:
    feat = global_mac[feat_name].dropna()
    if len(feat) < 100:
        continue
    for h in HORIZONS:
        ic, n, p = pooled_ic(feat, fwd_returns[h], tickers)
        viable = abs(ic) >= config.IC_MIN_VIABLE and p < 0.05 if not np.isnan(p) else False
        pooled_rows.append({
            "feature": feat_name,
            "horizon": h,
            "ic_pooled": round(ic, 4),
            "n": n,
            "p_value": round(p, 4),
            "verdict": "KEEP" if viable else "REJECT",
        })

pooled_df = pd.DataFrame(pooled_rows).sort_values(
    by=["horizon", "ic_pooled"],
    key=lambda s: s.abs() if s.name == "ic_pooled" else s,
    ascending=[True, False],
)

print("\n" + "=" * 90)
print("POOLED IC — all tickers combined (cross-sectional + time)")
print("=" * 90)
print(pooled_df.to_string(index=False))


# ── 2. PER-TICKER IC for each feature (horizon=5 main signal) ──────────────────

def per_ticker_ic(feature_series: pd.Series,
                  fwd_series: pd.Series,
                  ticker: str) -> tuple[float, int, float]:
    common = feature_series.index.intersection(fwd_series.index)
    df = pd.concat([feature_series.loc[common], fwd_series.loc[common]],
                   axis=1).dropna()
    df.columns = ["f", "r"]
    if len(df) < 50:
        return float("nan"), len(df), float("nan")
    ic = float(df["f"].corr(df["r"], method="spearman"))
    n = len(df)
    se = 1.0 / np.sqrt(n - 3) if n > 3 else float("nan")
    z = 0.5 * np.log((1 + ic) / (1 - ic)) if abs(ic) < 0.999 else 0.0
    p = _two_sided_pvalue(z / se) if se and se > 0 else float("nan")
    return ic, n, p


main_h = 5
per_ticker_rows = []
for feat_name in global_mac.columns:
    feat = global_mac[feat_name].dropna()
    if len(feat) < 100:
        continue
    for t in tickers:
        ic, n, p = per_ticker_ic(feat, fwd_returns[main_h][t], t)
        per_ticker_rows.append({
            "feature": feat_name,
            "ticker": t,
            "ic": round(ic, 4),
            "n": n,
            "p_value": round(p, 4),
            "sig": "*" if (not np.isnan(p) and p < 0.05) else " ",
        })
pt_df = pd.DataFrame(per_ticker_rows)
pivot = pt_df.pivot(index="feature", columns="ticker", values="ic")
sig_pivot = pt_df.pivot(index="feature", columns="ticker", values="sig")

print("\n" + "=" * 90)
print(f"PER-TICKER IC — horizon={main_h}d (* marks p<0.05)")
print("=" * 90)
for ticker in tickers:
    col_ic = pivot[ticker]
    col_sig = sig_pivot[ticker]
    print(f"\n  {ticker}:")
    sub = pd.DataFrame({"ic": col_ic, "sig": col_sig}).sort_values(
        "ic", key=lambda s: s.abs(), ascending=False)
    print(sub.to_string())


# ── 3. TICKER-SPECIFIC features (XLE oil elasticity, XLF/KRE spread, etc.) ─────

print("\n" + "=" * 90)
print(f"TICKER-SPECIFIC feature IC — horizon={main_h}d")
print("=" * 90)

ticker_specific_rows = []
for t in tickers:
    ts_feats = build_ticker_specific(close.index, close, t)
    if ts_feats.empty:
        continue
    fwd = fwd_returns[main_h][t]
    for col in ts_feats.columns:
        ic, n, p = per_ticker_ic(ts_feats[col].dropna(), fwd, t)
        ticker_specific_rows.append({
            "ticker": t,
            "feature": col,
            "ic": round(ic, 4),
            "n": n,
            "p_value": round(p, 4),
            "verdict": "KEEP" if (abs(ic) >= config.IC_MIN_VIABLE and
                                  not np.isnan(p) and p < 0.05) else "REJECT",
        })

ts_df = pd.DataFrame(ticker_specific_rows)
if not ts_df.empty:
    print(ts_df.to_string(index=False))
else:
    print("(no ticker-specific features generated)")


# ── 4. KEEP recommendation ─────────────────────────────────────────────────────

print("\n" + "=" * 90)
print("KEEP / REJECT 결정 (pooled, h=5)")
print("=" * 90)
h5 = pooled_df[pooled_df["horizon"] == 5].sort_values(
    "ic_pooled", key=lambda s: s.abs(), ascending=False)
print(h5.to_string(index=False))

keepers = h5[h5["verdict"] == "KEEP"]["feature"].tolist()
print(f"\nKEEP candidates (pooled h=5, |IC|≥0.01 & p<0.05): {keepers}")


# ── 5. Save ────────────────────────────────────────────────────────────────────

out_dir = "data/logs"
import os
os.makedirs(out_dir, exist_ok=True)
pooled_df.to_csv(f"{out_dir}/macro_ic_pooled.csv", index=False)
pt_df.to_csv(f"{out_dir}/macro_ic_per_ticker.csv", index=False)
ts_df.to_csv(f"{out_dir}/macro_ic_ticker_specific.csv", index=False)
print(f"\nSaved: {out_dir}/macro_ic_*.csv")
