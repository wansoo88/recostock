"""Enhanced backtest for v2 model: block bootstrap CI, regime split, multi-metric.

Inputs: phase3_v2_uniform_oos_proba.parquet (OOS predictions).
Outputs:
- bootstrap confidence intervals for Sharpe / WR / total return
- regime-decomposed performance (VIX <15, 15-25, >25)
- Calmar / Sortino / Ulcer Index
- per-ticker contribution analysis
- equity curve saved for plotting
"""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config
from data.collector import load_parquet
from models.train_lgbm import apply_ema_weekly

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

OOS_PROBA_PATH = Path("data/logs/phase3_v2_uniform_oos_proba.parquet")
assert OOS_PROBA_PATH.exists(), f"Missing {OOS_PROBA_PATH}. Run run_phase3_v2.py first."

proba = pd.read_parquet(OOS_PROBA_PATH)["proba"]
log.info("Loaded OOS proba: %d rows", len(proba))

ohlcv = load_parquet("etf_ohlcv")
close_full = ohlcv["Close"]
vix = load_parquet("vix").iloc[:, 0]


# ── Construct weekly returns (same logic as backtest in run_phase3_v2) ─────────

def to_weekly_returns(proba_s: pd.Series, close_df: pd.DataFrame,
                      threshold: float = 0.53) -> pd.DataFrame:
    """Return weekly portfolio DataFrame with columns: net_pnl, n_active, equity."""
    sig = apply_ema_weekly(proba_s, ema_span=5, threshold=threshold)
    ret_5d = close_df.pct_change(5).shift(-5)
    common = sig.index.intersection(ret_5d.index)
    sig = sig.reindex(common)
    ret_5d = ret_5d.reindex(common)
    common_t = [t for t in sig.columns if t in ret_5d.columns]
    sig = sig[common_t]
    ret_5d = ret_5d[common_t]
    on_t = sig.diff().fillna(sig).clip(lower=0)
    off_t = sig.diff().fillna(0).clip(upper=0).abs()
    cost = (on_t + off_t) * (config.TOTAL_COST_ROUNDTRIP / 2.0)
    fri = sig.index.dayofweek == 4
    sig_f = sig[fri]
    ret_f = ret_5d.reindex(sig_f.index)
    cost_f = cost.reindex(sig_f.index)
    n_act = sig_f.sum(axis=1)
    n_act_safe = n_act.replace(0, np.nan)
    gross = (sig_f * ret_f).sum(axis=1) / n_act_safe
    cost_w = cost_f.sum(axis=1) / n_act.replace(0, 1)
    net = (gross - cost_w).fillna(0)
    out = pd.DataFrame({"net": net, "gross": gross.fillna(0), "n_active": n_act})
    out["equity"] = (1 + out["net"]).cumprod()
    return out


tickers = [t for t in config.CORE_ETFS + config.SECTOR_ETFS + config.INVERSE_ETFS
           if t in close_full.columns]
close_df = close_full[tickers].dropna(how="all")
weekly = to_weekly_returns(proba, close_df)
log.info("Weekly portfolio: %d rows  (active weeks=%d)",
         len(weekly), int((weekly["n_active"] > 0).sum()))


# ── Multi-metric calculations ──────────────────────────────────────────────────

def metrics(weekly_ret: pd.Series, label: str) -> dict:
    if len(weekly_ret) < 2 or weekly_ret.std() < 1e-10:
        return {"label": label, "n_weeks": len(weekly_ret), "sharpe": 0,
                "sortino": 0, "calmar": 0, "ulcer": 0, "wr": 0, "mdd": 0,
                "total_ret": 0, "ann_ret": 0, "ann_vol": 0}

    ann_ret = (1 + weekly_ret.mean()) ** 52 - 1
    ann_vol = weekly_ret.std() * np.sqrt(52)
    sharpe = weekly_ret.mean() / weekly_ret.std() * np.sqrt(52)
    # Sortino — downside deviation only
    down = weekly_ret[weekly_ret < 0]
    down_vol = down.std() * np.sqrt(52) if len(down) > 1 else weekly_ret.std() * np.sqrt(52)
    sortino = weekly_ret.mean() * 52 / down_vol if down_vol > 0 else 0
    # MDD + Calmar
    eq = (1 + weekly_ret).cumprod()
    dd = (eq - eq.cummax()) / eq.cummax()
    mdd = float(dd.min())
    total_ret = float(eq.iloc[-1] - 1)
    n_years = len(weekly_ret) / 52
    cagr = (eq.iloc[-1]) ** (1 / n_years) - 1 if n_years > 0 else 0
    calmar = cagr / abs(mdd) if mdd < 0 else 0
    # Ulcer Index — sqrt(mean(dd**2))
    ulcer = float(np.sqrt((dd ** 2).mean())) * 100
    # WR (positive weeks among active ones)
    active = weekly_ret[weekly_ret != 0]
    wr = (active > 0).sum() / max(1, len(active))

    return {
        "label": label,
        "n_weeks": len(weekly_ret),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "ulcer_pct": round(ulcer, 2),
        "wr": round(wr, 3),
        "mdd_pct": round(mdd * 100, 2),
        "total_ret_pct": round(total_ret * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
    }


# ── Period decomposition ──────────────────────────────────────────────────────

print("\n" + "=" * 95)
print("PERIOD-DECOMPOSED METRICS — v2 uniform")
print("=" * 95)

periods = {
    "Full OOS":              weekly["net"],
    "2017-2020 (early bull)":      weekly["net"][(weekly.index >= "2017-01-01") & (weekly.index < "2020-01-01")],
    "2020 (COVID + recovery)":    weekly["net"][(weekly.index >= "2020-01-01") & (weekly.index < "2021-01-01")],
    "2021-2022 (bear)":            weekly["net"][(weekly.index >= "2021-01-01") & (weekly.index < "2023-01-01")],
    "2023-2024 (recovery)":        weekly["net"][(weekly.index >= "2023-01-01") & (weekly.index < "2025-01-01")],
    "2025+ (recent)":              weekly["net"][weekly.index >= "2025-01-01"],
    "Recent 12 months":            weekly["net"][weekly.index >= weekly.index.max() - pd.Timedelta(days=365)],
    "Recent 6 months":             weekly["net"][weekly.index >= weekly.index.max() - pd.Timedelta(days=183)],
}
rows = [metrics(s, k) for k, s in periods.items()]
df_metrics = pd.DataFrame(rows)
print(df_metrics.to_string(index=False))


# ── VIX-regime decomposition ──────────────────────────────────────────────────

print("\n" + "=" * 95)
print("VIX-REGIME DECOMPOSITION (VIX on Friday open week)")
print("=" * 95)

vix_friday = vix.reindex(weekly.index).ffill()
regime = pd.cut(vix_friday, bins=[0, 15, 20, 25, 100],
                labels=["calm (<15)", "normal (15-20)", "caution (20-25)", "fear (>25)"])
for r in regime.cat.categories:
    s = weekly["net"][regime == r]
    if len(s) < 5:
        continue
    print(f"  {r:<22}", metrics(s, r))

reg_df = pd.DataFrame({
    "regime": regime,
    "net": weekly["net"],
})
reg_summary = reg_df.groupby("regime", observed=True).agg(
    n=("net", "count"),
    avg=("net", "mean"),
    std=("net", "std"),
    pos_pct=("net", lambda x: (x > 0).mean()),
).round(5)
print("\n", reg_summary.to_string())


# ── Block bootstrap CI ────────────────────────────────────────────────────────

print("\n" + "=" * 95)
print("BLOCK BOOTSTRAP — 95% confidence intervals (block=8 weeks, B=2000)")
print("=" * 95)


def block_bootstrap(series: np.ndarray, block_size: int = 8,
                    B: int = 2000, seed: int = 42) -> dict:
    """Stationary block bootstrap for Sharpe ratio CI."""
    rng = np.random.default_rng(seed)
    n = len(series)
    if n < block_size * 2:
        return {"n": n, "sharpe_mean": np.nan, "sharpe_ci": (np.nan, np.nan),
                "wr_ci": (np.nan, np.nan), "ret_ci": (np.nan, np.nan)}

    blocks = max(1, n // block_size)
    sharpes = np.empty(B)
    wrs = np.empty(B)
    rets = np.empty(B)

    for b in range(B):
        starts = rng.integers(0, n - block_size + 1, size=blocks)
        boot = np.concatenate([series[s:s + block_size] for s in starts])[:n]
        std = boot.std()
        if std < 1e-12:
            sharpes[b] = 0
        else:
            sharpes[b] = boot.mean() / std * np.sqrt(52)
        wrs[b] = (boot > 0).sum() / max(1, (boot != 0).sum())
        rets[b] = np.prod(1 + boot) - 1

    return {
        "n": n,
        "sharpe_obs": series.mean() / series.std() * np.sqrt(52) if series.std() > 0 else 0,
        "sharpe_mean_boot": float(np.mean(sharpes)),
        "sharpe_ci_low": float(np.percentile(sharpes, 2.5)),
        "sharpe_ci_hi":  float(np.percentile(sharpes, 97.5)),
        "wr_obs": (series > 0).sum() / max(1, (series != 0).sum()),
        "wr_ci_low": float(np.percentile(wrs, 2.5)),
        "wr_ci_hi":  float(np.percentile(wrs, 97.5)),
        "ret_ci_low": float(np.percentile(rets, 2.5)),
        "ret_ci_hi":  float(np.percentile(rets, 97.5)),
        "sharpe_pct_positive": float((sharpes > 0).mean()),
    }


boot_periods = {
    "Full OOS": weekly["net"].values,
    "Recent 24m": weekly["net"][weekly.index >= weekly.index.max() - pd.Timedelta(days=730)].values,
    "Recent 12m": weekly["net"][weekly.index >= weekly.index.max() - pd.Timedelta(days=365)].values,
}
for label, arr in boot_periods.items():
    res = block_bootstrap(arr, block_size=8, B=2000)
    print(f"\n  [{label}]  n={res['n']}")
    print(f"    Sharpe obs={res['sharpe_obs']:+.3f}  "
          f"95% CI = [{res['sharpe_ci_low']:+.3f}, {res['sharpe_ci_hi']:+.3f}]  "
          f"P(Sharpe > 0) = {res['sharpe_pct_positive']:.1%}")
    print(f"    WR     obs={res['wr_obs']:.3f}  "
          f"95% CI = [{res['wr_ci_low']:.3f}, {res['wr_ci_hi']:.3f}]")
    print(f"    Total ret 95% CI = [{res['ret_ci_low']*100:+.2f}%, "
          f"{res['ret_ci_hi']*100:+.2f}%]")


# ── Per-ticker contribution ───────────────────────────────────────────────────

print("\n" + "=" * 95)
print("PER-TICKER CONTRIBUTION (recent 12 months)")
print("=" * 95)

threshold = 0.53
sig = apply_ema_weekly(proba, ema_span=5, threshold=threshold)
ret_5d = close_df.pct_change(5).shift(-5)
common = sig.index.intersection(ret_5d.index)
sig = sig.reindex(common); ret_5d = ret_5d.reindex(common)
cutoff = sig.index.max() - pd.Timedelta(days=365)
sig_recent = sig[sig.index >= cutoff]
ret_recent = ret_5d.reindex(sig_recent.index)
fri = sig_recent.index.dayofweek == 4
sig_recent = sig_recent[fri]; ret_recent = ret_recent.reindex(sig_recent.index)

per_t_rows = []
for t in sig_recent.columns:
    active_mask = sig_recent[t] > 0
    if active_mask.sum() < 3:
        continue
    rets_t = ret_recent[t][active_mask].dropna()
    if rets_t.empty:
        continue
    per_t_rows.append({
        "ticker": t,
        "active_weeks": int(active_mask.sum()),
        "wr": round((rets_t > 0).mean(), 4),
        "avg_ret_pct": round(rets_t.mean() * 100, 4),
        "total_ret_pct": round((rets_t.sum() - (config.TOTAL_COST_ROUNDTRIP) * active_mask.sum()) * 100, 4),
        "best_pct": round(rets_t.max() * 100, 4),
        "worst_pct": round(rets_t.min() * 100, 4),
    })
per_t_df = pd.DataFrame(per_t_rows).sort_values("total_ret_pct", ascending=False)
print(per_t_df.to_string(index=False))


# ── Save all results ─────────────────────────────────────────────────────────

out_dir = Path("data/logs")
df_metrics.to_csv(out_dir / "phase3_v2_metrics_by_period.csv", index=False)
weekly.to_csv(out_dir / "phase3_v2_weekly_equity.csv")
per_t_df.to_csv(out_dir / "phase3_v2_per_ticker_recent12m.csv", index=False)
log.info("Saved enhanced backtest results to %s", out_dir)
