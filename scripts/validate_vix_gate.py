"""Walk-forward OOS validation for the VIX gate hypothesis.

In-sample observation: v2 model has negative Sharpe when VIX < 20.
Question: is this a real, persistent pattern (use it) or a data-snoop
artifact (don't)?

Method:
  Split full OOS into 5 chronological folds.
  For each fold k:
    1. Take folds [0..k-1] as IS history
    2. Choose VIX threshold that maximises Sharpe in IS history
    3. Apply that threshold to fold k (OOS)
    4. Compare gated vs ungated OOS Sharpe
  Aggregate: was the gate beneficial in real OOS?
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

import config
from data.collector import load_parquet
from models.train_lgbm import apply_ema_weekly

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Load OOS proba (already walk-forward generated)
proba = pd.read_parquet("data/logs/phase3_v2_uniform_oos_proba.parquet")["proba"]
ohlcv = load_parquet("etf_ohlcv")
close_full = ohlcv["Close"]
vix = load_parquet("vix").iloc[:, 0]

tickers = [t for t in config.CORE_ETFS + config.SECTOR_ETFS + config.INVERSE_ETFS
           if t in close_full.columns]
close_df = close_full[tickers].dropna(how="all")


# ── Build weekly returns w/ optional VIX gate ──────────────────────────────────

def weekly_returns_with_gate(proba_s: pd.Series, vix_threshold: float | None = None,
                             threshold: float = 0.53) -> pd.Series:
    """Return weekly net portfolio returns. Gate disables signals when
    VIX < vix_threshold (signal forced to 0)."""
    sig = apply_ema_weekly(proba_s, ema_span=5, threshold=threshold)
    if vix_threshold is not None:
        vix_at = vix.reindex(sig.index, method="ffill")
        # zero out all signals on dates where VIX is too low
        mask = (vix_at >= vix_threshold).values
        gate = np.broadcast_to(mask[:, None], sig.shape)
        sig = pd.DataFrame(np.where(gate, sig.values, 0.0),
                           index=sig.index, columns=sig.columns)

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
    return (gross - cost_w).fillna(0)


def sharpe(r: pd.Series) -> float:
    if r.std() < 1e-12:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(52))


def mdd_pct(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float(((eq - eq.cummax()) / eq.cummax()).min()) * 100


# ── Step 1: In-sample search — what VIX threshold optimises Sharpe? ───────────

print("=" * 90)
print("IN-SAMPLE — VIX threshold search (full OOS proba period)")
print("=" * 90)

candidates = [None, 12, 14, 16, 18, 20, 22, 25]
rows = []
for thr in candidates:
    r = weekly_returns_with_gate(proba, vix_threshold=thr)
    label = "no gate" if thr is None else f"VIX≥{thr}"
    rows.append({
        "vix_gate": label,
        "n_weeks_active": int((r != 0).sum()),
        "sharpe": round(sharpe(r), 3),
        "mdd_pct": round(mdd_pct(r), 2),
        "total_ret_pct": round(((1+r).cumprod().iloc[-1] - 1) * 100, 2),
    })
in_sample = pd.DataFrame(rows)
print(in_sample.to_string(index=False))
best_thr = None
best_sharpe = sharpe(weekly_returns_with_gate(proba, vix_threshold=None))
for c in candidates[1:]:
    s = sharpe(weekly_returns_with_gate(proba, vix_threshold=c))
    if s > best_sharpe:
        best_sharpe = s
        best_thr = c
print(f"\nIn-sample optimal: VIX≥{best_thr}  Sharpe={best_sharpe:.3f}")


# ── Step 2: Walk-forward OOS validation ───────────────────────────────────────

print("\n" + "=" * 90)
print("WALK-FORWARD OOS — choose gate using IS, apply to next OOS fold")
print("=" * 90)

# 5 folds over chronological weekly index
weekly_full = weekly_returns_with_gate(proba, vix_threshold=None)
all_weeks = weekly_full.index
n = len(all_weeks)
fold_size = n // 5

fold_rows = []
chosen_thrs = []

for k in range(1, 5):  # fold 1..4 as OOS (need at least 1 fold of IS)
    is_end = k * fold_size
    oos_start = is_end
    oos_end = (k + 1) * fold_size if k < 4 else n
    is_dates = all_weeks[:is_end]
    oos_dates = all_weeks[oos_start:oos_end]

    # IS: choose threshold maximising IS Sharpe
    best_is_thr = None
    best_is_sharpe = sharpe(weekly_full.loc[is_dates])
    for thr in candidates[1:]:
        r_is = weekly_returns_with_gate(proba, vix_threshold=thr).loc[is_dates]
        s = sharpe(r_is)
        if s > best_is_sharpe:
            best_is_sharpe = s
            best_is_thr = thr

    # OOS: apply chosen threshold to OOS fold
    if best_is_thr is None:
        oos_gated = weekly_full.loc[oos_dates]
        oos_ungated = weekly_full.loc[oos_dates]
    else:
        oos_gated = weekly_returns_with_gate(proba, vix_threshold=best_is_thr).loc[oos_dates]
        oos_ungated = weekly_full.loc[oos_dates]

    fold_rows.append({
        "fold": k,
        "is_dates": f"{is_dates.min().date()}~{is_dates.max().date()}",
        "is_n": len(is_dates),
        "best_is_thr": best_is_thr if best_is_thr else "no gate",
        "is_sharpe": round(best_is_sharpe, 3),
        "oos_dates": f"{oos_dates.min().date()}~{oos_dates.max().date()}",
        "oos_n": len(oos_dates),
        "oos_sharpe_gated": round(sharpe(oos_gated), 3),
        "oos_sharpe_ungated": round(sharpe(oos_ungated), 3),
        "gate_delta": round(sharpe(oos_gated) - sharpe(oos_ungated), 3),
    })
    chosen_thrs.append(best_is_thr)

fold_df = pd.DataFrame(fold_rows)
print(fold_df.to_string(index=False))

print(f"\nChosen thresholds per fold: {chosen_thrs}")
mean_delta = fold_df["gate_delta"].mean()
positive_folds = (fold_df["gate_delta"] > 0).sum()
print(f"Mean OOS Sharpe Δ (gated - ungated): {mean_delta:+.3f}")
print(f"Folds where gate helped: {positive_folds}/{len(fold_df)}")


# ── Step 3: Robustness — try fixed threshold (no IS optimisation) ─────────────

print("\n" + "=" * 90)
print("ROBUSTNESS — fixed threshold across all OOS folds (no per-fold tuning)")
print("=" * 90)

# What if we just say "VIX>=20 always"? Apply to each fold separately
robust_rows = []
for fixed_thr in [18, 20, 22]:
    deltas = []
    for k in range(1, 5):
        is_end = k * fold_size
        oos_start = is_end
        oos_end = (k + 1) * fold_size if k < 4 else n
        oos_dates = all_weeks[oos_start:oos_end]
        r_gated = weekly_returns_with_gate(proba, vix_threshold=fixed_thr).loc[oos_dates]
        r_ungated = weekly_full.loc[oos_dates]
        deltas.append(sharpe(r_gated) - sharpe(r_ungated))
    robust_rows.append({
        "fixed_threshold": f"VIX≥{fixed_thr}",
        "deltas": [round(d, 3) for d in deltas],
        "mean_delta": round(np.mean(deltas), 3),
        "median_delta": round(np.median(deltas), 3),
        "positive_folds": sum(1 for d in deltas if d > 0),
    })
robust_df = pd.DataFrame(robust_rows)
print(robust_df.to_string(index=False))


# ── Step 4: Verdict ───────────────────────────────────────────────────────────

print("\n" + "=" * 90)
print("VERDICT")
print("=" * 90)
if mean_delta > 0.05 and positive_folds >= 3:
    print(f"✓ VIX gate is OOS-robust (mean Δ={mean_delta:+.3f}, "
          f"{positive_folds}/4 folds positive).")
    print(f"  RECOMMEND: adopt fixed VIX≥20 gate in production.")
elif mean_delta > 0:
    print(f"~ Marginal: mean Δ={mean_delta:+.3f}, {positive_folds}/4 folds positive.")
    print(f"  RECOMMEND: monitor in shadow mode, do not adopt yet.")
else:
    print(f"✗ VIX gate is NOT OOS-robust (mean Δ={mean_delta:+.3f}, "
          f"only {positive_folds}/4 folds positive).")
    print(f"  RECOMMEND: REJECT the gate — the in-sample observation is "
          f"likely data-snooping or regime-specific.")
