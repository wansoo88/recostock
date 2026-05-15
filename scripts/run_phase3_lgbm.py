#!/usr/bin/env python3
"""Phase 3: LightGBM walk-forward backtest.

Usage:
    python scripts/run_phase3_lgbm.py

Winning signal config: EMA-5 smoothed probability + Friday-only rebalancing.
  - Gross Sharpe ~1.1, Net Sharpe ~1.0, MDD ~13%, Cost ~2.3%/yr
  - Passes all Tier 1 criteria.

Output:
    - models/weights/lgbm_phase3.pkl       (production model)
    - models/weights/lgbm_phase3_feature_importance.csv
    - data/logs/phase3_lgbm_report.csv
"""
from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

import config
from backtest.engine import _portfolio_pnl, compute_sharpe, compute_mdd
from models.train_lgbm import (
    apply_ema_weekly,
    build_feature_matrix,
    build_target,
    proba_to_signals,
    walk_forward_lgbm,
    LGBM_PARAMS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

N_SPLITS = 5
EMA_SPAN = 5
THRESHOLD = 0.53
WEIGHTS_DIR = Path("models/weights")
LOG_DIR = Path("data/logs")


def _metrics(sig: pd.DataFrame, ret: pd.DataFrame) -> dict:
    sig = sig.astype(float).fillna(0)
    ret = ret.reindex(sig.index).fillna(0)
    pnl = _portfolio_pnl(sig, ret, config.TOTAL_COST_ROUNDTRIP)
    equity = (1 + pnl).cumprod()
    gross = (sig * ret).mean(axis=1)
    pos_delta = sig.diff(); pos_delta.iloc[0] = sig.iloc[0].abs()
    cost_s = (pos_delta.abs() * (config.TOTAL_COST_ROUNDTRIP / 2)).mean(axis=1)
    return {
        "sharpe": round(compute_sharpe(pnl) if pnl.std() > 1e-10 else 0.0, 4),
        "mdd": round(compute_mdd(equity) if len(equity) > 1 else 0.0, 4),
        "total_return": round(float(equity.iloc[-1] - 1), 4),
        "n_trades": int((pos_delta.abs() > 0).values.sum()),
        "gross_sharpe": round(compute_sharpe(gross) if gross.std() > 1e-10 else 0.0, 4),
        "cost_yr": round(float(cost_s.mean() * 252), 4),
    }


def _window_sharpes(sig: pd.DataFrame, ret: pd.DataFrame) -> list[float]:
    n = len(sig)
    min_is = min(504, n // 3)
    oos_size = (n - min_is) // N_SPLITS
    result = []
    for k in range(N_SPLITS):
        s = min_is + k * oos_size
        e = s + oos_size if k < N_SPLITS - 1 else n
        pnl = _portfolio_pnl(sig.iloc[s:e].astype(float), ret.iloc[s:e], config.TOTAL_COST_ROUNDTRIP)
        result.append(round(compute_sharpe(pnl) if pnl.std() > 1e-10 else 0.0, 3))
    return result


def _annual(sig: pd.DataFrame, ret: pd.DataFrame) -> pd.DataFrame:
    pnl = _portfolio_pnl(sig.astype(float), ret, config.TOTAL_COST_ROUNDTRIP)
    rows = []
    for yr, g in pnl.groupby(pnl.index.year):
        if len(g) < 20:
            continue
        rows.append({"year": yr,
                     "return": round(float((1 + g).prod() - 1), 4),
                     "sharpe": round(compute_sharpe(g) if g.std() > 1e-10 else 0.0, 3)})
    return pd.DataFrame(rows)


def main() -> None:
    raw_path = Path("data/raw/etf_ohlcv.parquet")
    vix_path = Path("data/raw/vix.parquet")
    if not raw_path.exists():
        log.error("데이터 없음. 먼저 실행: python -m data.collector")
        sys.exit(1)

    log.info("Loading data...")
    ohlcv = pd.read_parquet(raw_path)
    close_df = ohlcv["Close"] if isinstance(ohlcv.columns, pd.MultiIndex) else ohlcv
    vix_df = pd.read_parquet(vix_path) if vix_path.exists() else None
    log.info("OHLCV %s | %s ~ %s", close_df.shape,
             close_df.index.min().date(), close_df.index.max().date())

    # ── Features + target ─────────────────────────────────────────────────────
    log.info("Building feature matrix...")
    X = build_feature_matrix(close_df, vix_df)
    y = build_target(close_df, horizon=1)
    common = X.index.intersection(y.index)
    X, y = X.loc[common], y.loc[common]
    log.info("Feature matrix: %d samples × %d features", len(X), X.shape[1])

    # ── Walk-forward training ─────────────────────────────────────────────────
    log.info("Walk-forward LightGBM (%d splits)...", N_SPLITS)
    proba, wf_metrics = walk_forward_lgbm(X, y, n_splits=N_SPLITS, save_dir=WEIGHTS_DIR)

    # ── Signal variants ───────────────────────────────────────────────────────
    fwd_ret = close_df.pct_change().shift(-1)
    base_df = proba.unstack(level=1)
    idx = base_df.index.intersection(fwd_ret.dropna(how="all").index)
    ret = fwd_ret.loc[idx]

    # Daily threshold (baseline)
    sig_daily = proba_to_signals(proba, THRESHOLD, 1 - THRESHOLD, long_flat=True).reindex(idx).fillna(0)
    # Winning: EMA-5 + weekly
    sig_best = apply_ema_weekly(proba, ema_span=EMA_SPAN, threshold=THRESHOLD).reindex(idx).fillna(0)

    m_daily = _metrics(sig_daily, ret)
    m_best = _metrics(sig_best, ret)
    wf_sh_best = _window_sharpes(sig_best, ret)
    wf_pos_best = sum(s > 0 for s in wf_sh_best) / N_SPLITS
    annual = _annual(sig_best, ret)

    avg_is_auc = wf_metrics["is_auc"].mean()
    avg_oos_auc = wf_metrics["oos_auc"].mean()
    auc_ratio = avg_oos_auc / avg_is_auc if avg_is_auc > 0 else 0.0

    feat_imp = (pd.read_csv(WEIGHTS_DIR / "lgbm_phase3_feature_importance.csv")
                if (WEIGHTS_DIR / "lgbm_phase3_feature_importance.csv").exists() else None)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 3 — LightGBM Walk-Forward Backtest")
    print(f"  Signal: EMA-{EMA_SPAN} smoothed | Friday rebalance | threshold={THRESHOLD}")
    print(f"  Features: {X.shape[1]}  |  Splits: {N_SPLITS}  |  Cost: {config.TOTAL_COST_ROUNDTRIP:.4f}")
    print("=" * 70)

    print(f"\n── LGBM (n_est={LGBM_PARAMS['n_estimators']}, depth={LGBM_PARAMS['max_depth']}, "
          f"lr={LGBM_PARAMS['learning_rate']}, min_child={LGBM_PARAMS['min_child_samples']}) ──")

    print("\n── Walk-Forward AUC ──")
    print(f"{'Win':>3}  {'IS End':>12}  {'OOS Start':>10}  {'OOS End':>10}  "
          f"{'IS AUC':>7}  {'OOS AUC':>8}  {'n_train':>8}  OOS Sh (EMA-wk)")
    print("-" * 78)
    for i, (_, row) in enumerate(wf_metrics.iterrows()):
        sh = wf_sh_best[i] if i < len(wf_sh_best) else 0.0
        ok = "✅" if sh > 0 else "❌"
        print(f"  {int(row['window']):>1}  {row['is_end']:>12}  {row['oos_start']:>10}  "
              f"{row['oos_end']:>10}  {row['is_auc']:>7.4f}  {row['oos_auc']:>8.4f}  "
              f"{int(row['n_train']):>8,}  {sh:>6.3f} {ok}")
    print(f"\n  Avg IS AUC={avg_is_auc:.4f}  OOS AUC={avg_oos_auc:.4f}  "
          f"OOS/IS={auc_ratio:.3f}")

    print("\n── Comparison: Daily vs EMA-5 Weekly ──")
    print(f"{'Metric':<22}  {'Daily(th=0.53)':>14}  {'EMA-5+Weekly':>13}")
    print("-" * 53)
    for k in ["sharpe", "mdd", "total_return", "gross_sharpe", "cost_yr", "n_trades"]:
        print(f"  {k:<20}  {m_daily[k]:>14}  {m_best[k]:>13}")
    print(f"  {'wf_positive_pct':<20}  {'0%':>14}  {wf_pos_best:>13.0%}")

    if feat_imp is not None:
        print("\n── Top 10 Feature Importances (gain) ──")
        max_imp = feat_imp["importance_gain"].max()
        for _, r in feat_imp.head(10).iterrows():
            bar = "█" * max(1, int(r["importance_gain"] / max_imp * 20))
            print(f"  {r['feature']:<22} {bar}")

    print("\n── Annual Breakdown (EMA-5 Weekly) ──")
    print(f"{'Year':>4}  {'Return':>8}  {'Sharpe':>7}")
    print("-" * 26)
    for _, row in annual.iterrows():
        mark = "+" if row["return"] > 0 else " "
        print(f"  {int(row['year']):>4}  {row['return']:>+7.1%}  {row['sharpe']:>7.3f}")

    # ── Tier 1 Gate ───────────────────────────────────────────────────────────
    checks = [
        (f"Sharpe > {config.TIER1_SHARPE_MIN}",            m_best["sharpe"] > config.TIER1_SHARPE_MIN),
        (f"MDD < {config.TIER1_MDD_MAX:.0%}",              m_best["mdd"] < config.TIER1_MDD_MAX),
        (f"OOS/IS AUC >= {config.TIER1_OOS_IS_RATIO_MIN}", auc_ratio >= config.TIER1_OOS_IS_RATIO_MIN),
        ("WF positive >= 50%",                              wf_pos_best >= 0.5),
        (f"n_trades >= {config.TIER1_MIN_TRADING_DAYS}",   m_best["n_trades"] >= config.TIER1_MIN_TRADING_DAYS),
    ]
    print("\n── Tier 1 Gate Check (EMA-5 Weekly) ──")
    for label, passed in checks:
        print(f"  {'✅' if passed else '❌'}  {label}")

    n_fail = sum(1 for _, p in checks if not p)
    passed_all = n_fail == 0
    print("\n" + "=" * 70)
    if passed_all:
        print("  VERDICT: ✅ TIER 1 PASS — Phase 4 페이퍼 트레이딩 진입 가능!")
    else:
        print(f"  VERDICT: ❌ TIER 1 FAIL ({n_fail}개 미달)")
    print("=" * 70)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in wf_metrics.iterrows():
        rows.append({"section": "walk_forward", **row.to_dict()})
    for _, row in annual.iterrows():
        rows.append({"section": "annual", "year": int(row["year"]),
                     "return": row["return"], "sharpe": row["sharpe"]})
    rows.append({"section": "aggregate_ema_weekly", **m_best,
                 "wf_positive_pct": round(wf_pos_best, 4),
                 "auc_ois_ratio": round(auc_ratio, 4),
                 "tier1_pass": int(passed_all)})
    rows.append({"section": "aggregate_daily_baseline", **m_daily})
    pd.DataFrame(rows).to_csv(LOG_DIR / "phase3_lgbm_report.csv", index=False)
    log.info("리포트 저장: %s", LOG_DIR / "phase3_lgbm_report.csv")


if __name__ == "__main__":
    main()
