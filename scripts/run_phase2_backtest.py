#!/usr/bin/env python3
"""Phase 2: Baseline rules model walk-forward backtest.

Usage:
    python scripts/run_phase2_backtest.py

Output:
    - Console: per-window IS/OOS metrics, Tier 1 gate verdict
    - data/logs/phase2_backtest_report.csv
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

import config
from backtest.engine import BacktestResult, _portfolio_pnl, compute_sharpe, compute_mdd, run_walk_forward
from models.baseline import build_signals, IC_WEIGHTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

N_SPLITS = 5


def _spy_benchmark(close_df: pd.DataFrame, fwd_returns: pd.DataFrame) -> dict:
    """SPY buy-and-hold metrics on the OOS period for comparison."""
    if "SPY" not in close_df.columns:
        return {}
    spy_ret = fwd_returns["SPY"].dropna()
    equity = (1 + spy_ret).cumprod()
    return {
        "sharpe": round(compute_sharpe(spy_ret), 3),
        "total_return": round(float(equity.iloc[-1] - 1), 4),
        "mdd": round(compute_mdd(equity), 4),
    }


def _annual_breakdown(pnl: pd.Series) -> pd.DataFrame:
    """Year-by-year return and Sharpe."""
    rows = []
    for year, group in pnl.groupby(pnl.index.year):
        if len(group) < 20:
            continue
        ann_ret = float((1 + group).prod() - 1)
        sharpe = compute_sharpe(group) if group.std() > 1e-10 else 0.0
        rows.append({"year": year, "return": round(ann_ret, 4), "sharpe": round(sharpe, 3)})
    return pd.DataFrame(rows)


def _window_breakdown(
    sig: pd.DataFrame,
    ret: pd.DataFrame,
    n_splits: int,
) -> list[dict]:
    """Per walk-forward window IS / OOS Sharpe."""
    n = len(sig)
    min_is = min(504, n // 3)
    oos_size = (n - min_is) // n_splits
    rows = []
    for k in range(n_splits):
        is_end = min_is + k * oos_size
        oos_start = is_end
        oos_end = oos_start + oos_size if k < n_splits - 1 else n
        is_pnl = _portfolio_pnl(sig.iloc[:is_end], ret.iloc[:is_end], config.TOTAL_COST_ROUNDTRIP)
        oos_pnl = _portfolio_pnl(sig.iloc[oos_start:oos_end], ret.iloc[oos_start:oos_end], config.TOTAL_COST_ROUNDTRIP)
        is_sh = compute_sharpe(is_pnl) if is_pnl.std() > 1e-10 else 0.0
        oos_sh = compute_sharpe(oos_pnl) if oos_pnl.std() > 1e-10 else 0.0
        oos_ret_total = float((1 + oos_pnl).prod() - 1)
        rows.append({
            "window": k + 1,
            "is_end_date": str(sig.index[is_end - 1].date()),
            "oos_start_date": str(sig.index[oos_start].date()),
            "oos_end_date": str(sig.index[min(oos_end - 1, n - 1)].date()),
            "is_sharpe": round(is_sh, 3),
            "oos_sharpe": round(oos_sh, 3),
            "oos_return": round(oos_ret_total, 4),
            "positive": oos_ret_total > 0,
        })
    return rows


def main() -> None:
    raw_path = Path("data/raw/etf_ohlcv.parquet")
    if not raw_path.exists():
        log.error("데이터 없음. 먼저 실행: python -m data.collector")
        sys.exit(1)

    log.info("Loading OHLCV data...")
    ohlcv = pd.read_parquet(raw_path)
    close_df = ohlcv["Close"] if isinstance(ohlcv.columns, pd.MultiIndex) else ohlcv
    log.info("Shape: %s | %s ~ %s", close_df.shape,
             close_df.index.min().date(), close_df.index.max().date())

    # ── Signals & forward returns ─────────────────────────────────────────────
    log.info("Building cross-sectional IC-weighted signals...")
    signals = build_signals(close_df, threshold=0.0)
    log.info("Signal shape: %s | active tickers: %d", signals.shape, signals.shape[1])

    # Forward simple returns: return[T] = (close[T+1] - close[T]) / close[T]
    fwd_returns = close_df.pct_change().shift(-1)

    # Align
    idx = signals.index.intersection(fwd_returns.dropna(how="all").index)
    sig = signals.loc[idx]
    ret = fwd_returns.loc[idx]

    log.info("Backtest period: %s ~ %s (%d days)", idx[0].date(), idx[-1].date(), len(idx))

    # ── Walk-forward backtest ─────────────────────────────────────────────────
    log.info("Running %d-split walk-forward (cost=%.4f roundtrip)...", N_SPLITS, config.TOTAL_COST_ROUNDTRIP)
    result: BacktestResult = run_walk_forward(sig, ret, n_splits=N_SPLITS)

    windows = _window_breakdown(sig, ret, N_SPLITS)
    annual = _annual_breakdown(
        _portfolio_pnl(sig, ret, config.TOTAL_COST_ROUNDTRIP)
    )
    benchmark = _spy_benchmark(close_df, fwd_returns.loc[idx])

    # ── Print report ──────────────────────────────────────────────────────────
    import sys, io
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("\n" + "=" * 65)
    print("  PHASE 2 — BASELINE BACKTEST REPORT")
    print(f"  Model: IC-Weighted Composite ({len(IC_WEIGHTS)} factors, cross-sectional)")
    print(f"  Cost: {config.TOTAL_COST_ROUNDTRIP:.4f} roundtrip | Splits: {N_SPLITS}")
    print("=" * 65)

    print("\n── Model Factors ──")
    for fname, w in IC_WEIGHTS.items():
        print(f"  {fname:<14} IC={w:+.3f}")

    print("\n── Walk-Forward Windows ──")
    print(f"{'Win':>3}  {'IS End':>12}  {'OOS Start':>10}  {'OOS End':>10}  {'IS Sh':>6}  {'OOS Sh':>6}  {'OOS Ret':>8}  OK")
    print("-" * 70)
    for w in windows:
        ok = "✅" if w["positive"] else "❌"
        print(f"  {w['window']:>1}  {w['is_end_date']:>12}  {w['oos_start_date']:>10}  "
              f"{w['oos_end_date']:>10}  {w['is_sharpe']:>6.3f}  {w['oos_sharpe']:>6.3f}  "
              f"{w['oos_return']:>7.1%}  {ok}")

    print("\n── OOS Aggregate Metrics ──")
    print(f"  Sharpe (ann.)    : {result.sharpe:.3f}")
    print(f"  Max Drawdown     : {result.mdd:.1%}")
    print(f"  Total Return     : {result.total_return:.1%}")
    print(f"  Trades (changes) : {result.n_trades:,}")
    print(f"  OOS / IS Ratio   : {result.oos_is_ratio:.3f}")
    print(f"  WF Positive      : {result.wf_positive_pct:.0%}  ({sum(w['positive'] for w in windows)}/{N_SPLITS})")

    if benchmark:
        print(f"\n── Benchmark: SPY Buy-and-Hold ──")
        print(f"  Sharpe  : {benchmark['sharpe']:.3f}")
        print(f"  Return  : {benchmark['total_return']:.1%}")
        print(f"  MDD     : {benchmark['mdd']:.1%}")

    print("\n── Annual Breakdown (full period incl. IS) ──")
    print(f"{'Year':>4}  {'Return':>8}  {'Sharpe':>7}")
    print("-" * 25)
    for _, row in annual.iterrows():
        print(f"  {int(row['year']):>4}  {row['return']:>7.1%}  {row['sharpe']:>7.3f}")

    print("\n── Tier 1 Gate Check ──")
    checks = [
        (f"Sharpe > {config.TIER1_SHARPE_MIN}",          result.sharpe > config.TIER1_SHARPE_MIN),
        (f"MDD < {config.TIER1_MDD_MAX:.0%}",            result.mdd < config.TIER1_MDD_MAX),
        (f"OOS/IS >= {config.TIER1_OOS_IS_RATIO_MIN}",   result.oos_is_ratio >= config.TIER1_OOS_IS_RATIO_MIN),
        ("WF positive >= 50%",                            result.wf_positive_pct >= 0.5),
        (f"n_trades >= {config.TIER1_MIN_TRADING_DAYS}", result.n_trades >= config.TIER1_MIN_TRADING_DAYS),
    ]
    for label, passed in checks:
        mark = "✅" if passed else "❌"
        print(f"  {mark}  {label}")

    passed_all = result.passes_tier1()
    print("\n" + "=" * 65)
    if passed_all:
        print("  VERDICT: ✅ TIER 1 PASS — Phase 4 페이퍼 트레이딩 진입 가능")
    else:
        n_failed = sum(1 for _, p in checks if not p)
        print(f"  VERDICT: ❌ TIER 1 FAIL ({n_failed}개 기준 미달) — 팩터/임계값 재검토")
    print("=" * 65)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for w in windows:
        rows.append({
            "section": "walk_forward",
            "window": w["window"],
            "is_end_date": w["is_end_date"],
            "oos_start_date": w["oos_start_date"],
            "oos_end_date": w["oos_end_date"],
            "is_sharpe": w["is_sharpe"],
            "oos_sharpe": w["oos_sharpe"],
            "oos_return": w["oos_return"],
        })
    for _, row in annual.iterrows():
        rows.append({"section": "annual", "year": int(row["year"]),
                     "return": row["return"], "sharpe": row["sharpe"]})
    rows.append({
        "section": "aggregate",
        "sharpe": round(result.sharpe, 4),
        "mdd": round(result.mdd, 4),
        "total_return": round(result.total_return, 4),
        "n_trades": result.n_trades,
        "oos_is_ratio": round(result.oos_is_ratio, 4),
        "wf_positive_pct": round(result.wf_positive_pct, 4),
        "tier1_pass": int(passed_all),
    })

    out_path = log_dir / "phase2_backtest_report.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    log.info("리포트 저장: %s", out_path)


if __name__ == "__main__":
    main()
