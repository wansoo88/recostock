#!/usr/bin/env python3
"""Phase 1: Factor IC analysis script. Run locally after data collection.

Usage:
    python scripts/run_phase1_ic.py

Output:
    - Console table: factor IC / ICIR / p-value / verdict
    - data/logs/phase1_ic_report.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Windows CP949 콘솔에서도 UTF-8 출력 가능하도록
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
log = logging.getLogger(__name__)


def main() -> None:
    # ── Load data ─────────────────────────────────────────────────────────────
    raw_path = Path("data/raw/etf_ohlcv.parquet")
    if not raw_path.exists():
        log.error("데이터 없음. 먼저 실행: python -m data.collector")
        sys.exit(1)

    log.info("Loading ETF OHLCV data...")
    ohlcv = pd.read_parquet(raw_path)

    # Extract Close prices — handle both flat and MultiIndex columns
    if isinstance(ohlcv.columns, pd.MultiIndex):
        close_df = ohlcv["Close"]
    else:
        close_df = ohlcv

    log.info("Data shape: %s | Tickers: %d | Date range: %s ~ %s",
             close_df.shape, len(close_df.columns),
             close_df.index.min().date(), close_df.index.max().date())

    # ── IC Analysis ───────────────────────────────────────────────────────────
    from features.ic_analysis import run_full_ic_analysis

    log.info("Computing IC for horizons [1d, 5d]... (takes ~30s)")
    report = run_full_ic_analysis(close_df, horizons=[1, 5])

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  PHASE 1 — FACTOR IC REPORT")
    print("  Rejection threshold: |mean IC| < 0.01  or  p-value >= 0.05")
    print("=" * 75)

    h1 = report[report["horizon"] == 1].reset_index(drop=True)
    h5 = report[report["horizon"] == 5].reset_index(drop=True)

    for label, df in [("1-DAY FORWARD", h1), ("5-DAY FORWARD", h5)]:
        print(f"\n── {label} ──")
        print(f"{'Factor':<20} {'MeanIC':>8} {'ICIR':>7} {'p-val':>7} {'N':>5}  Verdict")
        print("-" * 60)
        for _, row in df.iterrows():
            verdict_mark = "✅" if row["verdict"] == "KEEP" else "❌"
            print(f"{row['factor']:<20} {row['mean_ic']:>8.4f} {row['icir']:>7.3f} "
                  f"{row['p_value']:>7.4f} {row['n_obs']:>5}  {verdict_mark} {row['verdict']}")

    keeps = report[report["verdict"] == "KEEP"]["factor"].unique()
    rejects = report[report["verdict"] == "REJECT"]["factor"].unique()

    print("\n" + "=" * 75)
    print(f"KEEP   ({len(keeps)}): {', '.join(keeps) if len(keeps) else '없음'}")
    print(f"REJECT ({len(rejects)}): {', '.join(rejects) if len(rejects) else '없음'}")
    print("=" * 75)

    # ── Save report ───────────────────────────────────────────────────────────
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / "phase1_ic_report.csv"
    report.to_csv(out_path, index=False)
    log.info("리포트 저장: %s", out_path)

    if len(keeps) == 0:
        print("\n⚠️  유효 팩터 없음 — 팩터 설계 재검토 필요. Phase 1 재시도.")
    else:
        print(f"\n✅ Phase 1 완료 — {len(keeps)}개 팩터 유효. Phase 2 (베이스라인 백테스트)로 진행 가능.")


if __name__ == "__main__":
    main()
