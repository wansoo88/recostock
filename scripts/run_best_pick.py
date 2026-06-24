#!/usr/bin/env python3
"""Print TODAY's single best-pick (long-shot) — the actionable output of the
signals.best_pick selector. RESEARCH / OPTIONAL satellite; does not touch the
live blend or send telegram. Run after `python -m data.collector` for fresh data.

Usage:
    python scripts/run_best_pick.py                 # both modes, weekly-pinned pick
    python scripts/run_best_pick.py --mode longshot
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd

from signals import best_pick

# The only tickers the two live modes need (disciplined = sectors,
# longshot = sectors + leveraged longs); SPY drives the trend gate.
LIVE_TICKERS = ["SPY", "QQQ"] + best_pick.SECTORS + best_pick.LEVER_LONGS


def load_close() -> pd.DataFrame:
    o = pd.read_parquet("data/raw/etf_ohlcv.parquet")
    etf = o["Close"] if isinstance(o.columns, pd.MultiIndex) else o
    try:
        s = pd.read_parquet("data/raw/single_stocks.parquet")
        stocks = s["Close"] if isinstance(s.columns, pd.MultiIndex) else s
        close = pd.concat([etf, stocks], axis=1)
    except FileNotFoundError:
        close = etf
    return close.loc[:, ~close.columns.duplicated()].sort_index()


def fetch_live_close() -> pd.DataFrame:
    """Fetch ~2y of fresh closes for the pick universe — in-memory only, never
    overwrites the committed live data file. Needs Yahoo access."""
    import yfinance as yf
    df = yf.download(LIVE_TICKERS, period="2y", interval="1d",
                     progress=False, auto_adjust=True)
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df
    return close.dropna(how="all").sort_index()


def show(res: dict) -> None:
    bt = res.get("backtest", {})
    mode = res.get("mode")
    print(f"\n{'='*72}")
    print(f"  BEST-PICK [{mode}]  ·  as of {res.get('asOf')}  ·  주간 교체(금요일 종가 고정)")
    print(f"  풀: {bt.get('pool')}  ·  랭킹: {bt.get('rank')}")
    print(f"{'='*72}")
    if res.get("pick"):
        lev = f" ({res['leverage']}x)" if res.get("leverage", 1) > 1 else ""
        print(f"  ▶ 오늘의 픽: {res['pick']}{lev}  {res.get('name','')}")
        print(f"     진입 ${res['entry']:.2f}  →  목표(TP +3%) ${res['tp']:.2f}  ·  "
              f"손절(200일선) ${res['stop']:.2f}  (여유 {res.get('distStopPct')}%)")
        if res.get("pickAsOf"):
            print(f"     선정: {res['pickAsOf']} 금요일 종가 기준 (주 1회 교체)")
    else:
        print(f"  ▶ 오늘의 픽: 현금 — {res.get('note','')}")
    print(f"\n  랭킹(점수 내림차순):")
    for r in res.get("ranked", [])[:6]:
        flag = "✓위" if r["above200"] else "✗아래"
        lev = f"{r['leverage']}x" if r["leverage"] > 1 else "  "
        print(f"     {r['ticker']:5s} {lev:3s} score {r['score']:>10.4f}  "
              f"200일선 {flag} (여유 {r['distStopPct']}%)")
    print(f"\n  검증(비용차감·look-ahead-safe, 2026-06-24):")
    print(f"     Full OOS 2021+  {bt.get('fullRet'):+}% / Sharpe {bt.get('fullSharpe')} / MDD {bt.get('fullMdd')}%")
    print(f"     Holdout  2024+  {bt.get('holdRet'):+}% / Sharpe {bt.get('holdSharpe')} / MDD {bt.get('holdMdd')}%")
    print(f"     연도 양수 {bt.get('yearsPositive')} · 2022 약세장 {bt.get('bear2022Pct'):+}% · 게이트 {bt.get('gate')}")
    print(f"  ⚠️ 주간 +3% 현실: 평균 +{bt.get('wkMeanPct')}%/주, +3% 달성 주 ~{bt.get('wkHit3Pct')}%, "
          f"최악 주 {bt.get('wkWorstPct')}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["disciplined", "longshot", "both"], default="both")
    ap.add_argument("--live", action="store_true",
                    help="fetch fresh closes from Yahoo (in-memory) instead of the cached parquet")
    args = ap.parse_args()
    close = fetch_live_close() if args.live else load_close()
    modes = ["disciplined", "longshot"] if args.mode == "both" else [args.mode]
    for m in modes:
        show(best_pick.select_weekly(close, m))
    print(f"\n  ※ 라이브 블렌드(추세코어85%+RSI슬리브15%)는 페이퍼 검증 중이라 불변. "
          f"이 픽은 참고용 새틀라이트 — 본인이 수동 실행.\n")


if __name__ == "__main__":
    main()
