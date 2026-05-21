#!/usr/bin/env python3
"""Realized win-rate / payoff / EV by confidence (proba) bucket for conviction_v4.

Simulates the ACTUAL exit policy (TP +3% / SL -1% intraday, else exit at close
after up to 5 trading days) on daily OHLC bars, applying the production filter
(5 regime gates + Multi-EMA confirm), then tabulates win-rate by EMA-5 proba
bucket. Replaces the single static CONVICTION_EXPECTED_WINRATE so higher-
conviction signals report (and can be ranked by) their true higher win-rate.

Conservative tie-break: if a bar touches both TP and SL, assume SL first.

Run: python scripts/conviction_winrate_buckets.py
"""
from __future__ import annotations
import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import config

LONG = ["SPY", "QQQ", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLI"]
HOLD = 5
TP, SL, COST = config.CONVICTION_TP_PCT, config.CONVICTION_SL_PCT, config.TOTAL_COST_ROUNDTRIP
THR = config.CONVICTION_THRESHOLD
HOLDOUT = pd.Timestamp("2024-01-01")
MACRO = Path("data/raw/macro")


def _s(n):
    return pd.read_parquet(MACRO / f"{n}.parquet").iloc[:, 0].dropna()


def _z(s, w=60):
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def regime_pass(close) -> pd.Series:
    """Boolean series: do all 5 conviction_v4 regime gates pass on each date?"""
    spy = close["SPY"]
    vix = _s("vix"); vix9d = _s("vix9d")
    g_vix = vix.reindex(spy.index) < config.CONVICTION_VIX_MAX
    g_spy = spy > spy.rolling(200).mean()
    g_term = (vix9d / vix).reindex(spy.index) < config.CONVICTION_VIX_TERM_MAX
    g_skew = _z(_s("skew")).reindex(spy.index) < config.CONVICTION_SKEW_Z_MAX
    g_move = _z(_s("move")).reindex(spy.index) < config.CONVICTION_MOVE_Z_MAX
    return (g_vix & g_spy & g_term & g_skew & g_move).fillna(False)


def simulate():
    proba = pd.read_parquet("data/logs/phase3_v3_oos_proba.parquet")["proba"].unstack(level=1)
    ema5 = proba.ewm(span=5).mean(); ema3 = proba.ewm(span=3).mean(); ema7 = proba.ewm(span=7).mean()
    o = pd.read_parquet("data/raw/etf_ohlcv.parquet")
    close, high, low = o["Close"], o["High"], o["Low"]
    rp = regime_pass(close)

    rows = []
    cols = [c for c in LONG if c in ema5.columns and c in close.columns]
    idx = close.index
    for tk in cols:
        e5, e3, e7 = ema5[tk], ema3[tk], ema7[tk]
        c, h, l = close[tk], high[tk], low[tk]
        for i in range(len(idx) - HOLD):
            d = idx[i]
            if d not in e5.index or np.isnan(e5.get(d, np.nan)):
                continue
            p5 = e5.loc[d]
            if p5 < 0.50:  # ignore clearly non-bullish for bucketing
                continue
            # multi-EMA confirm + regime gate (production conditions)
            if not (e3.loc[d] >= THR and e7.loc[d] >= THR and p5 >= THR):
                pass_full = False
            else:
                pass_full = True
            if not rp.get(d, False):
                continue  # regime must pass (matches production)
            entry = c.iloc[i]
            if entry <= 0:
                continue
            tp_px, sl_px = entry * (1 + TP), entry * (1 - SL)
            ret = None
            for hh in range(1, HOLD + 1):
                if l.iloc[i + hh] <= sl_px:        # SL first (conservative)
                    ret = -SL - COST; break
                if h.iloc[i + hh] >= tp_px:
                    ret = TP - COST; break
                if hh == HOLD:
                    ret = (c.iloc[i + hh] / entry - 1) - COST
            rows.append({"date": d, "ticker": tk, "proba": float(p5),
                         "ret": ret, "confirmed": pass_full})
    return pd.DataFrame(rows)


def tabulate(df, label, confirmed_only=True):
    if confirmed_only:
        df = df[df["confirmed"]]
    buckets = [(0.65, 0.70), (0.70, 0.75), (0.75, 1.01)]
    print(f"\n=== {label} (n_total={len(df)}) — exit TP{TP:.0%}/SL{SL:.0%}, cost {COST:.2%} ===")
    print(f"{'bucket':>12} {'n':>4} {'WR':>7} {'avgW%':>7} {'avgL%':>7} {'payoff':>7} {'EV%':>8}")
    for lo, hi in buckets:
        sub = df[(df["proba"] >= lo) & (df["proba"] < hi)]
        n = len(sub)
        if n == 0:
            print(f"  [{lo:.2f},{hi:.2f}) {0:>4}   —"); continue
        r = sub["ret"]; wins = r[r > 0]; losses = r[r <= 0]
        wr = len(wins) / n
        aw = wins.mean() * 100 if len(wins) else 0
        al = abs(losses.mean()) * 100 if len(losses) else 0
        payoff = (aw / al) if al > 0 else float("nan")
        ev = r.mean() * 100
        print(f"  [{lo:.2f},{hi:.2f}) {n:>4} {wr:>6.1%} {aw:>+6.2f} {al:>+6.2f} {payoff:>7.2f} {ev:>+7.3f}")


if __name__ == "__main__":
    df = simulate()
    print(f"Total simulated entries (regime-pass, proba>=0.50): {len(df)}")
    tabulate(df, "FULL OOS — confirmed (multi-EMA+regime)", True)
    tabulate(df[df["date"] >= HOLDOUT], "HOLDOUT 2024+ — confirmed", True)
    tabulate(df, "FULL OOS — proba only (no multi-EMA confirm req)", False)
