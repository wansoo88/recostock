#!/usr/bin/env python3
"""Research: vol-adaptive trend filter + alternate leverage ETF + multi-asset core.

Three pieces, tested on the same trend-core + fear-dip-tilt + cash-yield baseline:
  #1 vol-adaptive: 200SMA in low-VIX regime, golden-cross (50&200) in high-VIX
  #2 leverage instrument: SPXL (3x SPY, current) vs UPRO (3x SPY alt) vs SSO (2x SPY)
  #3 multi-asset core: SPY only, SPY/QQQ 50/50, momentum rotation
"""
from __future__ import annotations
import io, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import config
from scripts.simulate_trading import feardip_trades, close

ONEWAY = config.TOTAL_COST_ROUNDTRIP / 2
_IRX = pd.read_parquet("data/raw/macro/yield_2y.parquet").iloc[:, 0].dropna()
_VIX = pd.read_parquet("data/raw/macro/vix.parquet").iloc[:, 0].dropna()


def _fetch(tk):
    import yfinance as yf
    df = yf.download(tk, start="2015-01-01", auto_adjust=True, progress=False, threads=False)
    return df["Close"].squeeze().dropna()


# bring in SSO / UPRO if missing
EXTRA = {}
for tk in ["SSO", "UPRO"]:
    if tk not in close.columns:
        try:
            EXTRA[tk] = _fetch(tk)
            print(f"fetched {tk}: {len(EXTRA[tk])} bars")
        except Exception as e:
            print(f"failed {tk}: {e}")

SPY = close["SPY"].dropna()
QQQ = close["QQQ"].dropna()
SPXL = close["SPXL"].dropna()


def _fd_mask(idx_):
    m = pd.Series(False, index=idx_); s = list(idx_)
    for t in feardip_trades():
        if t["entry_date"] in idx_:
            i = s.index(t["entry_date"])
            for h in range(10):
                if i + h < len(s):
                    m.iloc[i + h] = True
    return m


def _cash_y(d):
    v = _IRX.asof(d); return (v / 100 / 252) if pd.notna(v) else 0.0


def _trend_filter(price, kind, idx):
    p = price.reindex(idx); s200 = p.rolling(200).mean(); s50 = p.rolling(50).mean()
    if kind == "200":
        return (p.shift(1) > s200.shift(1)).fillna(False)
    if kind == "golden":
        return ((p.shift(1) > s50.shift(1)) & (s50.shift(1) > s200.shift(1))).fillna(False)
    if kind == "adaptive_vix20":
        vix = _VIX.reindex(idx).ffill()
        high_vol = vix > 20
        f200 = (p.shift(1) > s200.shift(1)).fillna(False)
        fgolden = ((p.shift(1) > s50.shift(1)) & (s50.shift(1) > s200.shift(1))).fillna(False)
        return fgolden.where(high_vol, f200)
    if kind == "adaptive_vix22":
        vix = _VIX.reindex(idx).ffill()
        high_vol = vix > 22
        f200 = (p.shift(1) > s200.shift(1)).fillna(False)
        fgolden = ((p.shift(1) > s50.shift(1)) & (s50.shift(1) > s200.shift(1))).fillna(False)
        return fgolden.where(high_vol, f200)


def sim(price, lev_price, w_lev, filt_kind, since, base_label=""):
    idx = price.index.intersection(lev_price.index)
    p = price.reindex(idx); lp = lev_price.reindex(idx)
    pret = p.pct_change(); lret = lp.pct_change()
    core_on = _trend_filter(price, filt_kind, idx)
    fd = _fd_mask(idx)
    idx_s = idx[idx >= since]
    cap = 10000; peak = cap; mdd = 0; pw = lw = 0; rets = []
    for d in idx_s:
        c = bool(core_on.loc[d]); f = bool(fd.loc[d])
        if c and f:    spy_w, x_w = 1 - w_lev, w_lev
        elif c:        spy_w, x_w = 1.0, 0.0
        elif f:        spy_w, x_w = 1.0, 0.0
        else:          spy_w, x_w = 0.0, 0.0
        cash_w = max(0, 1 - pw - lw)
        turn = abs(spy_w - pw) + abs(x_w - lw)
        d_ret = pw * (pret.get(d, 0) or 0) + lw * (lret.get(d, 0) or 0) + cash_w * _cash_y(d) - turn * ONEWAY
        cap *= (1 + d_ret); peak = max(peak, cap); mdd = min(mdd, cap/peak - 1); rets.append(d_ret)
        pw, lw = spy_w, x_w
    r = pd.Series(rets); sh = r.mean()/r.std()*np.sqrt(252) if r.std() > 0 else 0
    return cap/10000 - 1, sh, mdd


def line(tot, sh, mdd):
    return f"{tot*100:>+7.1f}% Sharpe {sh:>+5.2f} MDD {mdd*100:>+6.1f}%"


def run(since, label):
    print(f"\n=== {label} ===")
    print("\n[#1 VOL-ADAPTIVE FILTER] (SPY core, SPXL 15% tilt)")
    for kind in ["200", "golden", "adaptive_vix20", "adaptive_vix22"]:
        print(f"  {kind:18} {line(*sim(SPY, SPXL, 0.15, kind, since))}")

    print("\n[#2 LEVERAGE INSTRUMENT] (SPY core, 200SMA, ~1.3x effective)")
    for tk, mult in [("SPXL", 3), ("UPRO", 3), ("SSO", 2)]:
        lp = close[tk].dropna() if tk in close.columns else EXTRA.get(tk)
        if lp is None: print(f"  {tk:6} unavailable"); continue
        w = 0.30 / (mult - 1)  # to achieve 1.3x effective
        tot, sh, mdd = sim(SPY, lp, w, "200", since)
        print(f"  {tk:6} ({mult}x, w={w:.2f}) {line(tot, sh, mdd)}")

    print("\n[#3 MULTI-ASSET CORE]")
    # 50/50 blend: simulate two trend cores combined
    def blend_sim(w_spy):
        idx = SPY.index.intersection(QQQ.index).intersection(SPXL.index)
        p_spy = SPY.reindex(idx); p_qqq = QQQ.reindex(idx); p_spxl = SPXL.reindex(idx)
        s_spy = (p_spy.shift(1) > p_spy.rolling(200).mean().shift(1)).fillna(False)
        s_qqq = (p_qqq.shift(1) > p_qqq.rolling(200).mean().shift(1)).fillna(False)
        fd = _fd_mask(idx)
        r_spy = p_spy.pct_change(); r_qqq = p_qqq.pct_change(); r_spxl = p_spxl.pct_change()
        idx_s = idx[idx >= since]
        cap = 10000; peak = cap; mdd = 0; rets = []
        prev_w_spy = prev_w_qqq = prev_w_spxl = 0
        for d in idx_s:
            on_s = bool(s_spy.loc[d]); on_q = bool(s_qqq.loc[d]); f = bool(fd.loc[d])
            # split capital: w_spy to SPY sleeve, (1-w_spy) to QQQ sleeve
            # within each sleeve: full asset if core_on, cash else; tilt only on SPY sleeve
            tgt_spy = w_spy if on_s else 0
            tgt_qqq = (1 - w_spy) if on_q else 0
            tgt_spxl = 0
            if on_s and f:
                tgt_spxl = w_spy * 0.15
                tgt_spy = w_spy * 0.85
            cash_w = max(0, 1 - prev_w_spy - prev_w_qqq - prev_w_spxl)
            turn = abs(tgt_spy - prev_w_spy) + abs(tgt_qqq - prev_w_qqq) + abs(tgt_spxl - prev_w_spxl)
            d_ret = (prev_w_spy * (r_spy.get(d, 0) or 0)
                     + prev_w_qqq * (r_qqq.get(d, 0) or 0)
                     + prev_w_spxl * (r_spxl.get(d, 0) or 0)
                     + cash_w * _cash_y(d) - turn * ONEWAY)
            cap *= (1 + d_ret); peak = max(peak, cap); mdd = min(mdd, cap/peak - 1); rets.append(d_ret)
            prev_w_spy, prev_w_qqq, prev_w_spxl = tgt_spy, tgt_qqq, tgt_spxl
        r = pd.Series(rets); sh = r.mean()/r.std()*np.sqrt(252) if r.std() > 0 else 0
        return cap/10000 - 1, sh, mdd

    for w in [1.0, 0.5, 0.0]:
        tot, sh, mdd = blend_sim(w)
        lab = f"SPY {int(w*100)}% / QQQ {int((1-w)*100)}%"
        print(f"  {lab:18} {line(tot, sh, mdd)}")


if __name__ == "__main__":
    run(pd.Timestamp("2021-01-01"), "FULL OOS 2021+")
    run(pd.Timestamp("2024-01-01"), "HOLDOUT 2024+")
