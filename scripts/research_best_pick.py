#!/usr/bin/env python3
"""Best-pick "long shot" research — expand the universe, pick ONE name weekly,
measure it against a +3%/week target. RESEARCH ONLY (does not touch live config).

WHY THIS EXISTS
    A /goal asked: "expand the universe, pick the single best long-shot per day,
    run it toward a weekly +3% return target." That is a concentrated, high-
    variance bet — the opposite of the shipped 85/15 blend. Per CLAUDE.md app. B
    we do not move live capital on un-reproduced numbers, and the 3-month paper
    window (~2026-08-29) forbids changing live weights. So this script answers the
    question the disciplined way: a cost-adjusted, look-ahead-safe backtest that
    reports what a weekly single-pick strategy ACTUALLY delivers vs the 3% target,
    across an expanded candidate universe, with no live change.

WHAT IT MEASURES (per universe x ranking-signal x regime-gate cell)
    - weekly net return distribution: mean / median / P(week >= +3%) / worst week
    - daily-replay Sharpe / MDD / total net return (Full OOS 2021+, Holdout 2024+)
    - Tier-1 gate PASS/FAIL using config thresholds (Sharpe>=0.7, MDD<=25%, ...)
    The honest verdict: is +3%/week a sustainable MEAN, or only a hit-rate offset
    by an equally large downside?

DRIFT-ZERO / LOOK-AHEAD SAFETY
    Day-by-day causal replay (mirrors scripts/sweep_blend_goal.py). The pick is
    chosen each Friday from data truncated at that Friday's close and held the
    next week; daily returns are realised on day d+1; turnover is charged
    config.TOTAL_COST_ROUNDTRIP one-way on each weight change. Ranking signals
    (RSI-14, momentum) use only closes up to and including the decision date.

Run:
    python scripts/research_best_pick.py                # full research (needs data)
    python scripts/research_best_pick.py --self-test    # synthetic, no network
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

import config
from signals.sector_rotation import compute_rsi

ONEWAY = config.TOTAL_COST_ROUNDTRIP / 2
FULL_OOS_START = pd.Timestamp("2021-01-01")
HOLDOUT_START = pd.Timestamp("2024-01-01")
TRADING_DAYS = 252
WEEKLY_TARGET = 0.03               # the goal: +3% per week
SMA_WINDOW = 200

# Candidate universes (the "expanded" long-shot pools).
LEV_LONG = ["TQQQ", "SOXL", "SPXL", "QLD"]
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI"]
CORE = ["SPY", "QQQ"]


# ── data ────────────────────────────────────────────────────────────────────
def load_real():
    o = pd.read_parquet("data/raw/etf_ohlcv.parquet")
    etf = o["Close"] if isinstance(o.columns, pd.MultiIndex) else o
    try:
        s = pd.read_parquet("data/raw/single_stocks.parquet")
        stocks = s["Close"] if isinstance(s.columns, pd.MultiIndex) else s
    except FileNotFoundError:
        stocks = pd.DataFrame(index=etf.index)
    close = pd.concat([etf, stocks], axis=1)
    close = close.loc[:, ~close.columns.duplicated()].sort_index()
    vix = pd.read_parquet("data/raw/macro/vix.parquet").iloc[:, 0].dropna()
    cash = pd.read_parquet("data/raw/macro/yield_2y.parquet").iloc[:, 0].dropna()
    return close, vix, cash, sorted(stocks.columns.tolist())


def make_synthetic(seed=7, n=1600):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    close = pd.DataFrame(index=idx)
    for tk in CORE + LEV_LONG + SECTORS:
        mu, sig = 0.0004, (0.025 if tk in LEV_LONG else 0.012)
        close[tk] = 100 * np.exp(np.cumsum(rng.normal(mu, sig, n)))
    vix = pd.Series(15 + 6 * np.abs(rng.normal(0, 1, n)), index=idx)
    cash = pd.Series(4.5, index=idx)
    return close, vix, cash, []


# ── ranking signals (vectorised, all strictly causal) ───────────────────────
def precompute(close: pd.DataFrame):
    """Per-ticker causal indicators computed ONCE over the full series.

    Every value at date d uses only closes up to and including d (rolling/shift),
    so indexing by date in the replay loop introduces no look-ahead. Returns:
      scores : {signal_name: DataFrame(date x ticker)}  ranking score
      above  : DataFrame(date x ticker) bool             close > own 200d SMA
    """
    rsi = close.apply(compute_rsi)                       # per-column RSI-14
    sma = close.rolling(SMA_WINDOW).mean()
    mom12 = close / close.shift(60) - 1
    mom4 = close / close.shift(20) - 1
    enough = close.notna() & (close.rolling(SMA_WINDOW).count() >= SMA_WINDOW)
    scores = {
        "rsi": rsi.where(enough),
        "mom12w": mom12.where(enough),
        "mom4w": mom4.where(enough),
        "rsi_x_mom": (rsi * mom12).where(enough),
    }
    above = (close > sma) & enough
    return scores, above


# ── backtest: weekly single best-pick, daily replay ─────────────────────────
def run_best_pick(close, cash_yield, scores, above, pool, signal, gated):
    """Day-by-day causal replay. Each Friday pick the single top-scoring candidate
    (gated: SPY uptrend required + pick must be above own 200SMA, else cash).
    Hold to next Friday. Returns daily net-return Series + the pick log."""
    avail = [t for t in pool if t in close.columns]
    idx = close.index
    cash_daily = cash_yield.reindex(idx).ffill().fillna(0.0) / 100.0 / TRADING_DAYS
    fwd = close[avail].pct_change().shift(-1)
    score = scores[signal][avail]
    abv = above[avail]
    spy_up = above["SPY"] if "SPY" in above.columns else pd.Series(True, index=idx)

    start = SMA_WINDOW + 1
    prev_pick = cur_pick = None
    out_dates, out_rets, picks = [], [], []

    for i in range(start, len(idx) - 1):
        d = idx[i]
        if cur_pick is None or d.dayofweek == 4:        # weekly Friday rebalance
            best = None
            if (not gated) or bool(spy_up.iloc[i]):
                row = score.iloc[i]
                if gated:
                    row = row.where(abv.iloc[i])
                if row.notna().any():
                    best = str(row.idxmax())
            cur_pick = best                              # None -> cash this week
            picks.append((d, cur_pick))

        if cur_pick is not None and cur_pick in fwd.columns and not np.isnan(fwd[cur_pick].iloc[i]):
            gross = float(fwd[cur_pick].iloc[i])
        else:
            gross = float(cash_daily.iloc[i])            # parked
        turnover = 0.0 if cur_pick == prev_pick else 1.0  # full switch (100% one name)
        net = gross - turnover * ONEWAY
        out_dates.append(idx[i + 1])
        out_rets.append(net)
        prev_pick = cur_pick

    return pd.Series(out_rets, index=pd.DatetimeIndex(out_dates)), picks


# ── metrics ──────────────────────────────────────────────────────────────────
def perf(daily: pd.Series) -> dict:
    if len(daily) < 2 or daily.std() == 0:
        return {"ret": 0.0, "sharpe": 0.0, "mdd": 0.0, "n": len(daily)}
    eq = (1 + daily).cumprod()
    mdd = float((eq / eq.cummax() - 1).min())
    sharpe = float(daily.mean() / daily.std() * np.sqrt(TRADING_DAYS))
    return {"ret": float(eq.iloc[-1] - 1), "sharpe": sharpe, "mdd": mdd, "n": len(daily)}


def weekly_stats(daily: pd.Series) -> dict:
    """Compound daily net returns into ISO weeks; profile vs the +3% target."""
    if daily.empty:
        return {"mean": 0, "median": 0, "pHit": 0, "worst": 0, "best": 0, "n": 0}
    wk = (1 + daily).groupby([daily.index.isocalendar().year,
                              daily.index.isocalendar().week]).prod() - 1
    return {
        "mean": float(wk.mean()), "median": float(wk.median()),
        "pHit": float((wk >= WEEKLY_TARGET).mean()),
        "worst": float(wk.min()), "best": float(wk.max()), "n": int(len(wk)),
    }


def wf_positive_frac(daily: pd.Series):
    if daily.empty:
        return 0.0, 0
    by_year = (1 + daily).groupby(daily.index.year).prod() - 1
    return (int((by_year > 0).sum()) / len(by_year) if len(by_year) else 0.0), len(by_year)


def gate(full, daily_full, is_sharpe):
    wf_frac, _ = wf_positive_frac(daily_full)
    oos_is = (full["sharpe"] / is_sharpe) if is_sharpe > 0 else float("inf")
    checks = {
        "sharpe": full["sharpe"] >= config.TIER1_SHARPE_MIN,
        "mdd": abs(full["mdd"]) <= config.TIER1_MDD_MAX,
        "oosis": oos_is >= config.TIER1_OOS_IS_RATIO_MIN,
        "days": full["n"] >= config.TIER1_MIN_TRADING_DAYS,
        "wf": wf_frac >= config.TIER1_WF_POSITIVE_MIN,
    }
    return all(checks.values()), checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", default="data/logs/best_pick_research.csv")
    args = ap.parse_args()

    if args.self_test:
        close, vix, cash, stock_list = make_synthetic()
    else:
        close, vix, cash, stock_list = load_real()
    scores, above = precompute(close)

    pools = {
        "lev4": LEV_LONG,
        "sectors": SECTORS,
        "sectors+lev": SECTORS + LEV_LONG,
        "core+lev": CORE + LEV_LONG,
    }
    if stock_list:
        pools["stocks50"] = stock_list
        pools["all"] = sorted(set(SECTORS + LEV_LONG + CORE + stock_list))

    signals = ["rsi", "mom12w", "mom4w", "rsi_x_mom"]
    gates = [True, False]

    print(f"\n{'='*118}")
    print(f"BEST-PICK 'LONG SHOT' RESEARCH  ·  weekly single pick  ·  target +{WEEKLY_TARGET:.0%}/wk  ·  "
          f"cost {config.TOTAL_COST_ROUNDTRIP:.2%} rt  ·  "
          f"{'SYNTHETIC' if args.self_test else 'real'}  ·  span {str(close.index[0])[:10]}→{str(close.index[-1])[:10]}")
    print('='*118)
    print(f"{'pool':>12} {'signal':>9} {'gate':>5} | "
          f"{'wkMean':>7}{'wkMed':>7}{'P>=3%':>7}{'worst':>7} | "
          f"{'Full ret':>9}{'Shp':>6}{'MDD':>7} | {'Hold ret':>9}{'Shp':>6}{'MDD':>7} | {'gate':>5}")
    print('-'*118)

    rows = []
    for pname, pool in pools.items():
        for sig in signals:
            for g in gates:
                daily, picks = run_best_pick(close, cash, scores, above, pool, sig, g)
                full = perf(daily[daily.index >= FULL_OOS_START])
                hold = perf(daily[daily.index >= HOLDOUT_START])
                is_sharpe = perf(daily[daily.index < FULL_OOS_START])["sharpe"]
                wk = weekly_stats(daily[daily.index >= FULL_OOS_START])
                gp, _ = gate(full, daily[daily.index >= FULL_OOS_START], is_sharpe)
                print(f"{pname:>12} {sig:>9} {'Y' if g else 'N':>5} | "
                      f"{wk['mean']*100:>+6.2f}%{wk['median']*100:>+6.2f}%{wk['pHit']*100:>6.0f}%{wk['worst']*100:>+6.0f}% | "
                      f"{full['ret']*100:>+8.0f}%{full['sharpe']:>6.2f}{full['mdd']*100:>+6.0f}% | "
                      f"{hold['ret']*100:>+8.0f}%{hold['sharpe']:>6.2f}{hold['mdd']*100:>+6.0f}% | "
                      f"{'PASS' if gp else 'FAIL':>5}", flush=True)
                rows.append({
                    "pool": pname, "signal": sig, "gated": g,
                    "wk_mean": round(wk["mean"], 4), "wk_median": round(wk["median"], 4),
                    "wk_p_hit_3pct": round(wk["pHit"], 4), "wk_worst": round(wk["worst"], 4),
                    "wk_best": round(wk["best"], 4), "wk_n": wk["n"],
                    "full_ret": round(full["ret"], 4), "full_sharpe": round(full["sharpe"], 3),
                    "full_mdd": round(full["mdd"], 4),
                    "hold_ret": round(hold["ret"], 4), "hold_sharpe": round(hold["sharpe"], 3),
                    "hold_mdd": round(hold["mdd"], 4), "gate_pass": gp,
                })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print('-'*118)
    print(f"wrote {out}  ({len(rows)} cells)")
    if args.self_test:
        print("NOTE: synthetic — numbers meaningless; proves the harness replays.")


if __name__ == "__main__":
    main()
