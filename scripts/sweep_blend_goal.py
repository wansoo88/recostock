#!/usr/bin/env python3
"""Reproducible blend-goal sweep — how high can the live 'goal' (risk-adjusted
return ceiling) be pushed *honestly*?

WHY THIS EXISTS
    The shipped sleeve weight (signals/portfolio.SECTOR_SLEEVE_WEIGHT = 0.15) is
    a conservative knee. Docstrings claim 0.25 lifts Sharpe at the same MDD, and
    portfolio.py literally says "Bump to 0.25 for the fuller tilt" — but those
    numbers came from a one-off 2026-05-30 run with NO committed reproduction
    script, and the engine figure is quoted as 114 / 124 / 132 % across three
    files (stale, inconsistent). Per CLAUDE.md appendix B we do NOT raise a
    real-money leverage/sleeve knob on unverified numbers. This script turns the
    claim into a permanent, auditable, cost-adjusted, look-ahead-safe backtest.

DRIFT-ZERO PRINCIPLE
    The backtest replays the EXACT production functions on expanding windows —
    signals.trend_core.evaluate, signals.sector_rotation.evaluate,
    signals.portfolio.compose — so there is no re-implementation drift. Whatever
    the user gets live on each daily run is what this sweeps.

    For each trading day d:
      tc  = trend_core.evaluate(close[:d], fear_dip_active(d), vix.asof(d))
      sat = sector_rotation.evaluate(close[:d])      # weekly (Fri) rebalance
      w   = portfolio.compose(tc, sat, sleeve_weight) # {ticker: capital_frac}
    then realise w on day d+1 actual returns (SPXL/sectors are REAL tickers, not
    synthetic leverage), cash earns a short-Treasury daily yield (2Y proxy; see
    load_real), and turnover is
    charged config.TOTAL_COST_ROUNDTRIP one-way on each weight change.

OUTPUT
    Full OOS (2021+) and Holdout (2024+): net return / Sharpe / MDD, walk-forward
    yearly fold signs, and a Tier-1 gate PASS/FAIL using the config thresholds.
    Writes data/logs/blend_goal_sweep.csv. The shipped cell (0.15 / STRONG 0.20)
    is printed first as a reproduction check against the docstring figures.

ENVIRONMENT NOTE
    Needs data/raw/etf_ohlcv.parquet + data/raw/macro/*.parquet (rebuild with
    `python -m data.collector`). The cloud planning sandbox blocks market-data
    hosts, so run this in CI (.github/workflows/blend_goal_sweep.yml) or locally,
    where yfinance reaches Yahoo. Use `--self-test` for a network-free smoke test
    on synthetic prices that verifies the plumbing end to end.

Run:
    python scripts/sweep_blend_goal.py                 # full sweep (needs data)
    python scripts/sweep_blend_goal.py --self-test     # synthetic, no network
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
from signals import portfolio, sector_rotation, trend_core

ONEWAY = config.TOTAL_COST_ROUNDTRIP / 2          # cost charged per |Δweight|
FULL_OOS_START = pd.Timestamp("2021-01-01")
HOLDOUT_START = pd.Timestamp("2024-01-01")
TRADING_DAYS = 252

# Tickers that can appear in a composed allocation (engine + sector sleeve).
ALLOC_TICKERS = ["SPY", "QQQ", "SPXL", "XLK", "XLF", "XLE", "XLV", "XLY", "XLI"]

# Sweep grid. STRONG_SPXL is monkey-patched onto the trend_core module per cell
# (it reads the module global); sleeve_weight is a direct compose() argument.
SLEEVE_GRID = [0.15, 0.20, 0.25]
STRONG_GRID = [0.20, 0.25]
SHIPPED = (0.15, 0.20)                             # current live config


# ── data loading ──────────────────────────────────────────────────────────────
def load_real():
    """Load committed/rebuilt parquet. Raises if data is absent (sandbox)."""
    raw = Path("data/raw/etf_ohlcv.parquet")
    if not raw.exists():
        raise FileNotFoundError(
            "data/raw/etf_ohlcv.parquet missing — run `python -m data.collector` "
            "first (needs Yahoo access; blocked in the cloud sandbox)."
        )
    o = pd.read_parquet(raw)
    close = o["Close"] if isinstance(o.columns, pd.MultiIndex) else o
    vix = pd.read_parquet("data/raw/macro/vix.parquet").iloc[:, 0].dropna()
    # Cash-leg yield: the 2Y Treasury, which data.collector keeps fresh. The truer
    # cash proxy is the 13-week bill (^IRX, macro/irx.parquet) but it lags (~2wk
    # stale) and is numerically near-identical here (mean ~2.09 vs ~2.10), so the
    # 2Y is the safer, fresher feed.
    cash_yield = pd.read_parquet("data/raw/macro/yield_2y.parquet").iloc[:, 0].dropna()
    return close, vix, cash_yield


def make_synthetic(seed: int = 7, n: int = 1600):
    """Deterministic geometric-brownian prices for a network-free smoke test.

    Mild upward drift + a vol regime so the trend filter and the calm-boost both
    activate. NOT a benchmark — only proves the harness plumbing runs.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    close = pd.DataFrame(index=idx)
    for tk in ALLOC_TICKERS:
        mu, sig = 0.0004, (0.02 if tk in ("SPXL",) else 0.011)
        rets = rng.normal(mu, sig, n)
        close[tk] = 100 * np.exp(np.cumsum(rets))
    vix = pd.Series(15 + 6 * np.abs(rng.normal(0, 1, n)), index=idx, name="VIX")
    cash_yield = pd.Series(4.5, index=idx)        # flat ~4.5% short rate
    return close, vix, cash_yield


def feardip_mask(index: pd.DatetimeIndex, synthetic: bool) -> pd.Series:
    """Boolean: is an uptrend fear-dip tilt active on each date (10-day windows)?

    Real path reuses the production fear-dip detector (scripts.simulate_trading)
    so the tilt days match live exactly. Synthetic path returns all-False (the
    smoke test only exercises the always-on / calm-boost branches).
    """
    mask = pd.Series(False, index=index)
    if synthetic:
        return mask
    from scripts.simulate_trading import feardip_trades  # lazy: reads parquet
    days = list(index)
    for t in feardip_trades():
        if t["entry_date"] in index:
            i = days.index(t["entry_date"])
            for h in range(10):
                if i + h < len(days):
                    mask.iloc[i + h] = True
    return mask


# ── backtest ───────────────────────────────────────────────────────────────────
def run_blend(close, vix, cash_yield, fd_mask, sleeve_weight, strong_spxl):
    """Day-by-day causal replay of the production blend. Returns a daily net-return
    Series indexed by the realisation date (d+1)."""
    # Per-cell override of the calm-boost SPXL fraction the engine reads.
    trend_core.STRONG_SPXL = strong_spxl

    idx = close.index
    cash_daily = cash_yield.reindex(idx).ffill().fillna(0.0) / 100.0 / TRADING_DAYS
    avail = [t for t in ALLOC_TICKERS if t in close.columns]
    fwd = close[avail].pct_change().shift(-1)           # day d -> realised on d+1

    start = trend_core.SMA_WINDOW + 1
    prev_w: dict[str, float] = {}
    last_sat: dict | None = None
    out_dates, out_rets = [], []

    for i in range(start, len(idx) - 1):
        d = idx[i]
        window = close.iloc[: i + 1]
        # Sleeve rebalances weekly (Fridays) — hold the prior pick in between.
        if last_sat is None or d.dayofweek == 4:
            last_sat = sector_rotation.evaluate(window)
        vix_d = float(vix.asof(d)) if len(vix) else None
        tc = trend_core.evaluate(window, bool(fd_mask.get(d, False)), vix_d)
        blend = portfolio.compose(tc, last_sat, sleeve_weight=sleeve_weight)
        w = blend.get("weights", {})

        # gross next-day return: invested legs + cash on the idle remainder
        gross = sum(wt * float(fwd[tk].iloc[i]) for tk, wt in w.items()
                    if tk in fwd.columns and not np.isnan(fwd[tk].iloc[i]))
        gross += blend.get("cashWeight", 0.0) * float(cash_daily.iloc[i])
        # turnover cost: one-way on each changed asset weight
        names = set(prev_w) | set(w)
        turnover = sum(abs(w.get(tk, 0.0) - prev_w.get(tk, 0.0)) for tk in names)
        net = gross - turnover * ONEWAY

        out_dates.append(idx[i + 1])
        out_rets.append(net)
        prev_w = w

    return pd.Series(out_rets, index=pd.DatetimeIndex(out_dates))


def perf(daily: pd.Series) -> dict:
    """Cost-adjusted total return / annualised Sharpe / max drawdown for a slice."""
    if len(daily) < 2 or daily.std() == 0:
        return {"ret": 0.0, "sharpe": 0.0, "mdd": 0.0, "n": len(daily)}
    eq = (1 + daily).cumprod()
    peak = eq.cummax()
    mdd = float((eq / peak - 1).min())
    sharpe = float(daily.mean() / daily.std() * np.sqrt(TRADING_DAYS))
    return {"ret": float(eq.iloc[-1] - 1), "sharpe": sharpe, "mdd": mdd, "n": len(daily)}


def wf_positive_frac(daily: pd.Series) -> tuple[float, int]:
    """Fraction of calendar-year walk-forward folds with positive net return."""
    if daily.empty:
        return 0.0, 0
    by_year = (1 + daily).groupby(daily.index.year).prod() - 1
    pos = int((by_year > 0).sum())
    return (pos / len(by_year) if len(by_year) else 0.0), len(by_year)


def gate_check(full: dict, holdout: dict, daily_full: pd.Series, is_sharpe: float) -> dict:
    """Tier-1 phase gate, evaluated on the Full-OOS slice (config thresholds)."""
    wf_frac, n_folds = wf_positive_frac(daily_full)
    oos_is = (full["sharpe"] / is_sharpe) if is_sharpe > 0 else float("inf")
    checks = {
        "sharpe>=0.7": full["sharpe"] >= config.TIER1_SHARPE_MIN,
        "mdd<25%": abs(full["mdd"]) <= config.TIER1_MDD_MAX,
        "oos/is>=0.4": oos_is >= config.TIER1_OOS_IS_RATIO_MIN,
        "days>=120": full["n"] >= config.TIER1_MIN_TRADING_DAYS,
        f"wf+>={config.TIER1_WF_POSITIVE_MIN:.0%}": wf_frac >= config.TIER1_WF_POSITIVE_MIN,
    }
    return {"pass": all(checks.values()), "checks": checks,
            "wfFrac": wf_frac, "wfFolds": n_folds, "oosIs": oos_is}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="synthetic prices, no network — verify the harness runs")
    ap.add_argument("--out", default="data/logs/blend_goal_sweep.csv")
    args = ap.parse_args()

    synthetic = args.self_test
    close, vix, cash_yield = make_synthetic() if synthetic else load_real()
    have = [t for t in ALLOC_TICKERS if t in close.columns]
    missing = [t for t in ALLOC_TICKERS if t not in have]
    if "SPY" in missing:
        raise RuntimeError("SPY absent — cannot run the trend core.")
    if missing:
        print(f"WARN: missing alloc tickers (treated as unavailable): {missing}")

    fd = feardip_mask(close.index, synthetic)
    shipped_strong = trend_core.STRONG_SPXL        # restore after sweep

    rows = []
    # shipped cell first (reproduction check), then the rest of the grid
    cells = [SHIPPED] + [(s, st) for s in SLEEVE_GRID for st in STRONG_GRID
                         if (s, st) != SHIPPED]
    print(f"\n{'='*100}")
    print(f"BLEND-GOAL SWEEP  ·  cost {config.TOTAL_COST_ROUNDTRIP:.2%} roundtrip  ·  "
          f"{'SYNTHETIC self-test' if synthetic else 'real data'}  ·  "
          f"span {str(close.index[0])[:10]}→{str(close.index[-1])[:10]}")
    print(f"{'='*100}")
    print(f"{'sleeve':>7}{'STRONG':>8}{'  ':2}"
          f"{'Full ret':>9}{'Full Shp':>9}{'Full MDD':>9}"
          f"{'Hold ret':>9}{'Hold Shp':>9}{'Hold MDD':>9}"
          f"{'WF+':>6}{'gate':>7}")
    for sleeve, strong in cells:
        daily = run_blend(close, vix, cash_yield, fd, sleeve, strong)
        is_slice = daily[daily.index < FULL_OOS_START]
        full = perf(daily[daily.index >= FULL_OOS_START])
        hold = perf(daily[daily.index >= HOLDOUT_START])
        is_sharpe = perf(is_slice)["sharpe"]
        gate = gate_check(full, hold, daily[daily.index >= FULL_OOS_START], is_sharpe)
        tag = "  <shipped" if (sleeve, strong) == SHIPPED else ""
        print(f"{sleeve:>7.2f}{strong:>8.2f}{'  ':2}"
              f"{full['ret']*100:>+8.0f}%{full['sharpe']:>9.2f}{full['mdd']*100:>+8.0f}%"
              f"{hold['ret']*100:>+8.0f}%{hold['sharpe']:>9.2f}{hold['mdd']*100:>+8.0f}%"
              f"{gate['wfFrac']*100:>5.0f}%{'PASS' if gate['pass'] else 'FAIL':>7}{tag}")
        rows.append({
            "sleeve_weight": sleeve, "strong_spxl": strong,
            "full_ret": round(full["ret"], 4), "full_sharpe": round(full["sharpe"], 3),
            "full_mdd": round(full["mdd"], 4),
            "hold_ret": round(hold["ret"], 4), "hold_sharpe": round(hold["sharpe"], 3),
            "hold_mdd": round(hold["mdd"], 4),
            "wf_pos_frac": round(gate["wfFrac"], 3), "wf_folds": gate["wfFolds"],
            "oos_is_ratio": round(gate["oosIs"], 3) if np.isfinite(gate["oosIs"]) else None,
            "gate_pass": gate["pass"],
        })
    trend_core.STRONG_SPXL = shipped_strong

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(rows)} cells)")
    if synthetic:
        print("NOTE: synthetic self-test — numbers are meaningless; this only "
              "proves the production funcs replay and the gate logic runs.")
    else:
        print("DECISION RULE: adopt the LARGEST sleeve whose Full AND Holdout "
              "Sharpe >= shipped, MDD not worse, and gate=PASS. Else keep 0.15.")


if __name__ == "__main__":
    main()
