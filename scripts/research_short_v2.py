"""Short alpha challenge v2 (2026-06-12) — pre-registered research.

Prior art: 9 short expressions ALL FAILED (2026-05-20 campaign — inverse-ETF
proba, reverse-proba, bear-regime gates, market-neutral pairs, single-feature
down-IC, fear-extreme shorts, cross-sectional stock shorts, trend-short+stop,
put buying). Documented failure mode: shorting WEAKNESS gets run over by the
violent mean reversion that follows fear extremes ("약하다 ≠ 내린다").

This campaign tests ONLY hypotheses that were not among the 9 and that target
that failure mode directly. Real inverse-ETF prices (SH/PSQ) are used, so
daily-reset decay and fund costs are included — no -1x return idealization.

H1  Sell-the-rip: in a confirmed bear (SPY close < 200SMA), enter SH AFTER a
    counter-trend bounce (r-day SPY return >= b%) — i.e. short into strength,
    so the mean-reversion bounce happens BEFORE entry, not after. Fixed h-day
    hold, non-overlapping. Grid r∈{2,3} × b∈{2,3,4}% × h∈{3,5,10}.
    Secondary: same grid on QQQ/PSQ.

H2  Inverse sleeve in the cash regime: while trend-OFF the live system parks
    100% in BIL/SGOV. Does replacing w∈{10,20,30}% of that parking with SH
    beat pure T-bills on the trend-OFF segment? (Portfolio question, not
    standalone alpha.)

H3  Bottom-RSI sectors (informational only): the validated RSI-14 key picks
    TOP sectors (IC +0.035). Are BOTTOM-ranked sectors actually NEGATIVE
    forward — i.e. shortable — or merely less positive? No liquid -1x sector
    ETF exists in the universe, so this is knowledge, not a trade.

PRE-REGISTERED DECISION RULES (fixed before the first run; all cells printed):
  * cost  = config.TOTAL_COST_ROUNDTRIP per round trip, charged on entry.
  * PASS needs ALL of: Sharpe > config.TIER1_SHARPE_MIN on 2021+ window,
    2024+ window total return not negative, |MDD| < config.TIER1_MDD_MAX,
    n_trades >= config.TIER1_MIN_TRADING_DAYS (120), and a majority of
    calendar years positive.
  * n_trades < 120 with positive stats → "insufficient sample" (표본 보류),
    NOT a pass (feedback-validation-discipline).
  * An isolated positive cell whose grid neighbors are negative = overfit
    artifact → FAIL regardless of its own stats.
  * Signals use close-T information only; entry at close T (project
    convention, same as the 2026-05-20 campaign and the blend backtests).

Run:  python scripts/research_short_v2.py
Outputs: console tables + data/logs/short_v2_results.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

COST = config.TOTAL_COST_ROUNDTRIP
TRADING_DAYS = 252
OUT = Path("data/logs/short_v2_results.csv")

FULL_OOS = "2021-01-01"   # same convention as the blend backtests
HOLDOUT = "2024-01-01"


def load_closes() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet("data/raw/etf_ohlcv.parquet")
    close = df["Close"].copy()
    close.index = pd.to_datetime(close.index).normalize()
    y = pd.read_parquet("data/raw/macro/yield_2y.parquet").iloc[:, 0]
    y.index = pd.to_datetime(y.index).normalize()
    cash_daily = (y.reindex(close.index).ffill() / 100.0 / TRADING_DAYS).fillna(0.0)
    return close, cash_daily


def ann_sharpe(rets: pd.Series) -> float:
    r = rets.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    eq = equity.values
    if len(eq) < 2:
        return 0.0
    return float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())


def yearly_positive_majority(daily: pd.Series) -> tuple[int, int]:
    by_year = (1 + daily).groupby(daily.index.year).prod() - 1
    by_year = by_year[by_year.abs() > 1e-12]  # years with actual exposure
    return int((by_year > 0).sum()), int(len(by_year))


# ── H1: sell-the-rip ──────────────────────────────────────────────────────────

def h1_cell(close: pd.DataFrame, cash: pd.Series, idx_tk: str, inv_tk: str,
            r: int, b: float, h: int) -> dict:
    """One grid cell: bear regime + bounce trigger → hold inverse ETF h days."""
    px = close[idx_tk].dropna()
    inv = close[inv_tk].dropna()
    common = px.index.intersection(inv.index)
    px, inv = px.loc[common], inv.loc[common]
    sma200 = px.rolling(200).mean()
    bounce = px.pct_change(r)
    inv_ret = inv.pct_change()

    bear = px < sma200
    trigger = bear & (bounce >= b) & sma200.notna()

    daily = pd.Series(0.0, index=common)  # strategy return when invested, else cash
    in_pos_until = -1
    trades = []
    dates = list(common)
    for i, dt in enumerate(dates):
        if i <= in_pos_until:
            continue
        if not bool(trigger.loc[dt]):
            continue
        end = min(i + h, len(dates) - 1)
        if end <= i:
            continue
        # invested daily returns T+1..T+h on the REAL inverse ETF
        seg = inv_ret.iloc[i + 1:end + 1]
        daily.iloc[i + 1:end + 1] = seg.values
        # cost charged on the first invested day
        daily.iloc[i + 1] -= COST
        trades.append(float((1 + seg).prod() - 1 - COST))
        in_pos_until = end

    invested = daily != 0.0
    flat_cash = cash.reindex(common).fillna(0.0).where(~invested, 0.0)
    full_daily = daily + flat_cash
    eq = (1 + full_daily).cumprod()

    tr = pd.Series(trades)
    pos_years, n_years = yearly_positive_majority(daily)

    d21 = daily[daily.index >= FULL_OOS]
    d24 = daily[daily.index >= HOLDOUT]
    out = {
        "idx": idx_tk, "inv": inv_tk, "r": r, "b": b, "h": h,
        "n": len(tr),
        "wr": round(float((tr > 0).mean()), 3) if len(tr) else None,
        "avg": round(float(tr.mean()), 4) if len(tr) else None,
        "total": round(float(eq.iloc[-1] - 1), 4),
        "sharpe_full": round(ann_sharpe(daily[daily != 0]), 2),
        "sharpe_2021": round(ann_sharpe(d21[d21 != 0]), 2),
        "ret_2021": round(float((1 + d21).prod() - 1), 4),
        "ret_2024": round(float((1 + d24).prod() - 1), 4),
        "mdd": round(max_drawdown(eq), 4),
        "yrs_pos": f"{pos_years}/{n_years}",
    }
    # pre-registered verdict
    if out["n"] < config.TIER1_MIN_TRADING_DAYS:
        out["verdict"] = ("insufficient_sample(+)" if (out["ret_2021"] > 0 and out["ret_2024"] >= 0)
                          else "FAIL")
    else:
        ok = (out["sharpe_2021"] > config.TIER1_SHARPE_MIN
              and out["ret_2024"] >= 0
              and abs(out["mdd"]) < config.TIER1_MDD_MAX
              and pos_years * 2 > n_years)
        out["verdict"] = "PASS" if ok else "FAIL"
    return out


# ── H2: inverse sleeve inside the cash regime ─────────────────────────────────

def h2_cell(close: pd.DataFrame, cash: pd.Series, w: float) -> dict:
    """Trend-OFF (SPY<200SMA) parking: w in SH + (1-w) cash vs 100% cash."""
    spy = close["SPY"].dropna()
    sh_ret = close["SH"].pct_change()
    off = (spy < spy.rolling(200).mean()) & spy.rolling(200).mean().notna()
    off = off.reindex(close.index).fillna(False)
    c = cash.reindex(close.index).fillna(0.0)
    sleeve = (w * sh_ret.fillna(0.0) + (1 - w) * c).where(off, 0.0)
    # cost: w bought on each OFF entry, sold on each OFF exit
    flips_in = off & ~off.shift(1, fill_value=False)
    flips_out = ~off & off.shift(1, fill_value=False)
    sleeve[flips_in] -= w * COST / 2
    sleeve[flips_out.shift(-1, fill_value=False)] -= w * COST / 2
    base = c.where(off, 0.0)

    def seg(s, start=None):
        x = s[s.index >= start] if start else s
        x = x[off[off.index >= start].reindex(x.index).fillna(False)] if start else x[off]
        return x

    rows = {}
    for label, start in (("2015+", None), ("2021+", FULL_OOS), ("2024+", HOLDOUT)):
        sl, ba = seg(sleeve, start), seg(base, start)
        rows[label] = {
            "days": len(sl),
            "sleeve_total": round(float((1 + sl).prod() - 1), 4),
            "cash_total": round(float((1 + ba).prod() - 1), 4),
            "sleeve_sharpe": round(ann_sharpe(sl), 2),
            "sleeve_mdd": round(max_drawdown((1 + sl).cumprod()), 4),
        }
    full = rows["2015+"]
    better = (full["sleeve_total"] > full["cash_total"]
              and rows["2021+"]["sleeve_total"] > rows["2021+"]["cash_total"])
    return {"w": w, "windows": rows,
            "verdict": "beats_cash" if better else "FAIL_vs_cash"}


# ── H3: bottom-RSI sectors forward returns (informational) ───────────────────

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI"]


def rsi14(s: pd.Series, window: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def h3_rank_study(close: pd.DataFrame) -> pd.DataFrame:
    px = close[SECTORS].dropna()
    r = {t: rsi14(px[t]) for t in SECTORS}
    fridays = [d for d in px.index if d.weekday() == 4]
    recs = []
    for k, d in enumerate(fridays[:-1]):
        nxt = fridays[k + 1]
        if (nxt - d).days > 9:
            continue
        ranked = sorted(SECTORS, key=lambda t: r[t].loc[d] if pd.notna(r[t].loc[d]) else -1,
                        reverse=True)
        for rank, t in enumerate(ranked, 1):
            fwd = float(px[t].loc[nxt] / px[t].loc[d] - 1)
            below200 = bool(px[t].loc[d] < px[t].rolling(200).mean().loc[d]) \
                if k >= 40 else None
            recs.append({"date": d, "ticker": t, "rank": rank, "fwd5": fwd,
                         "below200": below200})
    df = pd.DataFrame(recs)
    g = df.groupby("rank")["fwd5"]
    summary = pd.DataFrame({
        "mean_fwd5": g.mean().round(5),
        "tstat": (g.mean() / (g.std() / np.sqrt(g.count()))).round(2),
        "n": g.count(),
        "pct_neg": (df.groupby("rank")["fwd5"].apply(lambda s: (s < 0).mean())).round(3),
    })
    # bottom-2 below 200SMA — the most "shortable" candidate slice
    b2 = df[(df["rank"] >= 5) & (df["below200"] == True)]["fwd5"]  # noqa: E712
    extra = {"mean_fwd5": round(b2.mean(), 5) if len(b2) else None,
             "tstat": round(b2.mean() / (b2.std() / np.sqrt(len(b2))), 2) if len(b2) > 2 else None,
             "n": len(b2), "pct_neg": round((b2 < 0).mean(), 3) if len(b2) else None}
    summary.loc["bot2_below200"] = extra
    return summary


def main() -> None:
    close, cash = load_closes()
    print(f"data: {close.index[0].date()} → {close.index[-1].date()}  cost={COST:.4f}/rt")
    rows = []

    print("\n══ H1 sell-the-rip (bear + bounce → real inverse ETF, fixed hold) ══")
    for idx_tk, inv_tk in (("SPY", "SH"), ("QQQ", "PSQ")):
        for r in (2, 3):
            for b in (0.02, 0.03, 0.04):
                for h in (3, 5, 10):
                    c = h1_cell(close, cash, idx_tk, inv_tk, r, b, h)
                    rows.append({"section": "H1", **c})
    h1 = pd.DataFrame([r for r in rows if r["section"] == "H1"])
    cols = ["idx", "r", "b", "h", "n", "wr", "avg", "total",
            "sharpe_2021", "ret_2021", "ret_2024", "mdd", "yrs_pos", "verdict"]
    print(h1[cols].to_string(index=False))

    print("\n══ H2 inverse sleeve in trend-OFF cash parking (vs pure T-bill) ══")
    for w in (0.10, 0.20, 0.30):
        c = h2_cell(close, cash, w)
        rows.append({"section": "H2", "w": w, "verdict": c["verdict"],
                     **{f"{k}_{m}": v for k, win in c["windows"].items()
                        for m, v in win.items()}})
        print(f"w={w:.0%}  verdict={c['verdict']}")
        for label, win in c["windows"].items():
            print(f"   {label}: OFF {win['days']}d · sleeve {win['sleeve_total']:+.2%} "
                  f"(Sharpe {win['sleeve_sharpe']:.2f}, MDD {win['sleeve_mdd']:.2%}) "
                  f"vs cash {win['cash_total']:+.2%}")

    print("\n══ H3 sector RSI rank → fwd 5d (informational; no -1x sector ETF) ══")
    h3 = h3_rank_study(close)
    print(h3.to_string())
    for rank, rec in h3.iterrows():
        rows.append({"section": "H3", "rank": str(rank), **rec.to_dict()})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nall cells written → {OUT}")


if __name__ == "__main__":
    main()
