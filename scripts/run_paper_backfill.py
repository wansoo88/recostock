#!/usr/bin/env python3
"""Phase 4: Paper trading historical backfill (carry-over cost model).

Replays Phase 3 walk-forward OOS signals with CORRECT cost accounting:
  - Cost charged ONCE at entry and ONCE at exit per position
  - Carry-over weeks (same ticker, signal still ON) incur NO extra cost
  - Matches the Phase 3 backtest _portfolio_pnl cost model exactly

Output: data/paper/trades.parquet

Usage:
    python scripts/run_paper_backfill.py
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
from models.train_lgbm import apply_ema_weekly, build_feature_matrix, walk_forward_lgbm, build_target
from models.inference import EMA_SPAN, MIN_ACTIVE_FRIDAYS, HOLD_DAYS, ROLLING_WINDOW
from paper.tracker import TRADES_PATH, save_trades

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BACKFILL_TAG = "backfill"


def _rolling_stats_at_friday(
    smooth_raw: pd.DataFrame,
    fwd_ret_5d: pd.DataFrame,
    friday: pd.Timestamp,
    lookback_weeks: int,
) -> dict[str, dict]:
    """Per-ticker rolling stats using only data strictly before `friday`."""
    all_fridays = sorted(d for d in smooth_raw.index if pd.Timestamp(d).dayofweek == 4 and d < friday)
    window = all_fridays[-lookback_weeks:] if len(all_fridays) >= lookback_weeks else all_fridays
    if not window:
        return {}

    stats = {}
    for ticker in smooth_raw.columns:
        if ticker not in fwd_ret_5d.columns:
            continue
        sig = smooth_raw[ticker].reindex(window).dropna()
        ret = fwd_ret_5d[ticker].reindex(window).dropna()
        common = sig.index.intersection(ret.index)
        active = common[sig.reindex(common) >= config.SIGNAL_THRESHOLD]
        if len(active) < MIN_ACTIVE_FRIDAYS:
            continue
        r = ret.loc[active]
        wins = r[r > 0]
        losses = r[r <= 0]
        winrate = len(wins) / len(r)
        avg_win = max(float(wins.mean()) if len(wins) > 0 else 0.008, 0.004)
        avg_loss = max(float(abs(losses.mean())) if len(losses) > 0 else 0.006, 0.003)
        payoff = avg_win / avg_loss
        expectancy = winrate * avg_win - (1 - winrate) * avg_loss - config.TOTAL_COST_ROUNDTRIP
        if winrate > 0 and payoff >= config.MIN_PAYOFF and expectancy > 0:
            stats[ticker] = {
                "winrate": round(winrate, 4),
                "avg_win": round(avg_win, 5),
                "avg_loss": round(avg_loss, 5),
                "payoff": round(payoff, 3),
                "expectancy": round(expectancy, 5),
                "sample_n": len(r),
            }
    return stats


def main() -> None:
    raw_path = Path("data/raw/etf_ohlcv.parquet")
    vix_path = Path("data/raw/vix.parquet")
    if not raw_path.exists():
        log.error("Data not found. Run: python -m data.collector")
        sys.exit(1)

    log.info("Loading data...")
    ohlcv = pd.read_parquet(raw_path)
    close_df = ohlcv["Close"] if isinstance(ohlcv.columns, pd.MultiIndex) else ohlcv
    vix_df = pd.read_parquet(vix_path) if vix_path.exists() else None

    log.info("Walk-forward training (OOS probabilities only)...")
    X = build_feature_matrix(close_df, vix_df)
    y = build_target(close_df, horizon=1)
    common = X.index.intersection(y.index)
    X, y = X.loc[common], y.loc[common]
    proba, _ = walk_forward_lgbm(X, y, n_splits=5, save_dir=None)

    smooth = apply_ema_weekly(proba, ema_span=EMA_SPAN, threshold=config.SIGNAL_THRESHOLD)
    smooth_raw = proba.unstack(level=1).ewm(span=EMA_SPAN).mean()

    fwd_ret_5d = close_df.pct_change(HOLD_DAYS).shift(-HOLD_DAYS)
    all_dates = sorted(smooth.index)
    oos_start = proba.index.get_level_values("date").min()
    all_fridays = [d for d in all_dates if pd.Timestamp(d).dayofweek == 4 and d >= oos_start]
    lookback_weeks = ROLLING_WINDOW // 5
    eval_fridays = all_fridays[lookback_weeks:]

    log.info("OOS Fridays: %d total, %d for evaluation", len(all_fridays), len(eval_fridays))

    # ── Carry-over position tracking ──────────────────────────────────────────
    # current_positions: ticker -> {open_date, entry_price, stats}
    current_positions: dict[str, dict] = {}
    closed_trades: list[dict] = []

    for friday in eval_fridays:
        ts = pd.Timestamp(friday)
        stats = _rolling_stats_at_friday(smooth_raw, fwd_ret_5d, ts,
                                          lookback_weeks=lookback_weeks)

        # Determine signal=1 tickers that pass the gate
        signal_tickers: set[str] = set()
        for ticker in smooth.columns:
            if float(smooth.at[friday, ticker]) < 0.5:
                continue
            if ticker not in stats:
                continue
            signal_tickers.add(ticker)

        close_prices_today = close_df.loc[friday] if friday in close_df.index else pd.Series()

        # ── Close positions where signal turned OFF ───────────────────────────
        for ticker in list(current_positions.keys()):
            if ticker not in signal_tickers:
                pos = current_positions.pop(ticker)
                exit_price = close_prices_today.get(ticker)
                if exit_price is None or pd.isna(exit_price) or exit_price <= 0:
                    continue
                entry = pos["entry_price"]
                gross = (float(exit_price) - float(entry)) / float(entry)
                net = gross - config.TOTAL_COST_ROUNDTRIP  # single exit cost only
                closed_trades.append({
                    "open_date": pos["open_date"],
                    "ticker": ticker,
                    "entry_price": round(float(entry), 4),
                    "direction": "long",
                    "ema_proba": pos["ema_proba"],
                    "winrate": pos["winrate"],
                    "payoff": pos["payoff"],
                    "expectancy": pos["expectancy"],
                    "sample_n": pos["sample_n"],
                    "close_date": ts.date().isoformat(),
                    "exit_price": round(float(exit_price), 4),
                    "pnl_pct": round(net, 6),
                    "source": BACKFILL_TAG,
                })

        # ── Open NEW positions (not carry-overs) ─────────────────────────────
        for ticker in signal_tickers:
            if ticker in current_positions:
                continue  # carry-over: already in position, no cost
            entry_price = close_prices_today.get(ticker)
            if entry_price is None or pd.isna(entry_price) or entry_price <= 0:
                continue
            st = stats[ticker]
            ema_p = float(smooth_raw.at[friday, ticker]) if friday in smooth_raw.index else 0.55
            current_positions[ticker] = {
                "open_date": ts.date().isoformat(),
                "entry_price": float(entry_price),
                "ema_proba": round(ema_p, 4),
                "winrate": st["winrate"],
                "payoff": st["payoff"],
                "expectancy": st["expectancy"],
                "sample_n": st["sample_n"],
            }

    # ── Close remaining open positions at last available price ────────────────
    last_friday = eval_fridays[-1] if eval_fridays else None
    if last_friday and current_positions:
        last_close = close_df.loc[last_friday] if last_friday in close_df.index else pd.Series()
        for ticker, pos in current_positions.items():
            exit_price = last_close.get(ticker)
            if exit_price is None or pd.isna(exit_price) or exit_price <= 0:
                continue
            entry = pos["entry_price"]
            gross = (float(exit_price) - float(entry)) / float(entry)
            net = gross - config.TOTAL_COST_ROUNDTRIP
            closed_trades.append({
                "open_date": pos["open_date"],
                "ticker": ticker,
                "entry_price": round(float(entry), 4),
                "direction": "long",
                "ema_proba": pos["ema_proba"],
                "winrate": pos["winrate"],
                "payoff": pos["payoff"],
                "expectancy": pos["expectancy"],
                "sample_n": pos["sample_n"],
                "close_date": pd.Timestamp(last_friday).date().isoformat(),
                "exit_price": round(float(exit_price), 4),
                "pnl_pct": round(net, 6),
                "source": BACKFILL_TAG,
            })

    trades_df = pd.DataFrame(closed_trades)

    print("\n" + "=" * 65)
    print("  PAPER TRADING BACKFILL — CARRY-OVER MODEL")
    print("=" * 65)
    print(f"  Eval Fridays:     {len(eval_fridays):>5}")
    print(f"  Closed trades:    {len(trades_df):>5}")
    if not trades_df.empty:
        pnl = trades_df["pnl_pct"].astype(float)
        # Group by close_date for portfolio Sharpe
        weekly = trades_df.groupby("close_date")["pnl_pct"].mean().astype(float)
        weekly = weekly.sort_index()
        if len(weekly) >= 2:
            sharpe = float(weekly.mean() / weekly.std() * np.sqrt(52))
        else:
            sharpe = 0.0
        equity = (1 + weekly).cumprod()
        peak = equity.cummax()
        mdd = float(((equity - peak) / peak).min())
        wins = pnl[pnl > 0]
        wr = len(wins) / len(pnl)
        print(f"  Unique weeks:     {len(weekly):>5}")
        print(f"  Winrate:          {wr:>6.1%}")
        print(f"  Net Sharpe:       {sharpe:>6.3f}")
        print(f"  MDD:              {mdd:>6.1%}")
        print(f"  Total return:     {float(equity.iloc[-1]-1):>+.1%}")

        print("\n  By ticker:")
        by_ticker = trades_df.groupby("ticker")["pnl_pct"].agg(["count", "mean", "sum"])
        by_ticker.columns = ["n", "avg_pnl", "total_pnl"]
        by_ticker = by_ticker.sort_values("total_pnl", ascending=False)
        for tkr, row in by_ticker.iterrows():
            print(f"    {tkr:<6}  n={int(row['n']):>3}  avg={row['avg_pnl']:>+.2%}  total={row['total_pnl']:>+.1%}")

    print("=" * 65)

    # Merge with existing live trades
    TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if TRADES_PATH.exists():
        existing = pd.read_parquet(TRADES_PATH)
        live = existing[existing.get("source", pd.Series("live", index=existing.index)) != BACKFILL_TAG] \
               if "source" in existing.columns else existing
        combined = pd.concat([trades_df, live], ignore_index=True) if not live.empty else trades_df
    else:
        combined = trades_df

    save_trades(combined)
    log.info("Saved %d total trades to %s", len(combined), TRADES_PATH)


if __name__ == "__main__":
    main()
