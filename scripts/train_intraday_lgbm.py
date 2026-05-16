#!/usr/bin/env python3
"""Intraday LightGBM — Phase 3 pattern adapted to 5-min bars.

Pipeline:
  1. Fetch 60d of 5min OHLCV (existing fetch_history util).
  2. Build per-bar feature matrix (13 KEEP factors + minutes_from_open + VIX).
  3. Label: forward 12-bar (=60min) log-return > 0 -> 1 else 0, masked across day boundaries.
  4. Day-based walk-forward (4 folds), pooled LightGBM classifier.
  5. OOS probabilities -> directional signals (prob > th_long => LONG, prob < th_short => SHORT).
  6. Backtest OOS signals through the same simulate_day engine, report net EV.

Honesty note: even at OOS AUC ~0.55, beating 0.25% roundtrip cost needs gross EV/trade
~+0.30%. The IC ceiling was +0.03%, so LightGBM must find non-linear edge >> linear baseline.

Usage:
    python scripts/train_intraday_lgbm.py [--days 60] [--horizon 12]
                                          [--th-long 0.55] [--th-short 0.45]
"""
from __future__ import annotations

import argparse
import logging
import math
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.intraday import CORE_TICKERS, INVERSE_PAIR, SECTOR_TICKERS
from features.intraday_factors import compute_intraday_features
from scripts.run_intraday_backtest import (
    fetch_history,
    fetch_daily_vix,
    summarize,
    Trade,
    tp_sl,
    ATR_PERIOD,
)

log = logging.getLogger("intra-lgbm")

# Features the model sees. minutes_from_open captures time-of-day pattern.
FEATURE_COLS = [
    "rsi14", "stochrsi_k",
    "ema_spread_pct", "price_vs_vwap_pct", "vwap_dev_sd",
    "adx14",
    "obv_slope",
    "vol_ratio",
    "atr_pct",
    "minutes_from_open",
    "vix_prior",
    "ticker_code",
]

CATEGORICAL = ["ticker_code"]

TICKER_CODE = {t: i for i, t in enumerate(CORE_TICKERS + SECTOR_TICKERS)}

LGBM_PARAMS = dict(
    n_estimators=400,
    max_depth=5,
    num_leaves=24,
    learning_rate=0.03,
    subsample=0.7,
    subsample_freq=1,
    colsample_bytree=0.7,
    min_child_samples=200,
    reg_alpha=0.2,
    reg_lambda=1.0,
    is_unbalance=True,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)


# ── Feature/label construction ────────────────────────────────────────────────

def build_feature_table(
    ohlcv: dict[str, pd.DataFrame],
    vix: pd.Series,
    horizon: int,
) -> pd.DataFrame:
    """Returns long-form DataFrame: one row per (ticker, bar timestamp).

    Includes features + forward-return label (binary). Cross-day forward returns are NaN.
    """
    parts = []
    for ticker, df in ohlcv.items():
        for date_, day_df in df.groupby(df.index.date):
            if len(day_df) < 25:
                continue
            feat = compute_intraday_features(day_df).copy()
            feat["ema_spread_pct"] = (feat["ema5"] - feat["ema20"]) / feat["ema20"]
            feat["price_vs_vwap_pct"] = (feat["Close"] - feat["vwap"]) / feat["vwap"]

            # ATR(14) as percent of price
            hi, lo, cl = feat["High"], feat["Low"], feat["Close"]
            prev_cl = cl.shift(1)
            tr = pd.concat([hi - lo, (hi - prev_cl).abs(), (lo - prev_cl).abs()], axis=1).max(axis=1)
            atr = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()
            feat["atr_pct"] = atr / feat["Close"]

            # Forward log-return (within day only)
            feat["fwd_ret"] = np.log(feat["Close"].shift(-horizon) / feat["Close"])

            prior = vix.index[vix.index < date_]
            feat["vix_prior"] = float(vix.loc[prior[-1]]) if len(prior) else np.nan

            feat["ticker"] = ticker
            feat["ticker_code"] = TICKER_CODE[ticker]
            feat["session_date"] = pd.Timestamp(date_)

            parts.append(feat)

    df_all = pd.concat(parts, axis=0).sort_index()
    df_all["label"] = (df_all["fwd_ret"] > 0).astype(np.int8)
    df_all = df_all.dropna(subset=FEATURE_COLS + ["fwd_ret"])
    # yfinance DatetimeIndex may be named "Datetime" or None; pin it to "ts"
    df_all.index.name = "ts"
    df_all = df_all.reset_index()
    df_all["row_id"] = np.arange(len(df_all), dtype=np.int64)
    return df_all


# ── Walk-forward training ─────────────────────────────────────────────────────

def walk_forward(df: pd.DataFrame, n_splits: int = 4) -> tuple[pd.Series, pd.DataFrame, "object"]:
    """Day-based expanding-window walk-forward.

    Returns: (oos_prob Series indexed like df, fold_metrics DataFrame, final_model)
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    sessions = sorted(df["session_date"].unique())
    n = len(sessions)
    min_train = max(20, n // 3)   # at least 20 days warmup
    oos_size = max(5, (n - min_train) // n_splits)

    log.info("WF: %d sessions, %d folds, min_train=%d, oos_size=%d", n, n_splits, min_train, oos_size)

    fold_rows = []
    parts: list[pd.Series] = []
    final_model = None

    for k in range(n_splits):
        train_end = min_train + k * oos_size
        if train_end >= n:
            break
        test_end = min(train_end + oos_size, n)
        train_days = set(sessions[:train_end])
        test_days = set(sessions[train_end:test_end])

        tr = df[df["session_date"].isin(train_days)]
        te = df[df["session_date"].isin(test_days)]
        if len(tr) < 1000 or len(te) < 100:
            log.warning("fold %d skipped (tr=%d, te=%d)", k + 1, len(tr), len(te))
            continue

        X_tr, y_tr = tr[FEATURE_COLS], tr["label"]
        X_te, y_te = te[FEATURE_COLS], te["label"]

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            feature_name=list(FEATURE_COLS),
            categorical_feature=CATEGORICAL,
        )
        p_tr = model.predict_proba(X_tr)[:, 1]
        p_te = model.predict_proba(X_te)[:, 1]
        is_auc = roc_auc_score(y_tr, p_tr)
        oos_auc = roc_auc_score(y_te, p_te)
        log.info("fold %d  IS AUC=%.4f  OOS AUC=%.4f  ntr=%d nte=%d",
                 k + 1, is_auc, oos_auc, len(tr), len(te))

        parts.append(pd.Series(p_te, index=te["row_id"].values, name="proba"))
        fold_rows.append({
            "fold": k + 1,
            "train_end": str(sessions[train_end - 1].date()),
            "test_end": str(sessions[test_end - 1].date()),
            "n_train": len(tr),
            "n_test": len(te),
            "is_auc": round(is_auc, 4),
            "oos_auc": round(oos_auc, 4),
        })
        final_model = model

    oos_prob = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)
    return oos_prob, pd.DataFrame(fold_rows), final_model


# ── Backtest with ML signals (probability-driven) ─────────────────────────────

def simulate_ml(
    df: pd.DataFrame,
    oos_prob: pd.Series,
    th_long: float,
    th_short: float,
    tp_pct: float,
    sl_pct: float,
    time_stop_bars: int,
    horizon: int,
) -> list[Trade]:
    """Walk OOS bars with model probability driving entry direction.

    `df` is expected to have a unique `row_id` column; `oos_prob` is indexed by row_id.
    """
    df = df.copy()
    df["proba_raw"] = df["row_id"].map(oos_prob)
    df = df.dropna(subset=["proba_raw"])
    # EMA-smooth probability within each ticker+session to reduce bar-to-bar flip noise
    df = df.sort_values(["ticker", "session_date", "ts"]).reset_index(drop=True)
    df["proba"] = (
        df.groupby(["ticker", "session_date"])["proba_raw"]
          .transform(lambda s: s.ewm(span=3, adjust=False).mean())
    )

    trades: list[Trade] = []

    for (ticker, session), group in df.groupby(["ticker", "session_date"]):
        allow_short = ticker in CORE_TICKERS
        bars = group.sort_values("ts").reset_index(drop=True)
        n = len(bars)
        if n < 8:
            continue

        position = 0
        entry_idx = -1
        entry_px = 0.0
        tp_px = 0.0
        sl_px = 0.0
        time_stop_idx = n - 1 - time_stop_bars if time_stop_bars > 0 else None

        for i in range(n - 1):
            last = bars.iloc[i]
            nxt = bars.iloc[i + 1]

            # Time-based force close
            if position != 0 and time_stop_idx is not None and i >= time_stop_idx:
                exit_px = float(nxt["Open"])
                gross = position * (exit_px - entry_px) / entry_px
                trades.append(_make_trade(ticker, position, bars, entry_idx, entry_px,
                                          nxt, exit_px, "TIME", i + 1 - entry_idx, gross))
                position = 0
                continue

            # Manage TP/SL with next bar's high/low
            if position != 0:
                hi, lo = float(nxt["High"]), float(nxt["Low"])
                exit_px, reason = 0.0, ""
                if position == 1:
                    if lo <= sl_px:
                        exit_px, reason = sl_px, "SL"
                    elif hi >= tp_px:
                        exit_px, reason = tp_px, "TP"
                else:
                    if hi >= sl_px:
                        exit_px, reason = sl_px, "SL"
                    elif lo <= tp_px:
                        exit_px, reason = tp_px, "TP"
                if reason:
                    gross = position * (exit_px - entry_px) / entry_px
                    trades.append(_make_trade(ticker, position, bars, entry_idx, entry_px,
                                              nxt, exit_px, reason, i + 1 - entry_idx, gross))
                    position = 0
                    continue

            # Entry decision from probability
            p = float(last["proba"])
            if position == 0:
                new_dir = 0
                if p >= th_long:
                    new_dir = 1
                elif p <= th_short and allow_short:
                    new_dir = -1
                if new_dir != 0:
                    entry_idx = i + 1
                    entry_px = float(nxt["Open"])
                    tp_gap = entry_px * tp_pct
                    sl_gap = entry_px * sl_pct
                    if new_dir == 1:
                        tp_px, sl_px = entry_px + tp_gap, entry_px - sl_gap
                    else:
                        tp_px, sl_px = entry_px - tp_gap, entry_px + sl_gap
                    position = new_dir
            else:
                # Flip on opposite high-confidence signal
                opp = (position == 1 and p <= th_short) or (position == -1 and p >= th_long)
                if opp:
                    exit_px = float(nxt["Open"])
                    gross = position * (exit_px - entry_px) / entry_px
                    trades.append(_make_trade(ticker, position, bars, entry_idx, entry_px,
                                              nxt, exit_px, "FLIP", i + 1 - entry_idx, gross))
                    position = 0

        # EOD close
        if position != 0:
            last_bar = bars.iloc[-1]
            exit_px = float(last_bar["Close"])
            gross = position * (exit_px - entry_px) / entry_px
            trades.append(_make_trade(ticker, position, bars, entry_idx, entry_px,
                                      last_bar, exit_px, "EOD", n - entry_idx, gross))

    return trades


def _make_trade(ticker, direction, bars, entry_idx, entry_px,
                exit_row, exit_px, reason, hold, gross):
    return Trade(
        ticker=ticker,
        action_ticker=(INVERSE_PAIR.get(ticker, ticker) if direction == -1 else ticker),
        direction=direction,
        entry_date=str(bars.iloc[entry_idx]["ts"].date()),
        entry_time=str(bars.iloc[entry_idx]["ts"].time()),
        entry_price=entry_px,
        exit_time=str(exit_row["ts"].time() if hasattr(exit_row, "ts") else exit_row["ts"].time()),
        exit_price=exit_px,
        exit_reason=reason,
        hold_bars=hold,
        gross_pnl=gross,
        net_pnl=gross - config.TOTAL_COST_ROUNDTRIP,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=12, help="forward-return horizon in 5min bars")
    ap.add_argument("--n-splits", type=int, default=4)
    ap.add_argument("--th-long", type=float, default=0.55)
    ap.add_argument("--th-short", type=float, default=0.45)
    ap.add_argument("--tp-pct", type=float, default=0.005)
    ap.add_argument("--sl-pct", type=float, default=0.005)
    ap.add_argument("--time-stop-bars", type=int, default=6)
    ap.add_argument("--save-dir", type=str, default="models/weights")
    ap.add_argument("--out-trades", type=str, default="data/intraday_lgbm_trades.csv")
    ap.add_argument("--out-summary", type=str, default="data/intraday_lgbm_summary.txt")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("intraday LightGBM start: horizon=%d, th_long=%.2f, th_short=%.2f, tp=%.3f, sl=%.3f",
             args.horizon, args.th_long, args.th_short, args.tp_pct, args.sl_pct)

    tickers = CORE_TICKERS + SECTOR_TICKERS
    log.info("fetching %dd of 5m bars", args.days)
    ohlcv = fetch_history(tickers, args.days, interval="5m")
    if not ohlcv:
        log.error("no data")
        sys.exit(1)
    start = min(df.index[0] for df in ohlcv.values())
    end = max(df.index[-1] for df in ohlcv.values())
    vix = fetch_daily_vix(start, end + pd.Timedelta(days=1))

    log.info("building feature table")
    df = build_feature_table(ohlcv, vix, horizon=args.horizon)
    log.info("feature table: %d rows, %d sessions, %d tickers",
             len(df), df["session_date"].nunique(), df["ticker"].nunique())

    if len(df) < 5000:
        log.error("too few samples: %d", len(df))
        sys.exit(1)

    oos_prob, folds, model = walk_forward(df, n_splits=args.n_splits)
    log.info("OOS predictions: %d", len(oos_prob))
    log.info("\n%s", folds.to_string(index=False))

    # Persist model + feature importance
    if model is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "intraday_lgbm.pkl", "wb") as f:
            pickle.dump(model, f)
        pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        }).sort_values("importance_gain", ascending=False).to_csv(
            save_dir / "intraday_lgbm_feature_importance.csv", index=False
        )

    # Backtest using OOS probs
    log.info("backtesting with thresholds th_long=%.2f / th_short=%.2f", args.th_long, args.th_short)
    trades = simulate_ml(
        df, oos_prob,
        th_long=args.th_long, th_short=args.th_short,
        tp_pct=args.tp_pct, sl_pct=args.sl_pct,
        time_stop_bars=args.time_stop_bars,
        horizon=args.horizon,
    )

    stats = summarize(trades)
    if trades:
        pd.DataFrame([asdict(t) for t in trades]).to_csv(args.out_trades, index=False)

    avg_oos = folds["oos_auc"].mean() if not folds.empty else float("nan")
    avg_is = folds["is_auc"].mean() if not folds.empty else float("nan")

    lines = [
        "=" * 78,
        f"INTRADAY LIGHTGBM -- {args.days}d, 5m bars, horizon={args.horizon} bars",
        "=" * 78,
        f"folds: {len(folds)}  avg IS AUC: {avg_is:.4f}  avg OOS AUC: {avg_oos:.4f}",
        f"trades: {stats.get('n', 0)}",
    ]
    if stats.get("n", 0) > 0:
        lines += [
            f"sessions:      {stats['n_days']}  (avg {stats['avg_trades_per_day']} trades/day)",
            f"winrate:       {stats['winrate']:.2%}",
            f"avg win:       {stats['avg_win']:+.3%}   avg loss: {stats['avg_loss']:+.3%}",
            f"payoff:        {stats['payoff']:.2f}",
            f"expectancy:    gross {stats['expectancy_gross']:+.3%}  |  "
            f"net {stats['expectancy_net']:+.3%}  (after {config.TOTAL_COST_ROUNDTRIP*100:.2f}% cost)",
            f"sharpe (d):    {stats['sharpe_daily']:.2f}",
            f"max drawdown:  {stats['mdd']:.2%}",
            f"by reason:     {stats['by_reason']}",
            "",
            "by ticker:",
            str(stats["by_ticker"]),
        ]
    summary = "\n".join(lines)
    Path(args.out_summary).write_text(summary, encoding="utf-8")
    try:
        print(summary)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(summary.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
