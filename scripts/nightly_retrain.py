"""Nightly retrain of v3 LightGBM model with safety gate.

Runs from GitHub Actions cron (14:00 UTC, Mon–Fri) — well before signal cron
(20:30 / 21:30 UTC) so the signal pipeline picks up the new weights.

Steps:
  1. Fetch latest yfinance data (OHLCV + VIX + macro proxies)
  2. Build v3 feature matrix on expanded universe (17 ETFs)
  3. Walk-forward train (5-split, uniform weighting)
  4. Safety gates:
       - mean OOS AUC >= MIN_MEAN_OOS_AUC (0.50)
       - min  OOS AUC >= MIN_FOLD_OOS_AUC (0.45)
  5. PASS → save new weights + record metadata
     FAIL → keep old weights, record FAILED row, exit 2

Exit codes:
  0  weights updated
  2  safety gate failed — old weights kept
  1  pipeline error (fetch / train failure)

Telegram alert is sent on both PASS and FAIL when TELEGRAM_* env vars are set.
"""
from __future__ import annotations

import logging
import os
import pickle
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root is on PYTHONPATH when run as a script (matches run_daily.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

import config
from data.collector import (
    fetch_etf_ohlcv,
    fetch_vix,
    fetch_macro,
    fetch_macro_yfinance,
    save_parquet,
    load_parquet,
)
from data.macro_collector import load_macro_cache
from data.universe import UNIVERSE_BY_TICKER
from features.factors import compute_factors
from features.macro_factors import build_global_macro
from models.train_lgbm_v2 import (
    walk_forward_lgbm_v2,
    build_target_v2,
    MACRO_KEEP_FEATURES,
    FACTOR_COLS,
    RANK_COLS,
    _CATEGORY_MAP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("nightly_retrain")


# ── Safety thresholds ────────────────────────────────────────────────────────
MIN_MEAN_OOS_AUC = 0.50
MIN_FOLD_OOS_AUC = 0.45
MIN_TRADING_DAYS = 2000

# ── Universe (must match models/inference_v3.py expectation) ─────────────────
NEW_SECTORS = ["XLB", "XLU", "XLP", "XLC", "IBB"]
EXPANDED_TICKERS = (
    config.CORE_ETFS + config.SECTOR_ETFS + NEW_SECTORS + config.INVERSE_ETFS
)

WEIGHTS_DIR = Path("models/weights")
WEIGHTS_FILE = WEIGHTS_DIR / "lgbm_phase3_v3_uniform.pkl"
IMPORTANCE_FILE = WEIGHTS_DIR / "lgbm_phase3_v3_uniform_importance.csv"
HISTORY_CSV = Path("data/logs/retrain_history.csv")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_feature_matrix(close_df: pd.DataFrame, vix_df: pd.DataFrame,
                          macro: dict) -> pd.DataFrame:
    """Same shape as scripts/integrated_backtest_v3.build_feature_matrix_expanded.

    Inlined here so this script does not import a side-effect module."""
    ticker_factors: dict[str, pd.DataFrame] = {}
    for t in close_df.columns:
        close = close_df[t].dropna()
        if len(close) < 100:
            continue
        ticker_factors[t] = compute_factors(close)
    if not ticker_factors:
        return pd.DataFrame()

    common_idx = None
    for f in ticker_factors.values():
        common_idx = f.index if common_idx is None else common_idx.intersection(f.index)

    rank_mats: dict[str, pd.DataFrame] = {}
    for col in RANK_COLS:
        mat = pd.DataFrame({
            t: ticker_factors[t][col].reindex(common_idx)
            for t in ticker_factors if col in ticker_factors[t].columns
        })
        rank_mats[col] = mat.rank(axis=1, pct=True)

    global_mac = build_global_macro(common_idx, macro)
    macro_cols = [c for c in MACRO_KEEP_FEATURES if c in global_mac.columns]
    macro_aligned = global_mac[macro_cols].reindex(common_idx).ffill()

    parts: list[pd.DataFrame] = []
    for t, f in ticker_factors.items():
        meta = UNIVERSE_BY_TICKER.get(t)
        is_inverse = int(meta.inverse) if meta else 0
        category_code = _CATEGORY_MAP.get(meta.category, 1) if meta else 1
        sub = f.reindex(common_idx)[FACTOR_COLS].copy()
        sub["is_inverse"] = is_inverse
        sub["category_code"] = category_code
        for col in RANK_COLS:
            if col in rank_mats and t in rank_mats[col].columns:
                sub[f"{col}_rank"] = rank_mats[col][t]
        for col in macro_cols:
            sub[col] = macro_aligned[col].values
        sub.index = pd.MultiIndex.from_arrays(
            [sub.index, [t] * len(sub)], names=["date", "ticker"]
        )
        parts.append(sub)
    features = pd.concat(parts).sort_index()

    if vix_df is not None:
        vix = vix_df.iloc[:, 0].clip(lower=1)
        date_idx = features.index.get_level_values("date")
        features["vix_log"] = date_idx.map(np.log(vix).to_dict())
        features["vix_chg_1d"] = date_idx.map(vix.pct_change().to_dict())

    return features.dropna()


def _refresh_data(skip_fetch: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fetch fresh data unless --skip-fetch (for local dry-run)."""
    if skip_fetch:
        log.info("[--skip-fetch] using cached parquet only")
    else:
        log.info("Fetching OHLCV for %d tickers…", len(EXPANDED_TICKERS))
        ohlcv = fetch_etf_ohlcv(EXPANDED_TICKERS)
        save_parquet(ohlcv, "etf_ohlcv")

        log.info("Fetching VIX…")
        vix_series = fetch_vix()
        save_parquet(vix_series, "vix")

        # FRED is optional — present in env on GitHub Actions, absent locally
        fred_key = os.environ.get("FRED_API_KEY", "")
        if fred_key:
            log.info("Fetching FRED macro…")
            macro = fetch_macro(fred_key)
            for k, v in macro.items():
                save_parquet(v, f"macro_{k}")
        else:
            log.info("FRED_API_KEY not set — skipping FRED (yfinance macro still used)")

        log.info("Fetching yfinance macro proxies…")
        fetch_macro_yfinance()

    ohlcv = load_parquet("etf_ohlcv")
    vix_df = load_parquet("vix")
    macro = load_macro_cache()

    tickers = [t for t in EXPANDED_TICKERS if t in ohlcv["Close"].columns]
    close_df = ohlcv["Close"][tickers].dropna(how="all")
    log.info("Universe (%d): %s", len(tickers), tickers)
    log.info("Date range: %s → %s (n=%d)",
             close_df.index.min().date(), close_df.index.max().date(), len(close_df))
    return close_df, vix_df, macro


def _evaluate_safety(wf: pd.DataFrame, n_rows: int) -> tuple[bool, str]:
    """Return (pass, reason). Strict — if anything is off, refuse the deploy."""
    if wf.empty:
        return False, "WF table empty — training produced no folds"
    mean_oos = float(wf["oos_auc"].mean())
    min_oos = float(wf["oos_auc"].min())
    if mean_oos < MIN_MEAN_OOS_AUC:
        return False, f"mean OOS AUC {mean_oos:.4f} < {MIN_MEAN_OOS_AUC} threshold"
    if min_oos < MIN_FOLD_OOS_AUC:
        return False, f"min fold OOS AUC {min_oos:.4f} < {MIN_FOLD_OOS_AUC} floor"
    if n_rows < MIN_TRADING_DAYS:
        return False, f"feature matrix only {n_rows} rows (< {MIN_TRADING_DAYS})"
    return True, f"mean OOS AUC {mean_oos:.4f}, min {min_oos:.4f}, rows {n_rows}"


def _append_history(row: dict) -> None:
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if HISTORY_CSV.exists():
        df = pd.concat([pd.read_csv(HISTORY_CSV), df], ignore_index=True)
    df.to_csv(HISTORY_CSV, index=False)
    log.info("History appended → %s", HISTORY_CSV)


def _notify_telegram(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not (token and chat_id):
        return
    try:
        import urllib.parse
        import urllib.request
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
        with urllib.request.urlopen(url, data=data, timeout=10) as resp:
            resp.read()
    except Exception as exc:
        log.warning("Telegram notification failed: %s", exc)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    skip_fetch = "--skip-fetch" in sys.argv
    dry_run = "--dry-run" in sys.argv  # do not write weights / history
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    try:
        close_df, vix_df, macro = _refresh_data(skip_fetch=skip_fetch)
    except Exception as exc:
        log.exception("Data fetch failed: %s", exc)
        _notify_telegram(f"recostock retrain FAIL — fetch error: {exc}")
        return 1

    try:
        X = _build_feature_matrix(close_df, vix_df, macro)
        y = build_target_v2(close_df, horizon=5)
        log.info("Feature matrix: %d rows × %d cols", *X.shape)
    except Exception as exc:
        log.exception("Feature build failed: %s", exc)
        _notify_telegram(f"recostock retrain FAIL — feature build: {exc}")
        return 1

    # Train to a staging directory; only promote to WEIGHTS_DIR after safety gate.
    with tempfile.TemporaryDirectory(prefix="retrain_stage_") as tmpdir:
        stage_dir = Path(tmpdir)
        try:
            log.info("Walk-forward training (5-split, uniform) → %s", stage_dir)
            _, wf = walk_forward_lgbm_v2(
                X, y,
                n_splits=5,
                use_recency_weight=False,
                save_dir=stage_dir,
                save_suffix="v3",
            )
            log.info("WF summary:\n%s", wf.to_string(index=False))
        except Exception as exc:
            log.exception("Training failed: %s", exc)
            _notify_telegram(f"recostock retrain FAIL — training: {exc}")
            return 1

        passed, reason = _evaluate_safety(wf, n_rows=len(X))
        mean_oos = float(wf["oos_auc"].mean())
        min_oos = float(wf["oos_auc"].min())

        row = {
            "ts_utc": now_utc,
            "status": "PASS" if passed else "FAIL",
            "mean_oos_auc": round(mean_oos, 4),
            "min_oos_auc": round(min_oos, 4),
            "n_rows": int(len(X)),
            "n_folds": int(len(wf)),
            "reason": reason,
            "dry_run": dry_run,
        }

        if dry_run:
            log.info("[--dry-run] would record %s — skipping disk writes", row)
            return 0 if passed else 2

        _append_history(row)

        if not passed:
            log.error("SAFETY GATE FAILED: %s — staged weights discarded", reason)
            _notify_telegram(
                f"recostock retrain FAIL ({now_utc} UTC)\n"
                f"{reason}\n"
                f"weights NOT updated — production stays on previous version."
            )
            return 2

        # Promote staged files into production location.
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stage_dir / WEIGHTS_FILE.name, WEIGHTS_FILE)
        importance_src = stage_dir / IMPORTANCE_FILE.name
        if importance_src.exists():
            shutil.copy2(importance_src, IMPORTANCE_FILE)
        log.info("SAFETY GATE PASSED: %s", reason)
        log.info("Promoted weights → %s", WEIGHTS_FILE)
    _notify_telegram(
        f"recostock retrain OK ({now_utc} UTC)\n"
        f"mean OOS AUC {mean_oos:.4f}, min {min_oos:.4f}, rows {len(X)}\n"
        f"weights → {WEIGHTS_FILE.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
