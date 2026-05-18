"""Walk-forward retrain with regime features (MOVE/SKEW/VIX-term) added.

Compares against current v3 baseline OOS proba. Does NOT touch production
weights unless the new model wins on BOTH:
  1. Mean OOS AUC (raw model quality)
  2. Conviction_v4 strategy WR (downstream business metric)

Outputs (always written, regardless of promotion):
  data/logs/phase3_v5_oos_proba.parquet    — new OOS predictions
  data/logs/phase3_v5_wf.csv               — per-fold AUC + sample counts
  models/weights/lgbm_phase3_v5_staging.pkl — staged model

Run:  python scripts/retrain_v5_with_regime_features.py
"""
from __future__ import annotations

import io
import logging
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

import config
from data.collector import load_parquet
from models.train_lgbm_v2 import (
    build_feature_matrix_v2,
    build_target_v2,
    walk_forward_lgbm_v2,
    MACRO_KEEP_FEATURES,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# Universe must match nightly_retrain.py EXPANDED_TICKERS
EXPANDED_TICKERS = (
    config.CORE_ETFS + config.SECTOR_ETFS + config.INVERSE_ETFS
    + ["XLB", "XLU", "XLP", "XLC", "IBB"] + config.VOLATILITY_ETFS
)


def main():
    log.info("=== v5 retrain — regime features added ===")
    log.info("New MACRO_KEEP_FEATURES count: %d", len(MACRO_KEEP_FEATURES))
    log.info("Added vs v3: skew_z, skew_chg_5d, move_z, move_chg_5d, "
             "vix_term_9d, vix_term_9d_z, vix_term_3m")

    ohlcv = load_parquet("etf_ohlcv")
    close_full = ohlcv["Close"] if isinstance(ohlcv.columns, pd.MultiIndex) else ohlcv
    vix_df = load_parquet("vix")
    tickers = [t for t in EXPANDED_TICKERS if t in close_full.columns]
    close_df = close_full[tickers].dropna(how="all")

    log.info("Universe: %d ETFs; date range %s..%s (n=%d)",
             len(tickers),
             close_df.index.min().date(), close_df.index.max().date(),
             len(close_df))

    log.info("Building v5 feature matrix...")
    X = build_feature_matrix_v2(close_df, vix_df, macro=None)
    log.info("Feature matrix: %d rows × %d cols", *X.shape)
    log.info("Columns: %s", list(X.columns))

    y = build_target_v2(close_df, horizon=5)
    log.info("Target: %d rows (5d forward)", len(y))

    # Train uniform (no recency weighting — v3 production uses uniform).
    log.info("Walk-forward (5 splits, uniform weight)...")
    out_dir = Path("models/weights")
    proba_oos, wf = walk_forward_lgbm_v2(
        X, y, n_splits=5, use_recency_weight=False,
        save_dir=out_dir, save_suffix="v5_staging",
    )
    log.info("WF summary:\n%s", wf.to_string(index=False))
    log.info("Mean OOS AUC: %.4f (v3 baseline ~0.5457)",
             float(wf["oos_auc"].mean()))

    # Save OOS proba
    proba_path = Path("data/logs/phase3_v5_oos_proba.parquet")
    proba_path.parent.mkdir(parents=True, exist_ok=True)
    proba_oos.to_frame(name="proba").to_parquet(proba_path)
    wf.to_csv(Path("data/logs/phase3_v5_wf.csv"), index=False)
    log.info("Saved OOS proba → %s", proba_path)


if __name__ == "__main__":
    main()
