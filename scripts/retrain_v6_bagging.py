"""v6 bagging ensemble — 5 LightGBM models with different seeds, averaged.

Premise: a single random seed introduces variance in tree splits. Averaging
across 5 seeds (bagging-by-randomness) typically lowers noise and produces
better-calibrated probabilities, especially near the decision boundary
where strategy thresholds operate.

Only walks forward 5 folds × 5 seeds = 25 model fits. Saves the averaged
OOS proba so we can A/B test against v4 with the same conviction_v4 strategy.

Run:  python scripts/retrain_v6_bagging.py
"""
from __future__ import annotations

import io
import logging
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
    LGBM_PARAMS_V2,
    MACRO_KEEP_FEATURES,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

EXPANDED_TICKERS = (
    config.CORE_ETFS + config.SECTOR_ETFS + config.INVERSE_ETFS
    + ["XLB", "XLU", "XLP", "XLC", "IBB"] + config.VOLATILITY_ETFS
)
SEEDS = [42, 43, 44, 45, 46]


def main():
    log.info("=== v6 bagging ensemble — %d seeds, %d MACRO features ===",
             len(SEEDS), len(MACRO_KEEP_FEATURES))
    ohlcv = load_parquet("etf_ohlcv")
    close_full = ohlcv["Close"] if isinstance(ohlcv.columns, pd.MultiIndex) else ohlcv
    vix_df = load_parquet("vix")
    tickers = [t for t in EXPANDED_TICKERS if t in close_full.columns]
    close_df = close_full[tickers].dropna(how="all")

    X = build_feature_matrix_v2(close_df, vix_df, macro=None)
    y = build_target_v2(close_df, horizon=5)
    log.info("Feature matrix: %d rows × %d cols", *X.shape)

    proba_list = []
    for i, seed in enumerate(SEEDS):
        log.info("--- Seed %d/%d (random_state=%d) ---", i + 1, len(SEEDS), seed)
        original_seed = LGBM_PARAMS_V2["random_state"]
        LGBM_PARAMS_V2["random_state"] = seed
        try:
            proba, wf = walk_forward_lgbm_v2(
                X, y, n_splits=5, use_recency_weight=False,
                save_dir=None,   # do not save individual models
                save_suffix=f"v6_seed{seed}",
            )
            mean_auc = float(wf["oos_auc"].mean())
            log.info("Seed %d  Mean OOS AUC = %.4f", seed, mean_auc)
            proba_list.append(proba)
        finally:
            LGBM_PARAMS_V2["random_state"] = original_seed

    # Average probabilities across seeds (align on common index)
    common_idx = proba_list[0].index
    for p in proba_list[1:]:
        common_idx = common_idx.intersection(p.index)
    aligned = pd.concat([p.reindex(common_idx) for p in proba_list], axis=1)
    aligned.columns = [f"seed_{s}" for s in SEEDS]
    bag_proba = aligned.mean(axis=1)

    out = Path("data/logs/phase3_v6_bagging_oos_proba.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    bag_proba.to_frame(name="proba").to_parquet(out)
    log.info("Saved averaged OOS proba → %s (n=%d)", out, len(bag_proba))

    # Quick correlation check among seeds (high = bagging gives less benefit)
    corr = aligned.corr().mean().mean()
    log.info("Mean pairwise correlation across seeds: %.4f "
             "(lower = more independence = more bagging benefit)", corr)


if __name__ == "__main__":
    main()
