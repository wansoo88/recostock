"""Model inference — stub for Phase 0.

Phase 1-2: baseline rules.
Phase 3+: load LightGBM / GRU weights from models/weights/ and return scores.

Training always happens outside GitHub Actions (local/separate env).
Commit weights to models/weights/ after training.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

WEIGHTS_DIR = Path("models/weights")


def score_direction(features: pd.DataFrame) -> pd.Series:
    """Return direction score per ETF in [-1, +1]. Positive = long bias.

    Phase 0: returns 0.0 (no signal) until weights are available.
    """
    return pd.Series(0.0, index=features.columns if hasattr(features, "columns") else [])


def score_confidence(features: pd.DataFrame) -> pd.Series:
    """Return model confidence in [0, 1] per ETF.

    Phase 0: returns 0.5 (maximum uncertainty).
    """
    return pd.Series(0.5, index=features.columns if hasattr(features, "columns") else [])
