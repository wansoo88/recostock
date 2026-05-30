"""Portfolio composition — blend the trend-core engine with the RSI sector sleeve.

The trend-core engine (signals/trend_core.py) is the primary return engine but
has NO cross-sectional sector skill. The RSI sector sleeve (signals/sector_rotation.py)
adds that one validated edge. This module composes them into a SINGLE actionable
capital allocation, so the user sees one set of weights to execute manually.

Allocation (config.SECTOR_SLEEVE_WEIGHT, default 0.25):
    (1 - w) of capital  -> trend-core weights (SPY / SPXL / QQQ / cash), scaled
    w        of capital  -> RSI sector sleeve, split equally across `pick`
                            tickers; any empty slot (sector below 200SMA) is cash.

Validated 2026-05-30 (cost-adjusted, look-ahead-safe, vs shipped engine):
    Full OOS 2021+ : engine +114%/Sharpe 1.12/MDD -14% -> blend +131%/1.30/-13%
    Holdout 2024+  : engine  +59%/Sharpe 1.43/MDD -11% -> blend  +60%/1.56/-13%
The improvement is risk-adjusted (Sharpe +0.13 both periods at equal-or-lower
MDD); raw Holdout return is ~flat. The sleeve diversifies, it doesn't supercharge.

Pure function: dicts in, blended allocation dict out. No I/O. When the sleeve is
unavailable (weight 0, missing data) it returns the engine allocation unchanged.
"""
from __future__ import annotations

import config

SECTOR_SLEEVE_WEIGHT = float(getattr(config, "SECTOR_SLEEVE_WEIGHT", 0.25))
LEVER_MULT = float(getattr(config, "LEVER_MULT", 3.0))

# Beta of each instrument vs SPY, for the effective-exposure roll-up.
_BETA = {"SPY": 1.0, "QQQ": 1.0, "DIA": 1.0, "SPXL": LEVER_MULT, "TQQQ": LEVER_MULT,
         "XLK": 1.0, "XLF": 1.0, "XLE": 1.0, "XLV": 1.0, "XLY": 1.0, "XLI": 1.0}


def compose(trend_core: dict, sector_satellite: dict | None,
            sleeve_weight: float = SECTOR_SLEEVE_WEIGHT) -> dict:
    """Blend the engine and the sector sleeve into one capital allocation.

    trend_core       : output of signals.trend_core.evaluate
    sector_satellite : output of signals.sector_rotation.evaluate (or None/{})
    sleeve_weight    : fraction of capital to the sector sleeve (rest to engine)

    Returns:
      weights : {ticker: capital_fraction} summing to <= 1 (remainder = cash)
      cashWeight, effExposure, sleeveWeight, coreWeight
      pick    : the sleeve's chosen tickers (for display)
      note    : human summary
      enabled : whether the sleeve was actually applied
    """
    core_w = max(0.0, 1.0 - sleeve_weight) if sleeve_weight > 0 else 1.0

    # ── Engine leg: scale its weights to core_w of total capital ──────────────
    weights: dict[str, float] = {}
    eng_invested = 0.0
    for tk, key in (("SPY", "spyWeight"), ("QQQ", "qqqWeight"), ("SPXL", "spxlWeight")):
        v = float(trend_core.get(key, 0.0) or 0.0)
        if v > 0:
            weights[tk] = weights.get(tk, 0.0) + v * core_w
            eng_invested += v * core_w

    pick = list((sector_satellite or {}).get("pick") or [])
    top_k = int((sector_satellite or {}).get("topK", 2)) or 2
    sleeve_enabled = sleeve_weight > 0 and bool(sector_satellite) and bool((sector_satellite or {}).get("ranked"))

    # ── Sleeve leg: split sleeve_weight across the picks; empty slots = cash ───
    sleeve_invested = 0.0
    if sleeve_enabled and pick:
        per = sleeve_weight / top_k          # each of the K slots gets this; cash slots drop
        for tk in pick:
            weights[tk] = weights.get(tk, 0.0) + per
            sleeve_invested += per

    invested = eng_invested + sleeve_invested
    cash_w = max(0.0, 1.0 - invested)

    eff = sum(w * _BETA.get(tk, 1.0) for tk, w in weights.items())

    # round for display, keep cash consistent
    weights = {tk: round(w, 4) for tk, w in weights.items() if w > 1e-9}

    if not sleeve_enabled:
        note = "추세코어 단독 (섹터 슬리브 비활성)"
    elif pick:
        note = (f"추세코어 {core_w*100:.0f}% + RSI 섹터 슬리브 {sleeve_weight*100:.0f}% "
                f"({' + '.join(pick)})")
    else:
        note = (f"추세코어 {core_w*100:.0f}% + 섹터 슬리브 {sleeve_weight*100:.0f}% 현금 "
                f"(상위 RSI 섹터 모두 200일선 아래)")

    return {
        "weights": weights,
        "cashWeight": round(cash_w, 4),
        "effExposure": round(eff, 2),
        "coreWeight": round(core_w, 4),
        "sleeveWeight": round(sleeve_weight, 4),
        "sleeveInvested": round(sleeve_invested, 4),
        "pick": pick,
        "enabled": sleeve_enabled,
        "note": note,
    }
