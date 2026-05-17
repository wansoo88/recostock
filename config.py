"""Single source of truth for all constants. Never hardcode trading costs elsewhere."""

# ── Trading costs (Toss Securities) ──────────────────────────────────────────
COMMISSION_ONE_WAY = 0.001        # 0.1% per trade
COMMISSION_ROUNDTRIP = 0.002      # 0.2% roundtrip
SLIPPAGE_ROUNDTRIP = 0.0005       # 0.05% conservative slippage
TOTAL_COST_ROUNDTRIP = COMMISSION_ROUNDTRIP + SLIPPAGE_ROUNDTRIP  # 0.25%

# ── Universe ──────────────────────────────────────────────────────────────────
CORE_ETFS = ["SPY", "QQQ", "DIA"]
INVERSE_ETFS = ["SH", "PSQ", "DOG"]
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI"]
VOLATILITY_ETFS = ["VXX"]
LEVERAGE_LONG_ETFS = ["QLD", "TQQQ", "SPXL", "SOXL"]   # Phase 5 + education
LEVERAGE_INVERSE_ETFS = ["SQQQ", "SPXS", "SOXS"]        # Phase 5 + education

VIX_TICKER = "^VIX"

# ── Phase gate thresholds ─────────────────────────────────────────────────────
TIER1_SHARPE_MIN = 0.7
TIER1_MDD_MAX = 0.25
TIER1_OOS_IS_RATIO_MIN = 0.4
TIER1_MIN_TRADING_DAYS = 120
TIER1_WF_POSITIVE_MIN = 0.5       # ≥ half of walk-forward windows must be positive

TIER2_PAPER_MONTHS_MIN = 3
TIER2_PAPER_BACKTEST_GAP_MAX = 0.40
TIER2_PAPER_SHARPE_MIN = 0.5

# ── Model ─────────────────────────────────────────────────────────────────────
IC_MIN_VIABLE = 0.01              # Below this, signal is buried by costs
SIGNAL_THRESHOLD = 0.53           # EMA-smoothed probability gate
MIN_PAYOFF = 1.1                  # Minimum payoff (avg_win/avg_loss) for signal validity
LEVERAGE_CONFIDENCE_THRESHOLD = 0.80   # Empirically validated; update after Phase 5

# ── Conviction strategy v1 (env STRATEGY_MODE=conviction_v1 activates) ──────
# Walk-forward holdout (2024-01~2026-05, n=36):
#   WR 58.33%, Payoff 1.20, Sharpe 1.67, MDD -11.1%, Total +13.22%
#   All Tier 1 gates pass; Signal.is_valid() satisfied.
# Identified in scripts/experiment_tp_sweep.py (SL×TP grid).
CONVICTION_TOP_K = 1
CONVICTION_THRESHOLD = 0.65
CONVICTION_SL_PCT = 0.010         # 1.0% stop-loss
CONVICTION_TP_PCT = 0.030         # 3.0% take-profit
CONVICTION_VIX_MAX = 20.0         # Skip when VIX >= 20 (panic regime → signal degrades)
CONVICTION_REQUIRE_SPY_UPTREND = True   # Skip when SPY < 200-day SMA
# Backtested expectations to plug into Signal stats (rolling_stats can't reflect
# the new TP/SL exit, so we use measured holdout averages):
CONVICTION_EXPECTED_WINRATE = 0.583
CONVICTION_EXPECTED_PAYOFF = 1.20
CONVICTION_EXPECTED_SAMPLE_N = 36

# ── Data ──────────────────────────────────────────────────────────────────────
HISTORY_YEARS = 11                # Must cover 2018, 2020, 2022 bear markets

FRED_SERIES = {
    "fed_funds": "FEDFUNDS",
    "dxy": "DTWEXBGS",
    "credit_spread": "BAMLH0A0HYM2",
    "yield_10y": "GS10",
    "yield_2y": "GS2",
}
