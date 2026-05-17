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

# ── Conviction strategy (env STRATEGY_MODE=conviction_v1 activates) ────────
# Four iterations, each layer additive walk-forward improvement:
#   v1 (single EMA-5):                        n=36  WR 58.33%  Total +12.85%
#   v2 (+ Multi-EMA 3/5/7):                   n=33  WR 63.64%  Total +16.60%
#   v3 (+ SKEW z<1.0 + VIX9D/VIX<1.0):       n=20  WR 70.00%  Total +14.37%
#   v4 (+ MOVE bond-vol z<1.0):              n=19  WR 73.68%  Total +15.62%  ⭐
# Source: scripts/experiment_v4_layered.py.
# All pass Signal.is_valid(). v4 stable across Full period (WR 60.00% n=35).
CONVICTION_TOP_K = 1
CONVICTION_THRESHOLD = 0.65
CONVICTION_SL_PCT = 0.010         # 1.0% stop-loss
CONVICTION_TP_PCT = 0.030         # 3.0% take-profit
CONVICTION_VIX_MAX = 20.0         # Skip when VIX >= 20 (panic regime → signal degrades)
CONVICTION_REQUIRE_SPY_UPTREND = True   # Skip when SPY < 200-day SMA
CONVICTION_MULTI_EMA_CONFIRM = True      # v2: require EMA-3, 5, 7 all ≥ threshold
# v3: Options-market regime overlays (set to None to disable).
CONVICTION_SKEW_Z_MAX: float | None = 1.0    # Skip when CBOE SKEW z-score (60d) >= 1.0
CONVICTION_VIX_TERM_MAX: float | None = 1.0  # Skip when VIX9D/VIX >= 1.0 (backwardation)
CONVICTION_SKEW_Z_WINDOW = 60                # Days for SKEW z-score baseline
# v4: Bond-market regime overlay (MOVE z-score, 60-day window).
CONVICTION_MOVE_Z_MAX: float | None = 1.0    # Skip when MOVE z-score >= 1.0 (bond stress)
CONVICTION_MOVE_Z_WINDOW = 60
# Backtested expectations (v4 combo holdout, n=19):
CONVICTION_EXPECTED_WINRATE = 0.737
CONVICTION_EXPECTED_PAYOFF = 1.25
CONVICTION_EXPECTED_SAMPLE_N = 19

# ── Data ──────────────────────────────────────────────────────────────────────
HISTORY_YEARS = 11                # Must cover 2018, 2020, 2022 bear markets

FRED_SERIES = {
    "fed_funds": "FEDFUNDS",
    "dxy": "DTWEXBGS",
    "credit_spread": "BAMLH0A0HYM2",
    "yield_10y": "GS10",
    "yield_2y": "GS2",
}
