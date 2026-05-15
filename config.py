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

TIER2_PAPER_MONTHS_MIN = 3
TIER2_PAPER_BACKTEST_GAP_MAX = 0.40
TIER2_PAPER_SHARPE_MIN = 0.5

# ── Model ─────────────────────────────────────────────────────────────────────
IC_MIN_VIABLE = 0.01              # Below this, signal is buried by costs
LEVERAGE_CONFIDENCE_THRESHOLD = 0.80   # Empirically validated; update after Phase 5

# ── Data ──────────────────────────────────────────────────────────────────────
HISTORY_YEARS = 11                # Must cover 2018, 2020, 2022 bear markets

FRED_SERIES = {
    "fed_funds": "FEDFUNDS",
    "dxy": "DTWEXBGS",
    "credit_spread": "BAMLH0A0HYM2",
    "yield_10y": "GS10",
    "yield_2y": "GS2",
}
