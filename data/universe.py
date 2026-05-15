from dataclasses import dataclass
from enum import IntEnum
import config


class Phase(IntEnum):
    PHASE4 = 4
    PHASE5 = 5


@dataclass(frozen=True)
class ETFMeta:
    ticker: str
    name: str
    category: str  # core | inverse | sector | volatility | leverage_long | leverage_inverse
    active_from: Phase
    leverage: int
    inverse: bool
    requires_education: bool  # Leverage ETF needs prior completion of KFB online course


UNIVERSE: list[ETFMeta] = [
    # Core 1x
    ETFMeta("SPY", "SPDR S&P 500", "core", Phase.PHASE4, 1, False, False),
    ETFMeta("QQQ", "Invesco QQQ", "core", Phase.PHASE4, 1, False, False),
    ETFMeta("DIA", "SPDR Dow Jones", "core", Phase.PHASE4, 1, False, False),
    # Inverse 1x
    ETFMeta("SH", "ProShares Short S&P500", "inverse", Phase.PHASE4, 1, True, False),
    ETFMeta("PSQ", "ProShares Short QQQ", "inverse", Phase.PHASE4, 1, True, False),
    ETFMeta("DOG", "ProShares Short Dow30", "inverse", Phase.PHASE4, 1, True, False),
    # Sector 1x
    ETFMeta("XLK", "Technology Select", "sector", Phase.PHASE4, 1, False, False),
    ETFMeta("XLF", "Financial Select", "sector", Phase.PHASE4, 1, False, False),
    ETFMeta("XLE", "Energy Select", "sector", Phase.PHASE4, 1, False, False),
    ETFMeta("XLV", "Health Care Select", "sector", Phase.PHASE4, 1, False, False),
    ETFMeta("XLY", "Consumer Disc Select", "sector", Phase.PHASE4, 1, False, False),
    ETFMeta("XLI", "Industrial Select", "sector", Phase.PHASE4, 1, False, False),
    # Volatility
    ETFMeta("VXX", "iPath S&P 500 VIX ST Futures", "volatility", Phase.PHASE4, 1, False, False),
    # Leverage long — Phase 5 + mandatory KFB education
    ETFMeta("QLD", "ProShares Ultra QQQ 2x", "leverage_long", Phase.PHASE5, 2, False, True),
    ETFMeta("TQQQ", "ProShares UltraPro QQQ 3x", "leverage_long", Phase.PHASE5, 3, False, True),
    ETFMeta("SPXL", "Direxion Daily S&P500 Bull 3x", "leverage_long", Phase.PHASE5, 3, False, True),
    ETFMeta("SOXL", "Direxion Daily Semiconductor Bull 3x", "leverage_long", Phase.PHASE5, 3, False, True),
    # Leverage inverse — Phase 5 + mandatory KFB education
    ETFMeta("SQQQ", "ProShares UltraPro Short QQQ 3x", "leverage_inverse", Phase.PHASE5, 3, True, True),
    ETFMeta("SPXS", "Direxion Daily S&P500 Bear 3x", "leverage_inverse", Phase.PHASE5, 3, True, True),
    ETFMeta("SOXS", "Direxion Daily Semiconductor Bear 3x", "leverage_inverse", Phase.PHASE5, 3, True, True),
]

UNIVERSE_BY_TICKER: dict[str, ETFMeta] = {e.ticker: e for e in UNIVERSE}


def get_active_universe(phase: int, leverage_education_done: bool = False) -> list[ETFMeta]:
    """Return ETFs active for the given phase. Leverage excluded until education complete."""
    active = [e for e in UNIVERSE if e.active_from <= phase]
    if not leverage_education_done:
        active = [e for e in active if not e.requires_education]
    return active


def get_tickers(phase: int, leverage_education_done: bool = False) -> list[str]:
    return [e.ticker for e in get_active_universe(phase, leverage_education_done)]
