"""Map free text → list of ETF tickers from the v3 universe (17 ETFs).

Matching rules (strict — false positives matter more than recall for IC):
  1. `$TICKER` cashtag form always matches (case-insensitive).
  2. Bare TICKER (no $) matches only when wrapped by word boundaries AND
     the ticker has length ≥ 3 (avoids "SH" / "DOG" colliding with English).
     2-char tickers (none currently in universe) would require cashtag.
  3. ETF name fragments map onto the ticker (e.g. "Nasdaq Biotech" → IBB,
     "VIX futures" → VXX). Kept narrow — only unambiguous phrases.

Returns a set so a single article is counted once per ticker.
"""
from __future__ import annotations

import re
from typing import Iterable

import config

# Tickers we score on — superset of nightly_retrain.EXPANDED_TICKERS plus VXX
# (which is in the active Phase 4 universe and surfaces in v3 top-5 signals).
NEW_SECTORS = ["XLB", "XLU", "XLP", "XLC", "IBB"]
TRACKED_TICKERS: list[str] = (
    config.CORE_ETFS + config.SECTOR_ETFS + NEW_SECTORS
    + config.INVERSE_ETFS + config.VOLATILITY_ETFS
)

# Short tickers that collide with English words — require cashtag prefix.
_AMBIGUOUS = {"SH", "DOG", "DIA"}

# Narrow alias map. Keep only phrases unambiguous in finance context.
_ALIASES: dict[str, list[str]] = {
    "SPY":  ["s&p 500 etf", "spdr s&p 500"],
    "QQQ":  ["nasdaq 100 etf", "invesco qqq"],
    "DIA":  ["dow jones etf", "spdr dow"],
    "SH":   ["short s&p500", "short s&p 500"],
    "PSQ":  ["short qqq", "short nasdaq"],
    "DOG":  ["short dow"],
    "XLK":  ["technology select", "tech sector etf"],
    "XLF":  ["financial select", "financials etf"],
    "XLE":  ["energy select", "energy sector etf"],
    "XLV":  ["health care select", "healthcare etf"],
    "XLY":  ["consumer discretionary select"],
    "XLI":  ["industrial select"],
    "XLB":  ["materials select"],
    "XLU":  ["utilities select"],
    "XLP":  ["consumer staples select"],
    "XLC":  ["communication services select"],
    "IBB":  ["nasdaq biotech", "biotech etf"],
    "VXX":  ["vix futures etf", "ipath vix"],
}

_CASHTAG_RE = re.compile(r"\$([A-Za-z]{2,5})\b")
_BARE_RES: dict[str, re.Pattern[str]] = {
    t: re.compile(rf"(?<![A-Za-z0-9$]){t}(?![A-Za-z0-9])")
    for t in TRACKED_TICKERS
    if len(t) >= 3 and t not in _AMBIGUOUS
}


def extract_tickers(text: str | None) -> set[str]:
    """Return the set of tracked tickers mentioned in `text`."""
    if not text:
        return set()
    found: set[str] = set()

    for raw in _CASHTAG_RE.findall(text):
        t = raw.upper()
        if t in TRACKED_TICKERS:
            found.add(t)

    for t, rx in _BARE_RES.items():
        if rx.search(text):
            found.add(t)

    text_low = text.lower()
    for t, phrases in _ALIASES.items():
        for p in phrases:
            if p in text_low:
                found.add(t)
                break

    return found


def extract_from_documents(docs: Iterable[dict]) -> dict[str, int]:
    """Count ticker mentions across a stream of {title, body} dicts.

    A given ticker increments by 1 per document — body and title are merged
    so the same article never double-counts."""
    counts: dict[str, int] = {t: 0 for t in TRACKED_TICKERS}
    for d in docs:
        text = " ".join(filter(None, [d.get("title"), d.get("body")]))
        for t in extract_tickers(text):
            counts[t] += 1
    return counts
