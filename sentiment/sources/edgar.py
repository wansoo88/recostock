"""SEC EDGAR full-text search — no API key required.

Endpoint: https://efts.sec.gov/LATEST/search-index
Docs:     https://efts.sec.gov/LATEST/search-index?q=&forms=&dateRange=custom&startdt=&enddt=

We query the *formal* ETF name (e.g. "Energy Select Sector SPDR") to catch
filings that reference each fund. Each hit becomes a document carrying a
`$TICKER` cashtag in its body so the existing ticker_extract picks it up
without ETF-name aliasing.

Rate limit: SEC requires User-Agent identifying the requester and caps at
10 req/s. We do 17 ETF queries with 0.2s delay → ~3.5s total.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from typing import Iterable

log = logging.getLogger(__name__)

BASE = "https://efts.sec.gov/LATEST/search-index"
# SEC fair-access policy: identify yourself with company + contact email.
UA = "recostock-research teamdata@serveone.co.kr"
DELAY_S = 0.2
TIMEOUT_S = 15

# Forms most likely to mention an ETF by formal name. We exclude 13F-HR
# (institutional 13F filings list hundreds of names — too noisy).
DEFAULT_FORMS = "8-K,10-Q,10-K,N-PORT,N-CSR,485BPOS"

# Quoted formal ETF names → ticker. Strict on phrasing so we don't pull in
# unrelated funds (e.g. "Energy ETF" matches dozens of funds; "Energy Select
# Sector SPDR" is uniquely the State Street XLE family).
ETF_QUERIES: dict[str, str] = {
    "SPY":  '"SPDR S&P 500"',
    "QQQ":  '"Invesco QQQ"',
    "DIA":  '"SPDR Dow Jones Industrial"',
    "SH":   '"ProShares Short S&P500"',
    "PSQ":  '"ProShares Short QQQ"',
    "DOG":  '"ProShares Short Dow30"',
    "XLK":  '"Technology Select Sector SPDR"',
    "XLF":  '"Financial Select Sector SPDR"',
    "XLE":  '"Energy Select Sector SPDR"',
    "XLV":  '"Health Care Select Sector SPDR"',
    "XLY":  '"Consumer Discretionary Select Sector SPDR"',
    "XLI":  '"Industrial Select Sector SPDR"',
    "XLB":  '"Materials Select Sector SPDR"',
    "XLU":  '"Utilities Select Sector SPDR"',
    "XLP":  '"Consumer Staples Select Sector SPDR"',
    "XLC":  '"Communication Services Select Sector SPDR"',
    "IBB":  '"iShares Biotechnology"',
    "VXX":  '"iPath Series B S&P 500 VIX"',
}


def _fetch_one(ticker: str, query: str, startdt: str, enddt: str,
               forms: str) -> list[dict]:
    params = {
        "q": query,
        "forms": forms,
        "dateRange": "custom",
        "startdt": startdt,
        "enddt": enddt,
    }
    url = f"{BASE}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        log.warning("EDGAR %s query %r failed: %s", ticker, query, exc)
        return []
    except json.JSONDecodeError:
        return []

    docs: list[dict] = []
    for hit in data.get("hits", {}).get("hits", []):
        src = hit.get("_source", {})
        file_date = src.get("file_date")
        if not file_date:
            continue
        try:
            pub = datetime.strptime(file_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        company = (src.get("display_names") or ["?"])[0]
        form = src.get("form", "?")
        docs.append({
            "source": "edgar",
            "query_ticker": ticker,
            "title": f"{company} filed {form}",
            # Embed cashtag so ticker_extract attributes the hit without
            # depending on ETF-name aliases.
            "body": f"${ticker} — EDGAR {form}",
            "published": pub,
        })
    return docs


def fetch(tickers: Iterable[str] | None = None, lookback_days: int = 2,
          forms: str = DEFAULT_FORMS) -> list[dict]:
    """Fetch recent filings whose text mentions each ETF's formal name."""
    if tickers is None:
        tickers = list(ETF_QUERIES.keys())
    else:
        tickers = [t for t in tickers if t in ETF_QUERIES]

    today = datetime.now(timezone.utc).date()
    startdt = (today - timedelta(days=lookback_days)).isoformat()
    enddt = today.isoformat()

    out: list[dict] = []
    for t in tickers:
        q = ETF_QUERIES[t]
        out.extend(_fetch_one(t, q, startdt, enddt, forms))
        time.sleep(DELAY_S)
    log.info("edgar: fetched %d filings across %d tickers [%s..%s]",
             len(out), len(tickers), startdt, enddt)
    return out
