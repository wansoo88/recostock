"""SPY Put/Call ratio collector — computes from yfinance options chain.

yfinance does not carry the historical CBOE ^CPC / ^CPCE series, so we
compute our own P/C from the SPY options chain. Each daily run reads the
current chain for the first N expirations and aggregates put/call volume
and open-interest. The result is appended to a daily history parquet.

Monitoring-only as of 2026-05-18: we will NOT use P/C as a regime gate
until ~60 trading days have accumulated and we can walk-forward validate
its effect on WR.

Schema of data/raw/options/spy_pc_daily.csv:
    date          (ISO date)  run date
    vol_pc        (float)     sum(put volume) / sum(call volume) across sampled expirations
    oi_pc         (float)     sum(put OI) / sum(call OI)
    n_expirations (int)       number of expirations actually sampled
    underlying_px (float)     SPY price at fetch time (sanity)

CSV (not parquet) — yfinance pulls accumulate one row per day; the file
stays small enough that CSV is simpler and side-steps a pyarrow
extension-type collision that occasionally trips read/write in the same
process.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

PC_PATH = Path("data/raw/options/spy_pc_daily.csv")
DEFAULT_N_EXPIRATIONS = 10   # sample first 10 expirations (~3 months ahead)


def compute_spy_pc(n_expirations: int = DEFAULT_N_EXPIRATIONS) -> dict | None:
    """Compute current SPY put/call ratios from yfinance options chain.

    Returns dict with vol_pc / oi_pc / n_expirations / underlying_px,
    or None on failure (network error, no data, etc.).
    """
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed — skipping P/C fetch")
        return None

    try:
        spy = yf.Ticker("SPY")
        expirations = spy.options[:n_expirations]
        if not expirations:
            log.warning("SPY options chain empty — skipping P/C")
            return None

        # Underlying price for sanity reference
        try:
            info = spy.history(period="1d")
            underlying_px = float(info["Close"].iloc[-1]) if not info.empty else float("nan")
        except Exception:
            underlying_px = float("nan")

        total_call_volume = 0.0
        total_put_volume = 0.0
        total_call_oi = 0.0
        total_put_oi = 0.0
        n_sampled = 0

        for exp in expirations:
            try:
                chain = spy.option_chain(exp)
            except Exception as exc:
                log.warning("P/C: failed to fetch chain for %s: %s", exp, exc)
                continue
            total_call_volume += float(chain.calls["volume"].fillna(0).sum())
            total_put_volume += float(chain.puts["volume"].fillna(0).sum())
            total_call_oi += float(chain.calls["openInterest"].fillna(0).sum())
            total_put_oi += float(chain.puts["openInterest"].fillna(0).sum())
            n_sampled += 1

        if n_sampled == 0:
            log.warning("P/C: no expirations sampled successfully")
            return None
        if total_call_volume <= 0 or total_call_oi <= 0:
            log.warning("P/C: zero call volume/OI — skip")
            return None

        return {
            "vol_pc": round(total_put_volume / total_call_volume, 4),
            "oi_pc": round(total_put_oi / total_call_oi, 4),
            "n_expirations": int(n_sampled),
            "underlying_px": round(underlying_px, 2),
        }
    except Exception as exc:
        log.exception("P/C computation failed: %s", exc)
        return None


def load_history() -> pd.DataFrame:
    """Return cumulative P/C history. Empty DataFrame if not yet started."""
    if PC_PATH.exists():
        df = pd.read_csv(PC_PATH)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df.sort_values("date").reset_index(drop=True)
    return pd.DataFrame(columns=["date", "vol_pc", "oi_pc", "n_expirations", "underlying_px"])


def append_today(today: date | None = None, *,
                 n_expirations: int = DEFAULT_N_EXPIRATIONS) -> dict | None:
    """Compute today's P/C and append to history (idempotent for the same date).

    If today's row already exists, it is OVERWRITTEN with the fresh value
    (handles intra-day reruns of the workflow).
    """
    if today is None:
        today = date.today()
    result = compute_spy_pc(n_expirations=n_expirations)
    if result is None:
        return None
    row = {"date": today.isoformat(), **result}

    PC_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PC_PATH.exists():
        existing = pd.read_csv(PC_PATH)
        existing["date"] = existing["date"].astype(str)
        existing = existing[existing["date"] != today.isoformat()]
    else:
        existing = pd.DataFrame(columns=["date", "vol_pc", "oi_pc", "n_expirations", "underlying_px"])

    if existing.empty:
        merged = pd.DataFrame([row])
    else:
        merged = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    merged = merged.sort_values("date").reset_index(drop=True)
    merged.to_csv(PC_PATH, index=False)
    log.info("P/C appended for %s: vol=%s  oi=%s  exp=%s  px=%s  (history=%d rows)",
             today, row["vol_pc"], row["oi_pc"], row["n_expirations"],
             row["underlying_px"], len(merged))
    return {**row, "date": today}


def latest_pc() -> dict | None:
    """Return the most recent (date, vol_pc, oi_pc) row, or None if empty."""
    hist = load_history()
    if hist.empty:
        return None
    last = hist.iloc[-1]
    return {
        "date": last["date"],
        "vol_pc": float(last["vol_pc"]),
        "oi_pc": float(last["oi_pc"]),
        "n_expirations": int(last["n_expirations"]),
    }


def status_summary() -> str:
    """Human-friendly description of how much history is accumulated."""
    hist = load_history()
    n = len(hist)
    if n == 0:
        return "P/C history empty — collection starts today"
    first, last = hist["date"].iloc[0], hist["date"].iloc[-1]
    days_back = (last - first).days
    ready_for_backtest = "✓ ready (≥60d)" if n >= 60 else f"⏳ need {60 - n} more days"
    return f"P/C history: {n} rows {first}..{last} ({days_back}d span) — {ready_for_backtest}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    row = append_today()
    if row is not None:
        log.info("Today's P/C: %s", row)
    log.info(status_summary())
