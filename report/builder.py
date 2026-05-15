"""HTML report builder.

Injects the REPORT JS object into the template and writes to docs/ for GitHub Pages.
The REPORT object shape is the data contract between this module and the template.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from signals.generator import Signal

log = logging.getLogger(__name__)

TEMPLATE_PATH = Path("report/templates/daily-signal-report-template.html")
OUTPUT_DIR = Path("docs")  # GitHub Pages serves from docs/ on main branch


def build_report(signals: list[Signal], regime: dict, report_date: date) -> Path:
    """Write YYYY-MM-DD.html to docs/. Returns the output path."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    valid_signals = [s for s in signals if s.is_valid()]
    report_obj = {
        "date": report_date.isoformat(),
        "regime": regime,
        "signals": [_signal_to_dict(s) for s in valid_signals],
    }

    injected = template.replace(
        "/* REPORT_DATA_PLACEHOLDER */",
        f"const REPORT = {json.dumps(report_obj, ensure_ascii=False, indent=2)};",
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{report_date.isoformat()}.html"
    out_path.write_text(injected, encoding="utf-8")
    log.info("Report written: %s (%d valid signals)", out_path, len(valid_signals))
    return out_path


def _signal_to_dict(s: Signal) -> dict:
    """Data contract — any change here must be reflected in the HTML template."""
    return {
        "ticker": s.ticker,
        "name": s.name,
        "dir": s.direction,
        "leverage": s.leverage,
        "entry": s.entry,
        "tp": s.tp,
        "sl": s.sl,
        "winrate": round(s.winrate, 4),
        "sampleN": s.sample_n,
        "ci": [round(s.ci_low, 4), round(s.ci_high, 4)],
        "payoff": round(s.payoff, 3),
        "expectancy": round(s.expectancy, 4),
        "confidence": round(s.confidence, 3),
        "factors": {k: round(v, 4) for k, v in s.factors.items()},
    }
