"""HTML report builder.

Injects the REPORT JS object into the template and writes to docs/ for GitHub Pages.
The REPORT object shape is the data contract between this module and the template.
Also maintains docs/index.html — a stable bookmarkable URL that redirects to the
latest dated report (the Pages root used to 404).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path

from signals.generator import Signal

log = logging.getLogger(__name__)

TEMPLATE_PATH = Path("report/templates/daily-signal-report-template.html")
OUTPUT_DIR = Path("docs")  # GitHub Pages serves from docs/ on main branch

_DATED_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.html$")
_INDEX_RECENT_N = 15


def build_report(
    signals: list[Signal],
    regime: dict,
    report_date: date,
    paper_metrics: dict | None = None,
    paper_open: list[dict] | None = None,
    gates: list[dict] | None = None,
    strategy_info: dict | None = None,
) -> Path:
    """Write YYYY-MM-DD.html to docs/. Returns the output path.

    `gates` (new 2026-05-19): list of {name, value, threshold, op, passed} dicts
    for the v4 conviction regime gates. Rendered as a visual status panel so
    the user can see WHY today's signal fired (or didn't).

    `strategy_info` (new 2026-05-19): {version, description} for the strategy
    that produced today's signals. Shown next to the regime panel.
    """
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    valid_signals = [s for s in signals if s.is_valid()]
    report_obj = {
        "date": report_date.isoformat(),
        "regime": regime,
        "signals": [_signal_to_dict(s) for s in valid_signals],
        "paper": paper_metrics or {},
        "paperOpen": paper_open or [],
        "gates": gates or [],
        "strategy": strategy_info or {},
    }

    injected = template.replace(
        "/* REPORT_DATA_PLACEHOLDER */",
        f"const REPORT = {json.dumps(report_obj, ensure_ascii=False, indent=2)};",
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{report_date.isoformat()}.html"
    out_path.write_text(injected, encoding="utf-8")
    log.info("Report written: %s (%d valid signals)", out_path, len(valid_signals))

    try:
        write_index(OUTPUT_DIR)
    except Exception:  # index is a convenience — never fail the report for it
        log.warning("index.html generation failed (non-fatal)", exc_info=True)
    return out_path


def write_index(output_dir: Path = OUTPUT_DIR) -> Path | None:
    """Regenerate docs/index.html → instant redirect to the newest dated report.

    Gives the user one bookmarkable URL (the Pages root) instead of a new link
    every day; also lists recent reports for quick history access.
    """
    dated = sorted((p.name for p in output_dir.glob("*.html") if _DATED_RE.match(p.name)),
                   reverse=True)
    if not dated:
        return None
    latest = dated[0]
    links = "\n".join(
        f'    <li><a href="{name}">{name[:-5]}</a></li>' for name in dated[:_INDEX_RECENT_N]
    )
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="0; url={latest}">
<title>recostock — 최신 리포트로 이동</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0;
         max-width: 640px; margin: 48px auto; padding: 0 16px; }}
  a {{ color: #63b3ed; text-decoration: none; }}
  li {{ margin: 4px 0; }}
</style>
</head>
<body>
<p>최신 리포트로 이동 중… 자동 이동이 안 되면 <a href="{latest}">{latest[:-5]}</a>를 여세요.</p>
<ul>
{links}
</ul>
</body>
</html>
"""
    index_path = output_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    log.info("Index written: %s -> %s", index_path, latest)
    return index_path


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
