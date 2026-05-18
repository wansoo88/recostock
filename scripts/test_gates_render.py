"""Dry-run: render the v4 daily report with gates panel using mock data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import date
from signals.generator import Signal
from report.builder import build_report

regime = {
    "label": "normal",
    "exposure": 1.0,
    "vix": 18.43,
    "credit_spread": 3.2,
    "yield_spread": 0.05,
    "gates": [
        {"name": "VIX 절댓값", "value": 18.43, "threshold": 20.0, "op": "<", "passed": True,
         "format": ".2f", "desc": "공포지수 (≥20 = 패닉)"},
        {"name": "SPY 추세", "value": 739.17, "threshold": 673.34, "op": ">", "passed": True,
         "format": ".2f", "desc": "200일 이동평균 대비"},
        {"name": "VIX 단기구조", "value": 0.888, "threshold": 1.0, "op": "<", "passed": True,
         "format": ".3f", "desc": "VIX9D/VIX (≥1.0 = backwardation)"},
        {"name": "SKEW z-score", "value": 0.381, "threshold": 1.0, "op": "<", "passed": True,
         "format": "+.2f", "desc": "60일 SKEW 표준화"},
        {"name": "MOVE z-score", "value": 0.133, "threshold": 1.0, "op": "<", "passed": True,
         "format": "+.2f", "desc": "60일 채권변동성"},
    ],
    "strategy": {
        "version": "conviction_v4",
        "description": "K=1 + Multi-EMA + 5-gate regime + 고정 SL 1.0%/TP 3.0%",
        "holdout_wr": 0.7368,
    },
}

sig = Signal(
    ticker="QQQ", name="Invesco QQQ", direction="long", leverage=1,
    entry=708.93, tp=730.20, sl=701.84,
    winrate=0.737, sample_n=19,
    ci_low=0.50, ci_high=0.90,
    payoff=1.25, expectancy=0.00033,
    confidence=0.71,
)

path = build_report(
    [sig], regime, date(2026, 5, 19),
    paper_metrics={"sharpe": 0.0, "mdd": 0.0, "winrate": 0.0, "n_trades": 0,
                   "avg_win": 0.0, "avg_loss": 0.0, "payoff": 0.0,
                   "total_return": 0.0, "n_weeks": 0, "weeks_elapsed": 0},
)
print(f"Report: {path}  ({path.stat().st_size} bytes)")
content = path.read_text(encoding="utf-8")
checks = [
    ("'gates-panel' div", "gates-panel" in content),
    ("'gates' field in REPORT JSON", '"gates":' in content),
    ("'conviction_v4' shown", "conviction_v4" in content),
    ("'Regime Gates' label", "Regime Gates" in content),
    ("'ALL 5 GATES PASS'", "GATES PASS" in content),
]
for label, ok in checks:
    print(f"  [{'✓' if ok else '✗'}] {label}")
