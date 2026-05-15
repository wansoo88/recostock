"""Intraday HTML report builder.

Generates docs/intraday-YYYY-MM-DD.html on each signal change.
Shows 5-minute price chart (EMA5, EMA20, VWAP) + RSI + signal table.
Non-expert friendly: plain Korean explanations, colour-coded verdicts.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from features.intraday_factors import compute_intraday_features
from signals.intraday_generator import IntraSignal

ET = ZoneInfo("America/New_York")
DOCS_DIR = Path("docs")

_VERDICT_KO = {1: "LONG 매수", -1: "SHORT 인버스 매수", 0: "관망 (FLAT)"}
_VERDICT_COLOR = {1: "#16a34a", -1: "#dc2626", 0: "#6b7280"}
_REASON_LONG = [
    ("EMA5 > EMA20", "단기 이동평균이 장기를 돌파 — 상승 추세 확인"),
    ("가격 > VWAP", "거래량 가중 평균가 위에 위치 — 매수세 우세"),
    ("RSI > 52", "모멘텀 지표가 중립선 위 — 상승 탄력 있음"),
]
_REASON_SHORT = [
    ("EMA5 < EMA20", "단기 이동평균이 장기 아래 — 하락 추세 확인"),
    ("가격 < VWAP", "거래량 가중 평균가 아래 위치 — 매도세 우세"),
    ("RSI < 48", "모멘텀 지표가 중립선 아래 — 하락 탄력 있음"),
]


def _make_chart_data(df: pd.DataFrame) -> dict:
    feat = compute_intraday_features(df)
    times = [t.strftime("%H:%M") for t in feat.index.to_pydatetime()]
    return {
        "labels": times,
        "close": [round(float(v), 4) for v in feat["Close"]],
        "ema5":  [round(float(v), 4) for v in feat["ema5"]],
        "ema20": [round(float(v), 4) for v in feat["ema20"]],
        "vwap":  [round(float(v), 4) for v in feat["vwap"]],
        "rsi":   [round(float(v), 2) for v in feat["rsi14"]],
    }


def _signal_card(sig: IntraSignal, chart_json: str) -> str:
    color = _VERDICT_COLOR[sig.direction]
    verdict = _VERDICT_KO[sig.direction]
    reasons = _REASON_LONG if sig.direction == 1 else (_REASON_SHORT if sig.direction == -1 else [])
    chart_id = f"chart_{sig.ticker}"
    rsi_id = f"rsi_{sig.ticker}"

    tp_pct = abs(sig.tp - sig.price) / sig.price * 100 if sig.tp else 0
    sl_pct = abs(sig.sl - sig.price) / sig.price * 100 if sig.sl else 0

    reason_rows = "".join(
        f'<tr><td class="reason-label">{r[0]}</td>'
        f'<td class="checkmark">&#10003;</td>'
        f'<td>{r[1]}</td></tr>'
        for r in reasons
    ) if reasons else '<tr><td colspan="3">시그널 조건 미충족 — 관망</td></tr>'

    return f"""
<div class="signal-card" style="border-left:4px solid {color}">
  <div class="card-header">
    <span class="ticker">{sig.ticker}</span>
    <span class="verdict" style="color:{color}">{verdict}</span>
    {"<span class='action-ticker'>→ " + sig.action_ticker + " 매수</span>" if sig.action_ticker != sig.ticker else ""}
  </div>

  <div class="price-grid">
    <div class="price-item"><span class="label">현재가</span><span class="value">${sig.price:.2f}</span></div>
    <div class="price-item"><span class="label">VWAP</span><span class="value">${sig.vwap:.2f}</span></div>
    <div class="price-item"><span class="label">RSI(14)</span><span class="value">{sig.rsi:.1f}</span></div>
    <div class="price-item"><span class="label">EMA5</span><span class="value">${sig.ema5:.2f}</span></div>
    <div class="price-item"><span class="label">EMA20</span><span class="value">${sig.ema20:.2f}</span></div>
    <div class="price-item"><span class="label">ATR(5분)</span><span class="value">${sig.atr:.3f}</span></div>
    <div class="price-item tp"><span class="label">익절가 (TP)</span><span class="value">${sig.tp:.2f} <small>(+{tp_pct:.1f}%)</small></span></div>
    <div class="price-item sl"><span class="label">손절가 (SL)</span><span class="value">${sig.sl:.2f} <small>(-{sl_pct:.1f}%)</small></span></div>
    <div class="price-item"><span class="label">예상 적중률</span><span class="value">{sig.winrate:.0%}</span></div>
    <div class="price-item"><span class="label">순 기대수익</span><span class="value">{sig.exp_return:+.2%}</span></div>
  </div>

  <h4>시그널 근거</h4>
  <table class="reason-table">
    <thead><tr><th>조건</th><th></th><th>설명</th></tr></thead>
    <tbody>{reason_rows}</tbody>
  </table>

  <div class="chart-wrap">
    <canvas id="{chart_id}" height="140"></canvas>
    <canvas id="{rsi_id}" height="60"></canvas>
  </div>
</div>
<script>
(function(){{
  var d = {chart_json};
  new Chart(document.getElementById('{chart_id}'), {{
    type:'line', data:{{
      labels: d.labels,
      datasets:[
        {{label:'종가', data:d.close, borderColor:'#1d4ed8', borderWidth:1.5, pointRadius:0, tension:0.2}},
        {{label:'EMA5', data:d.ema5, borderColor:'#f59e0b', borderWidth:1, pointRadius:0, borderDash:[]}},
        {{label:'EMA20', data:d.ema20, borderColor:'#ef4444', borderWidth:1, pointRadius:0, borderDash:[4,2]}},
        {{label:'VWAP', data:d.vwap, borderColor:'#8b5cf6', borderWidth:1, pointRadius:0, borderDash:[2,2]}},
      ]
    }},
    options:{{responsive:true, plugins:{{legend:{{position:'bottom'}}}},
      scales:{{x:{{ticks:{{maxTicksLimit:8}}}}, y:{{}}}}}}
  }});
  new Chart(document.getElementById('{rsi_id}'), {{
    type:'line', data:{{
      labels: d.labels,
      datasets:[{{label:'RSI(14)', data:d.rsi, borderColor:'#0891b2', borderWidth:1.5, pointRadius:0, fill:false}}]
    }},
    options:{{responsive:true, plugins:{{legend:{{position:'bottom'}}}},
      scales:{{x:{{ticks:{{maxTicksLimit:8}}}}, y:{{min:0,max:100,
        grid:{{color: ctx => (ctx.tick.value===70||ctx.tick.value===30)?'rgba(239,68,68,0.4)':'rgba(0,0,0,0.05)'}}
      }}}}}}
  }});
}})();
</script>
"""


_HTML_SHELL = """\
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>인트라데이 시그널 {date_str}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f8fafc;color:#1e293b;padding:16px}}
h1{{font-size:1.3rem;margin-bottom:4px}}
.meta{{font-size:.85rem;color:#64748b;margin-bottom:20px}}
.signal-card{{background:#fff;border-radius:10px;padding:20px;margin-bottom:24px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.card-header{{display:flex;align-items:center;gap:12px;margin-bottom:16px}}
.ticker{{font-size:1.5rem;font-weight:700}}
.verdict{{font-size:1rem;font-weight:600}}
.action-ticker{{font-size:.85rem;background:#f1f5f9;padding:2px 8px;border-radius:4px}}
.price-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;margin-bottom:16px}}
.price-item{{background:#f8fafc;border-radius:6px;padding:8px 12px}}
.price-item.tp{{background:#dcfce7}}
.price-item.sl{{background:#fee2e2}}
.label{{font-size:.75rem;color:#64748b;display:block}}
.value{{font-size:1rem;font-weight:600}}
h4{{margin:12px 0 8px;font-size:.9rem;color:#475569}}
.reason-table{{width:100%;border-collapse:collapse;font-size:.85rem;margin-bottom:16px}}
.reason-table th{{text-align:left;padding:6px 8px;background:#f1f5f9;font-size:.8rem}}
.reason-table td{{padding:6px 8px;border-bottom:1px solid #f1f5f9}}
.reason-label{{font-weight:600;white-space:nowrap}}
.checkmark{{color:#16a34a;font-size:1.1rem;text-align:center;width:30px}}
.chart-wrap{{margin-top:16px}}
.order-box{{background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:12px;margin-bottom:16px;font-size:.85rem}}
.order-box h4{{color:#9a3412;margin-bottom:6px}}
.order-box ol{{padding-left:20px;line-height:1.8}}
.no-signal{{background:#fff;border-radius:10px;padding:20px;color:#6b7280;text-align:center;margin-bottom:16px}}
</style>
</head>
<body>
<h1>인트라데이 ETF 시그널</h1>
<div class="meta">{datetime_str}  |  VIX {vix_str}  |  레짐: {regime}</div>

{order_guide_html}

{cards_html}

{no_signal_html}

<div style="margin-top:24px;font-size:.75rem;color:#94a3b8">
자동 생성 · 수동 실행 전용 · 왕복 비용 0.25% 포함 기대수익만 추천
</div>
</body>
</html>
"""

_ORDER_GUIDE = """\
<div class="order-box">
  <h4>주문 방법 (토스증권)</h4>
  <ol>
    <li>시그널 확인 후 <b>시장가 매수</b> (빠른 체결 우선)</li>
    <li>매수 직후 <b>지정가 매도</b> 설정 — 익절가(TP) 입력</li>
    <li><b>조건부 주문</b>으로 손절가(SL) 설정</li>
    <li>TP/SL 미체결 시 <b>3:45 PM ET 이전 시장가 청산</b> (당일 의무 청산)</li>
  </ol>
  <p style="margin-top:8px;color:#6b7280">LOC(Limit on Close) 주문은 토스증권 미지원 → 3:45 PM ET 수동 시장가 청산으로 대체</p>
</div>
"""


def build_intraday_report(
    signals: dict[str, IntraSignal],
    ohlcv: dict[str, pd.DataFrame],
    vix: float | None,
    now: datetime | None = None,
) -> Path:
    """Generate and save intraday HTML report. Returns the path."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    if now is None:
        now = datetime.now(ET)

    date_str = now.strftime("%Y-%m-%d")
    datetime_str = now.strftime("%Y-%m-%d %H:%M ET")
    vix_str = f"{vix:.1f}" if vix else "N/A"
    regime = "normal"
    if vix:
        regime = "공포 (fear)" if vix >= 30 else ("주의 (caution)" if vix >= 20 else "정상 (normal)")

    # Sort: longs first (by score desc), then shorts, then flats
    active = [s for s in signals.values() if s.direction != 0]
    active.sort(key=lambda s: -s.score)

    cards = []
    for sig in active:
        df = ohlcv.get(sig.ticker)
        if df is not None and len(df) >= 5:
            chart_data = _make_chart_data(df)
            chart_json = json.dumps(chart_data)
        else:
            chart_json = json.dumps({"labels": [], "close": [], "ema5": [], "ema20": [], "vwap": [], "rsi": []})
        cards.append(_signal_card(sig, chart_json))

    cards_html = "\n".join(cards)
    no_signal_html = '<div class="no-signal">현재 유효 시그널 없음 — 관망</div>' if not active else ""
    order_html = _ORDER_GUIDE if active else ""

    html = _HTML_SHELL.format(
        date_str=date_str,
        datetime_str=datetime_str,
        vix_str=vix_str,
        regime=regime,
        order_guide_html=order_html,
        cards_html=cards_html,
        no_signal_html=no_signal_html,
    )

    out_path = DOCS_DIR / f"intraday-{date_str}.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path
