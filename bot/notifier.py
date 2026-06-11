"""Telegram one-way push notification.

Daily batch: send_daily_signal (GitHub Actions)
Intraday:   send_intraday_alert, send_eod_close_alert (run_intraday.py local loop)

send_daily_signal (rewritten 2026-06-11, decision-first):
  1. Header   — date + data as-of; loud stale-data warning when frozen.
  2. 오늘 할 일 — the single decision (hold / rebalance trades / all-cash),
                diffed against the user's current holdings by signals/decision.
  3. 목표 포트폴리오 — one allocation line + stop levels.
  4. 근거     — why-bullets (trend / VIX / sleeve), from the decision object.
  5. ⚡ 참고   — conviction satellite signal, only when one actually fired.
  6. 🧪 페이퍼 — one-line validation progress (detail lives in the report).
  7. Footer   — report link + manual-execution disclaimer.

Removed vs the old layout (the "중구난방" cleanup): the regime/exposure/signal-
count/expectancy header (three different exposure concepts in one message), the
RSI watchlist line, the satellite "추천 ~15%" line that duplicated sectors
already inside the blend weights, and the flat ~57% calibrated-probability
noise. One message, one instruction.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

log = logging.getLogger(__name__)

_WEEKDAY_KO = ["월", "화", "수", "목", "금", "토", "일"]
_PAPER_STATUS_KO = {
    "warming up": "검증 초기 — 수치는 아직 노이즈",
    "review": "점검 필요",
    "PASS": "게이트 통과",
}


def build_daily_message(
    signals: list,  # list[Signal] — avoid circular import
    regime: dict,
    report_url: str,
    report_date: date,
) -> str:
    """Pure message builder — separated from sending for testability."""
    from signals.decision import format_target_line

    lines = [f"📊 recostock 데일리 · {report_date} ({_WEEKDAY_KO[report_date.weekday()]})"]

    # Stale-data warning leads — the user must never trade on frozen prices.
    if regime.get("stale"):
        lines.append(f"⚠️ 데이터 지연: 최신 종가 {regime.get('dataAsOf')} "
                     f"({regime.get('staleDays')}일 전) — 오늘 지시를 따르지 말고 파이프라인 확인")
    elif regime.get("dataAsOf"):
        lines.append(f"데이터: {regime['dataAsOf']} 종가 기준")

    d = regime.get("decision") or {}
    tc = regime.get("trendCore") or {}
    pf = regime.get("portfolio") or {}

    # ── 1. 오늘 할 일 — the single instruction ────────────────────────────────
    if d:
        icon = {"hold": "✅", "rebalance": "🔄", "all_cash": "🚨", "start": "🆕"}.get(d.get("stance"), "📌")
        lines += ["", f"{icon} {d.get('headline', '')}"]
        for t in d.get("trades", []):
            lev = " (3x)" if t["ticker"] in ("SPXL", "TQQQ") else ""
            lines.append(f"   • {t['ticker']}{lev}  {t['fromPct']:g}% → {t['toPct']:g}%  ({t['action']})")

    # ── 2. 목표 포트폴리오 + 손절선 ───────────────────────────────────────────
    target_line = eff = None
    if d:
        target_line, eff = format_target_line(d), d.get("effExposure")
    elif pf.get("weights"):
        # Fallback when the decision step failed — still show the target.
        parts = [f"{t} {w*100:.0f}%{' (3x)' if t in ('SPXL', 'TQQQ') else ''}"
                 for t, w in sorted(pf["weights"].items(), key=lambda kv: -kv[1]) if w > 0.001]
        if pf.get("cashWeight", 0) > 0.001:
            parts.append(f"현금/BIL {pf['cashWeight']*100:.0f}%")
        target_line, eff = (" · ".join(parts) or "현금 100%"), pf.get("effExposure")

    if target_line:
        eff_s = f" — 시장노출 ≈{eff:.2f}x" if isinstance(eff, (int, float)) else ""
        lines += ["", f"📐 목표 포트폴리오{eff_s}", f"   {target_line}"]
        ex = tc.get("exec") or {}
        stops = []
        if ex.get("spy"):
            stops.append(f"SPY 종가 < ${ex['spy']['stop']:.2f} → SPY·SPXL 청산")
        if ex.get("qqq"):
            stops.append(f"QQQ 종가 < ${ex['qqq']['stop']:.2f} → QQQ 청산")
        if stops:
            lines.append("   └ 손절: " + " / ".join(stops))
        if ex.get("tiltDaysLeft") is not None:
            lines.append(f"   └ SPXL 공포 틸트 만료까지 ~{ex['tiltDaysLeft']}일")
    elif not signals:
        lines += ["", "오늘 유효 포지션·시그널 없음"]

    # ── 3. 근거 ───────────────────────────────────────────────────────────────
    if d.get("why"):
        lines += ["", "💡 근거"]
        lines += [f"   • {w}" for w in d["why"]]

    # ── 4. conviction 새틀라이트 — 실제 발사 시에만 ───────────────────────────
    for s in signals or []:
        lines += ["", (f"⚡ 참고 — conviction 신호: {s.ticker} 진입 ${s.entry:.2f} · "
                       f"TP ${s.tp:.2f} · SL ${s.sl:.2f} "
                       f"(홀드아웃 WR {s.winrate*100:.0f}%·n={s.sample_n} 소표본) "
                       f"— 새틀라이트(참고용), 위 블렌드와 별개")]

    # ── 5. 페이퍼 검증 — 한 줄 (상세는 리포트) ────────────────────────────────
    pp = regime.get("portfolioPaper") or {}
    if pp.get("nDays", 0) > 0:
        status = _PAPER_STATUS_KO.get(pp.get("status"), pp.get("status", ""))
        lines += ["", (f"🧪 페이퍼 검증(실자본 아님) {pp['nDays']}일/3개월 · "
                       f"NAV {pp['totalReturn']*100:+.1f}% · Sharpe {pp['annSharpe']:.2f} "
                       f"(목표 {pp['targetSharpe']:.2f}) · {status}")]

    # ── 6. Footer ─────────────────────────────────────────────────────────────
    lines.append("")
    if report_url:
        lines.append(f"🔗 상세 리포트: {report_url}")
    lines.append("⚠️ 모든 매매는 본인이 수동 실행 · 참고용이며 투자 권유 아님")

    return "\n".join(lines)


async def send_daily_signal(
    bot_token: str,
    chat_id: str,
    signals: list,  # list[Signal] — avoid circular import
    regime: dict,
    report_url: str,
    report_date: date,
) -> None:
    try:
        import telegram
    except ImportError:
        log.error("python-telegram-bot not installed — run: pip install python-telegram-bot")
        return

    bot = telegram.Bot(token=bot_token)
    message = build_daily_message(signals, regime, report_url, report_date)
    await bot.send_message(chat_id=chat_id, text=message)
    log.info("Telegram sent for %s", report_date)


async def send_intraday_alert(
    bot_token: str,
    chat_id: str,
    changes: list,   # list[tuple[IntraSignal | None, IntraSignal]]
    now: datetime,
) -> None:
    """Send Telegram alert when intraday signal direction changes."""
    try:
        import telegram
    except ImportError:
        log.error("python-telegram-bot not installed")
        return

    bot = telegram.Bot(token=bot_token)
    time_str = now.strftime("%H:%M ET")

    lines = [f"[{time_str}] 인트라데이 시그널 변경"]
    for prev_sig, curr_sig in changes:
        prev_label = prev_sig.action_label if prev_sig else "FLAT"
        arrow = "→"
        regime_note = f"  VIX={curr_sig.vix:.1f} [{curr_sig.regime}]" if curr_sig.vix else ""

        if curr_sig.direction == 1:
            icon = "🟢"
            action = f"{curr_sig.ticker} 매수"
        elif curr_sig.direction == -1:
            icon = "🔴"
            action = f"{curr_sig.action_ticker} 매수 ({curr_sig.ticker} SHORT 포지션)"
        else:
            icon = "⬜"
            action = f"{curr_sig.ticker} 청산 (FLAT)"

        lines.append(
            f"{icon} {curr_sig.ticker}: {prev_label} {arrow} {curr_sig.action_label}"
            f"  |  현재가 ${curr_sig.price}  VWAP ${curr_sig.vwap}"
            f"  RSI {curr_sig.rsi:.1f}"
        )
        if curr_sig.direction != 0:
            lines.append(f"   → {action}")
        if regime_note and curr_sig.direction != 0:
            lines.append(f"  {regime_note}")

    lines.append("⚠️ 수동 진입 · 3:45PM ET 전 반드시 청산")
    await bot.send_message(chat_id=chat_id, text="\n".join(lines))
    log.info("Intraday alert sent: %d change(s)", len(changes))


async def send_eod_close_alert(bot_token: str, chat_id: str) -> None:
    """3:45 PM ET force-close reminder."""
    try:
        import telegram
    except ImportError:
        return
    bot = telegram.Bot(token=bot_token)
    await bot.send_message(
        chat_id=chat_id,
        text="⏰ [3:45 PM ET] 장 마감 15분 전 — 당일 보유 포지션 전량 청산하세요.",
    )
    log.info("EOD close alert sent")
