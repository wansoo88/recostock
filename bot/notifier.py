"""Telegram one-way push notification.

Daily batch: send_daily_signal (GitHub Actions)
Intraday:   send_intraday_alert, send_eod_close_alert (run_intraday.py local loop)
"""

from __future__ import annotations

import logging
from datetime import date, datetime

log = logging.getLogger(__name__)


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

    longs = [s.ticker for s in signals if s.direction == "long" and s.leverage == 1]
    inverses = [s.ticker for s in signals if s.direction == "short" and s.leverage == 1]
    lev_signals = [s for s in signals if s.leverage > 1]

    avg_exp = sum(s.expectancy for s in signals) / max(len(signals), 1) if signals else 0.0
    exp_str = f"{avg_exp:+.2%}"

    lines = [
        f"📊 [{report_date}] 일일 ETF 시그널",
        f"레짐: {regime.get('label', 'N/A')} / 노출도 {regime.get('exposure', 1.0):.2f}× "
        f"/ 시그널 {len(signals)}개 / 종합 기대값 {exp_str}",
    ]
    if longs:
        lines.append(f"🟢 LONG: {', '.join(longs)}")
    if inverses:
        lines.append(f"🔴 INVERSE: {', '.join(inverses)}")
    if lev_signals:
        lev_str = ", ".join(
            f"{s.ticker} (신뢰도 {s.confidence:.2f})" for s in lev_signals
        )
        lines.append(f"⚡ 레버리지: {lev_str}")
    if not signals:
        lines.append("오늘 유효 시그널 없음")

    if report_url:
        lines.append(f"상세(적중률·근거·팩터): {report_url}")
    lines.append("⚠️ 수동 실행 · 진입가 미충족 시 미진입")

    message = "\n".join(lines)
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
