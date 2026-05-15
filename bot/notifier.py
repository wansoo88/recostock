"""Telegram one-way push notification.

Single message per day: summary text + GitHub Pages report link.
No command handling (no persistent process in GitHub Actions).
양방향 명령어는 OCI 상시 서버 이전 후 구현 예정.
"""

from __future__ import annotations

import logging
from datetime import date

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
