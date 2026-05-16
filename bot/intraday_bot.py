"""Intraday Telegram bot with two-way UI (python-telegram-bot v21).

Features:
  - 5-minute signal polling via JobQueue
  - Inline keyboard: [✅ 진입] [❌ 패스] on each signal change
  - /positions — open positions with [TP 달성][SL 달성][수동청산] buttons
  - /stats — today's realized winrate and P&L
  - /help — command list

Usage:
    TELEGRAM_BOT_TOKEN=xxx TELEGRAM_CHAT_ID=yyy python scripts/run_intraday.py
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

from data.intraday import (
    INTRADAY_TICKERS,
    fetch_5min_ohlcv,
    get_vix_level,
    is_eod_warn_window,
    is_market_open,
    now_et,
)
from signals.intraday_generator import (
    IntraSignal,
    diff_signals,
    generate_signals,
    top_signals,
)
from paper.intraday_tracker import (
    get_open_trades,
    get_stats,
    log_entry,
    log_exit,
    skip_trade,
)
from report.intraday_report import build_intraday_report

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

POLL_INTERVAL = 300  # seconds

# In-memory state shared across callbacks
_prev_signals: dict[str, IntraSignal] = {}
_eod_alerted: bool = False
_last_date: str | None = None


# ── Signal message builder ────────────────────────────────────────────────────

def _signal_row(sig: IntraSignal) -> str:
    dir_icon = "🟢" if sig.direction == 1 else "🔴"
    if sig.direction == 1:
        trade_desc = f"매수 {sig.ticker}  ${sig.price:.2f}"
    else:
        trade_desc = f"매수 {sig.action_ticker} (SHORT {sig.ticker})  ${sig.price:.2f}"
    tp_pct = abs(sig.tp - sig.price) / sig.price * 100
    sl_pct = abs(sig.sl - sig.price) / sig.price * 100
    adx_str = f"ADX {sig.adx:.0f}" if not (sig.adx != sig.adx) else ""  # skip NaN
    srsi_str = f"StochRSI {sig.stochrsi_k:.0f}" if not (sig.stochrsi_k != sig.stochrsi_k) else ""
    return (
        f"{dir_icon} {trade_desc}\n"
        f"   익절가: ${sig.tp:.2f} (+{tp_pct:.1f}%)  |  손절가: ${sig.sl:.2f} (-{sl_pct:.1f}%)\n"
        f"   RSI {sig.rsi:.1f}  {srsi_str}  {adx_str}  예상WR {sig.winrate:.0%}"
        f"  기대수익 {sig.exp_return:+.2%}"
    )


def _build_signal_message(
    longs: list[IntraSignal],
    shorts: list[IntraSignal],
    now: datetime,
    vix: float | None,
) -> str:
    time_str = now.strftime("%H:%M ET")
    regime = "normal"
    if vix:
        regime = "fear" if vix >= 30 else ("caution" if vix >= 20 else "normal")
    vix_str = f"VIX {vix:.1f} [{regime}]" if vix else ""

    parts = [f"[{time_str}] 인트라데이 TOP5 추천  {vix_str}",
             "⚠️ 데이터 15분 지연 · 예상WR/기대수익은 미검증 추정치"]
    if longs:
        parts.append("\nLONG 추천")
        for i, s in enumerate(longs, 1):
            parts.append(f"{i}. {_signal_row(s)}")
    if shorts:
        parts.append("\nSHORT 추천")
        for i, s in enumerate(shorts, 1):
            parts.append(f"{i}. {_signal_row(s)}")
    if not longs and not shorts:
        parts.append("현재 유효 시그널 없음 — FLAT")
    parts.append("\n수동 진입 · 3:45PM ET 이전 반드시 청산 · LOC 주문 권장")
    return "\n".join(parts)


def _signal_keyboard(longs: list[IntraSignal], shorts: list[IntraSignal]) -> InlineKeyboardMarkup:
    """One [진입] [패스] button pair per signal (max 5+5=10)."""
    rows = []
    for sig in longs + shorts:
        label = f"{'🟢' if sig.direction==1 else '🔴'} {sig.action_ticker} 진입 ${sig.price:.2f}"
        rows.append([
            InlineKeyboardButton(label, callback_data=f"b:{sig.ticker}"),
            InlineKeyboardButton("❌ 패스", callback_data=f"s:{sig.ticker}"),
        ])
    return InlineKeyboardMarkup(rows)


# ── Job: 5-minute signal poll ─────────────────────────────────────────────────

async def signal_poll_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    global _prev_signals, _eod_alerted, _last_date

    chat_id = context.bot_data.get("chat_id")
    if not chat_id:
        return

    n = now_et()
    today_str = n.strftime("%Y-%m-%d")

    # Reset daily state
    if _last_date != today_str:
        _prev_signals = {}
        _eod_alerted = False
        _last_date = today_str

    if not is_market_open():
        return

    # EOD force-close reminder
    if is_eod_warn_window() and not _eod_alerted:
        _eod_alerted = True
        await context.bot.send_message(
            chat_id=chat_id,
            text="⏰ [3:45 PM ET] 장 마감 15분 전\n"
                 "• LOC 주문 미설정 시 → 지금 시장가 청산\n"
                 "• LOC 주문 설정된 경우 → 4:00 PM ET 자동 체결 대기\n"
                 "• TP/SL 체결 확인 후 LOC 취소 여부 확인\n"
                 "/positions 에서 버튼으로 청산 기록하세요.",
        )
        return

    # Fetch + generate
    try:
        ohlcv = fetch_5min_ohlcv(INTRADAY_TICKERS + ["^VIX"])
        vix = get_vix_level(ohlcv)
        ohlcv.pop("^VIX", None)
        curr_signals = generate_signals(ohlcv, vix)
    except Exception as exc:
        log.warning("Signal poll error: %s", exc)
        return

    # Store for callback access
    context.bot_data["curr_signals"] = curr_signals
    context.bot_data["ohlcv"] = ohlcv
    context.bot_data["vix"] = vix

    # Check for changes
    changes = diff_signals(_prev_signals, curr_signals)
    flat_changes = [(p, c) for p, c in changes if c.direction == 0]

    if changes:
        longs, shorts = top_signals(curr_signals, n=5)
        msg = _build_signal_message(longs, shorts, n, vix)

        if flat_changes:
            flat_str = ", ".join(f"{c.ticker} FLAT (청산 고려)" for _, c in flat_changes)
            msg += f"\n\n⬜ 시그널 소멸: {flat_str}"

        keyboard = _signal_keyboard(longs, shorts)
        await context.bot.send_message(chat_id=chat_id, text=msg, reply_markup=keyboard)

        # Generate HTML report for signal evidence
        ohlcv = context.bot_data.get("ohlcv", {})
        if ohlcv:
            pages_url = context.bot_data.get("pages_url", "")
            try:
                report_path = build_intraday_report(curr_signals, ohlcv, vix, n)
                if pages_url:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"📊 상세 차트 (근거): {pages_url}/{report_path.name}",
                    )
            except Exception as exc:
                log.warning("Intraday report build failed: %s", exc)

    _prev_signals = curr_signals


# ── Callback handlers ─────────────────────────────────────────────────────────

async def handle_buy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """User clicked [진입] for a ticker."""
    query = update.callback_query
    await query.answer()

    ticker = query.data.split(":")[1]
    curr_signals: dict = context.bot_data.get("curr_signals", {})
    sig = curr_signals.get(ticker)

    if sig is None:
        await query.edit_message_text(f"⚠️ {ticker} 시그널이 만료됐습니다. 최신 시그널을 확인하세요.")
        return

    trade_id = log_entry(
        ticker=sig.ticker,
        action_ticker=sig.action_ticker,
        direction=sig.action_label,
        signal_price=sig.price,
        entry_price=sig.price,
        tp=sig.tp,
        sl=sig.sl,
    )

    tp_pct = abs(sig.tp - sig.price) / sig.price * 100
    sl_pct = abs(sig.sl - sig.price) / sig.price * 100

    # Order guide (LOC order recommendation)
    if sig.direction == 1:
        order_guide = (
            f"주문 방법 (토스증권):\n"
            f"  1. {sig.action_ticker} 시장가 매수 @ ${sig.price:.2f}\n"
            f"  2. 지정가 매도(TP): ${sig.tp:.2f} (+{tp_pct:.1f}%)\n"
            f"  3. 조건부 손절(SL): ${sig.sl:.2f} (-{sl_pct:.1f}%)\n"
            f"  4. LOC 매도 주문 설정 → 4:00 PM ET 자동 종가 청산\n"
            f"     (TP/SL 체결 시 LOC 수동 취소 필요)"
        )
    else:
        order_guide = (
            f"주문 방법 (토스증권):\n"
            f"  1. {sig.action_ticker}(인버스) 시장가 매수 @ ${sig.price:.2f}\n"
            f"  2. 지정가 매도(TP): ${sig.tp:.2f} (+{tp_pct:.1f}%)\n"
            f"  3. 조건부 손절(SL): ${sig.sl:.2f} (-{sl_pct:.1f}%)\n"
            f"  4. LOC 매도 주문 설정 → 4:00 PM ET 자동 종가 청산\n"
            f"     (TP/SL 체결 시 LOC 수동 취소 필요)"
        )

    text = (
        f"✅ {sig.action_ticker} 진입 기록 (거래 #{trade_id})\n"
        f"   진입가: ${sig.price:.2f}  TP: ${sig.tp:.2f}  SL: ${sig.sl:.2f}\n\n"
        f"{order_guide}\n\n"
        f"/positions 에서 청산 처리 가능"
    )
    await query.edit_message_text(text)


async def handle_skip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer("패스")
    ticker = query.data.split(":")[1]
    skip_trade(ticker)
    await query.edit_message_text(f"❌ {ticker} 패스 처리됨")


async def handle_exit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """User clicked exit button from /positions."""
    query = update.callback_query
    await query.answer()

    _, trade_id_str, reason = query.data.split(":")
    trade_id = int(trade_id_str)

    # Determine exit price from current signal or last known
    curr_signals = context.bot_data.get("curr_signals", {})
    open_trades = {t["id"]: t for t in get_open_trades()}
    trade = open_trades.get(trade_id)
    if trade is None:
        await query.edit_message_text("⚠️ 거래를 찾을 수 없습니다.")
        return

    ticker = trade["ticker"]
    sig = curr_signals.get(ticker)
    if reason == "TP":
        exit_price = trade["tp"]
        reason_label = "TP 달성 (익절)"
    elif reason == "SL":
        exit_price = trade["sl"]
        reason_label = "SL 히트 (손절)"
    elif reason == "EOD" and sig:
        exit_price = sig.price
        reason_label = "EOD 청산"
    elif sig:
        exit_price = sig.price
        reason_label = "수동 청산"
    else:
        exit_price = trade["tp"]  # fallback
        reason_label = reason

    pnl = log_exit(trade_id, exit_price, reason)
    if pnl is None:
        await query.edit_message_text("⚠️ 이미 청산된 거래입니다.")
        return

    pnl_icon = "🟢" if pnl > 0 else "🔴"
    await query.edit_message_text(
        f"{pnl_icon} [{trade['action_ticker']}] {reason_label}\n"
        f"   진입 ${trade['entry_price']:.2f}  →  청산 ${exit_price:.2f}\n"
        f"   손익: {pnl:+.2%}\n\n"
        f"/stats 로 오늘 종합 성과 확인"
    )


# ── Command handlers ──────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "인트라데이 ETF 시그널 봇 시작됨.\n\n"
        "/positions — 오픈 포지션\n"
        "/stats — 오늘 손익 현황\n"
        "/help — 도움말"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "명령어 목록:\n"
        "/positions — 현재 오픈 포지션 + 청산 버튼\n"
        "/stats — 오늘 실현 손익, 적중률\n\n"
        "시그널 알림은 변경 시에만 자동 발송됩니다.\n"
        "3:45 PM ET에 마감 알림이 발송됩니다.\n"
        "주문 방식: 시장가 진입 + 지정가 익절 + 조건부 손절"
    )


async def cmd_positions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    trades = get_open_trades()
    if not trades:
        await update.message.reply_text("현재 오픈 포지션 없음")
        return

    curr_signals = context.bot_data.get("curr_signals", {})
    lines = ["📋 오픈 포지션:"]
    keyboard_rows = []

    for t in trades:
        sig = curr_signals.get(t["ticker"])
        curr_price_str = f"현재 ${sig.price:.2f}" if sig else ""
        entry = t["entry_price"]
        tp, sl = t["tp"], t["sl"]
        if t["direction"] == "LONG":
            pnl_est = (sig.price - entry) / entry if sig else 0.0
        else:
            pnl_est = (entry - sig.price) / entry if sig else 0.0
        pnl_str = f"{pnl_est:+.2%}" if sig else "N/A"

        lines.append(
            f"• #{t['id']} {t['action_ticker']} {t['direction']}  "
            f"진입 ${entry:.2f}  {curr_price_str}  미실현 {pnl_str}\n"
            f"  TP ${tp:.2f}  SL ${sl:.2f}"
        )
        tid = t["id"]
        keyboard_rows.append([
            InlineKeyboardButton(f"#{tid} TP 달성", callback_data=f"x:{tid}:TP"),
            InlineKeyboardButton(f"SL 히트", callback_data=f"x:{tid}:SL"),
            InlineKeyboardButton(f"수동청산", callback_data=f"x:{tid}:MAN"),
        ])

    await update.message.reply_text(
        "\n".join(lines),
        reply_markup=InlineKeyboardMarkup(keyboard_rows),
    )


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    stats = get_stats()
    if stats["n"] == 0:
        await update.message.reply_text("오늘 청산된 거래 없음")
        return

    pnl_icon = "🟢" if stats["total_pnl"] > 0 else "🔴"
    await update.message.reply_text(
        f"📊 오늘 실현 성과\n"
        f"  청산 {stats['n']}건  적중률 {stats['winrate']:.1%}\n"
        f"  평균 손익: {stats['avg_pnl']:+.2%}\n"
        f"  {pnl_icon} 합산 손익: {stats['total_pnl']:+.2%}"
    )


# ── Application builder ───────────────────────────────────────────────────────

def build_app(bot_token: str, chat_id: str) -> Application:
    app = Application.builder().token(bot_token).build()
    app.bot_data["chat_id"] = chat_id
    app.bot_data["curr_signals"] = {}
    app.bot_data["ohlcv"] = {}
    app.bot_data["vix"] = None
    app.bot_data["pages_url"] = os.environ.get("PAGES_URL", "")

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("positions", cmd_positions))
    app.add_handler(CommandHandler("stats", cmd_stats))

    # Callbacks
    app.add_handler(CallbackQueryHandler(handle_buy, pattern=r"^b:"))
    app.add_handler(CallbackQueryHandler(handle_skip, pattern=r"^s:"))
    app.add_handler(CallbackQueryHandler(handle_exit, pattern=r"^x:"))

    # Job: poll every 5 minutes
    app.job_queue.run_repeating(signal_poll_job, interval=POLL_INTERVAL, first=10)

    return app
