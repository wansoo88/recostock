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
    # Stale-data warning — lead with it so the user can't miss frozen prices.
    if regime.get("stale"):
        lines.insert(1, f"⚠️ 데이터 지연: 최신 종가 {regime.get('dataAsOf')} "
                        f"({regime.get('staleDays')}일 전) — 시그널 신뢰 말고 파이프라인 확인")
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

    # Sector/core RSI monitor (top 5 by RSI-14). Replaces the old confidence-ranked
    # "TOP PICK" block (removed 2026-05-31): it ranked by model confidence — which
    # has ~zero cross-sectional skill (Spearman -0.02) — AND referenced tp/sl keys
    # that no longer exist in the candidate dict (latent KeyError when a name passed
    # the gate). The actionable RSI sectors are already in the 🛰️ and 📐 lines below;
    # this is a context watchlist only, matching the report's RSI monitor panel.
    candidates = regime.get("candidates") or []
    if candidates:
        top5 = candidates[:5]  # candidates are RSI-sorted in run_daily
        wl = ", ".join(
            f"{c['ticker']} {c['rsi']:.0f}" for c in top5
            if c.get("rsi") is not None
        )
        if wl:
            lines.append(f"📋 섹터·코어 RSI순: {wl} (관전용 · 실행은 아래 포지션)")

    # RSI-14 sector-rotation satellite (validated 2026-05-30) — optional layer.
    sat = regime.get("sectorSatellite") or {}
    if sat.get("ranked"):
        pick = sat.get("pick") or []
        wpct = int(round((sat.get("weight", 0.25)) * 100))
        if pick:
            extra = f" (+현금 {sat['cashHalf']}칸)" if sat.get("cashHalf") else ""
            lines.append(f"🛰️ RSI 섹터 로테이션(선택): 이번 주 {' + '.join(pick)}{extra} "
                         f"— 추세코어 자본의 ~{wpct}%까지 권장")
        else:
            lines.append("🛰️ RSI 섹터 로테이션(선택): 상위 섹터 모두 200일선 아래 → 전량 현금")

    # Fear-dip experimental signal (paper-only). Surface the entry on a signal
    # day AND the accumulating paper track so the user sees it build over time.
    # Portfolio NAV paper validation (3-month Tier-2 track) — reporting only.
    pp = regime.get("portfolioPaper") or {}
    if pp.get("nDays", 0) > 0:
        gap_s = f"{pp['gap']*100:.0f}%" if pp.get("gap") is not None else "n/a"
        lines.append(
            f"🧪 3개월 페이퍼 검증(주력 블렌드): {pp['nDays']}일/{pp['months']:.1f}개월 · "
            f"NAV {pp['totalReturn']*100:+.1f}% · 실현Sharpe {pp['annSharpe']:.2f}"
            f"(목표 {pp['targetSharpe']:.2f}, 괴리 {gap_s}) · {pp['status']}"
        )

    fd = regime.get("fearDip")
    if fd:
        if fd.get("isEntry"):
            pct = fd.get("percentile")
            pct_s = f"{pct*100:.0f}%" if pct is not None else "—"
            lines.append(f"🧪 [실험·페이퍼] 공포매수 진입신호 — SPY (공포 백분위 {pct_s}, 10일 보유). 실전 아님")
        pm = fd.get("paper") or {}
        n_closed, n_open = pm.get("n", 0), pm.get("open", 0)
        if n_closed > 0 or n_open > 0:
            track = (f"🧪 공포매수 페이퍼 트랙: 청산 {n_closed}건 · 승률 {pm.get('winrate',0)*100:.0f}% "
                     f"· 누적 {pm.get('total',0)*100:+.1f}% · 보유 {n_open}건")
            if n_closed >= 5:
                track += "  ← 표본 누적, 검토 시점"
            lines.append(track)

    # Trend-core position (primary engine) — lead the actionable section.
    # When the RSI sector sleeve is blended in, the composed portfolio is the
    # headline allocation; otherwise fall back to the pure engine weights.
    tc = regime.get("trendCore")
    pf = regime.get("portfolio") or {}
    if tc and tc.get("coreOn") is not None:
        boost = " ⚡캄-불 부스트" if tc.get("calmBoost") else ""
        if pf.get("enabled") and pf.get("weights"):
            lev = lambda t: " (3x)" if t in ("SPXL", "TQQQ") else ""
            parts = [f"{t} {w*100:.0f}%{lev(t)}"
                     for t, w in sorted(pf["weights"].items(), key=lambda kv: -kv[1]) if w > 0.001]
            if pf.get("cashWeight", 0) > 0.001:
                parts.append(f"현금/BIL {pf['cashWeight']*100:.0f}%")
            alloc = " + ".join(parts) if parts else "현금 100%"
            lines.append(f"📐 오늘의 포지션(주력 블렌드): {alloc} "
                         f"(≈{pf.get('effExposure',0):.2f}x = 시장 베타 환산){boost}")
            lines.append(f"   └ 추세코어 {pf.get('coreWeight',0.75)*100:.0f}% + RSI 섹터 슬리브 {pf.get('sleeveWeight',0.25)*100:.0f}%")
        else:
            parts = []
            if tc.get("spyWeight", 0) > 0:  parts.append(f"SPY {tc['spyWeight']*100:.0f}%")
            if tc.get("spxlWeight", 0) > 0: parts.append(f"SPXL {tc['spxlWeight']*100:.0f}%")
            if tc.get("qqqWeight", 0) > 0:  parts.append(f"QQQ {tc['qqqWeight']*100:.0f}%")
            if tc.get("cashWeight", 0) > 0: parts.append(f"현금/BIL {tc['cashWeight']*100:.0f}%")
            alloc = " + ".join(parts) if parts else "현금 100%"
            lines.append(f"📐 오늘의 포지션(주력): {alloc} (≈{tc.get('effExposure',0):.2f}x = 시장 베타 환산){boost}")
        # Per-leg prices and stop-loss levels for execution
        ex = tc.get("exec") or {}
        if ex.get("spy"):
            lines.append(f"   └ SPY 진입 ${ex['spy']['price']:.2f} · 추세 손절선 ${ex['spy']['stop']:.2f} (이 가격 하향이탈 시 청산)")
        if ex.get("spxl"):
            d = ex.get("tiltDaysLeft")
            extra = f"틸트 만료 ~{d}일" if d is not None else "공포매수 트리거 만료 시 청산"
            lines.append(f"   └ SPXL 진입 ${ex['spxl']['price']:.2f} · {extra}")
        if ex.get("qqq"):
            lines.append(f"   └ QQQ 진입 ${ex['qqq']['price']:.2f} · 추세 손절선 ${ex['qqq']['stop']:.2f}")

    # Reference signal — conviction is already reflected in the blend's RSI sector
    # sleeve, so this is confirmation context, NOT a separate trade instruction
    # (reframed 2026-05-31; the old "reduce SPY 8%, rotate" double-counted the sleeve).
    ens = regime.get("ensemble")
    if ens:
        if ens.get("action") == "conviction":
            lines.append(f"📋 참고: conviction 발사({ens['ticker']}) — 이미 위 블렌드의 RSI 섹터 슬리브에 반영됨 "
                         f"(별도 매매 불필요, 같은 방향 확인용).")
        elif ens.get("action") == "feardip":
            # fear-dip already in core SPXL tilt; here = OOS validation only
            pass  # silenced from telegram, surfaced in 🧪 paper line below
        # 'none' case: no satellite line — core position is the action.

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
