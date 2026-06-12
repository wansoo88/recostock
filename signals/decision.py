"""Single daily decision — the ONE answer to "그래서 오늘 뭘 하면 되나?".

The report/telegram used to surface several parallel recommendation surfaces
(blend weights, an RSI-satellite suggestion line, a confidence watchlist where
every name showed the same flat ~57% calibrated probability, conviction notes)
that the user had to reconcile manually. This module reduces them to one
decision: diff today's target blend against what the user actually holds
(the previous portfolio-tracker record) and emit concrete trades plus the
reasons, in plain language.

Pure function: dicts in, decision dict out. No I/O — the caller supplies the
previous holdings (paper.portfolio_tracker.last_weights_before).

Output contract (consumed by bot/notifier.py and the HTML template):
    stance        'hold' | 'rebalance' | 'all_cash' | 'start'
    headline      one-line Korean instruction
    trades        [{ticker, action, fromPct, toPct, deltaPct}] sorted by |delta|
    nTrades       len(trades)
    targetWeights {ticker: capital fraction}   cashWeight, effExposure
    why           list[str] — reasons in priority order
    prevDate      ISO date of the holdings being diffed against (None on start)
"""
from __future__ import annotations

# Weight drifts below this are rounding noise, not a trade instruction.
REBALANCE_MIN_DELTA = 0.005
_LEV_TICKERS = {"SPXL", "TQQQ"}
# Warn when a sleeve's close is within this % of its trend stop — gives the
# manual executor advance notice that an all-cash flip may be near, instead
# of a surprise exit instruction one morning.
STOP_WARN_PCT = 3.0


def _fmt_pct(w: float) -> str:
    pct = w * 100
    return f"{pct:.1f}%" if abs(round(pct) - pct) > 0.05 else f"{pct:.0f}%"


def _action_label(frm: float, to: float) -> str:
    if frm <= REBALANCE_MIN_DELTA:
        return "신규 매수"
    if to <= REBALANCE_MIN_DELTA:
        return "전량 매도"
    return "증액" if to > frm else "감액"


def build_decision(portfolio: dict, trend_core: dict | None,
                   prev: dict | None = None,
                   satellite: dict | None = None,
                   fear_dip: dict | None = None,
                   vix: float | None = None,
                   prices: dict | None = None) -> dict:
    """Compose today's single decision from the blend target + prior holdings.

    portfolio  : signals.portfolio.compose output (target allocation)
    trend_core : signals.trend_core.evaluate output (for the why-bullets)
    prev       : {date, weights, cashWeight[, source]} — the holdings to diff
                 against: broker.reconcile.load_holdings() (실잔고, source=
                 "broker") when a fresh snapshot exists, else
                 paper.portfolio_tracker.last_weights_before, or None
    satellite  : signals.sector_rotation.evaluate_weekly output (why-bullets)
    fear_dip   : regime["fearDip"] dict (why-bullets)
    prices     : {ticker: latest close} — passed through (filtered to target
                 tickers) for the report's share-count calculator
    """
    target = {k: float(v) for k, v in (portfolio.get("weights") or {}).items()}
    cash_w = float(portfolio.get("cashWeight", max(0.0, 1.0 - sum(target.values()))))
    eff = float(portfolio.get("effExposure", sum(target.values())))

    prev_w = {k: float(v) for k, v in ((prev or {}).get("weights") or {}).items()}
    prev_date = (prev or {}).get("date")

    # ── Trades: diff prev -> target ───────────────────────────────────────────
    trades = []
    for tk in sorted(set(prev_w) | set(target)):
        frm, to = prev_w.get(tk, 0.0), target.get(tk, 0.0)
        if abs(to - frm) < REBALANCE_MIN_DELTA:
            continue
        trades.append({
            "ticker": tk,
            "action": _action_label(frm, to),
            "fromPct": round(frm * 100, 1),
            "toPct": round(to * 100, 1),
            "deltaPct": round((to - frm) * 100, 1),
        })
    trades.sort(key=lambda t: -abs(t["deltaPct"]))

    # ── Stance + headline ─────────────────────────────────────────────────────
    invested = sum(target.values())
    if prev is None:
        stance = "start"
        headline = "오늘 할 일: 신규 설정 — 아래 목표 비중대로 매수"
    elif not trades:
        stance = "hold"
        headline = "오늘 할 일: 없음 — 그대로 보유"
    elif invested < REBALANCE_MIN_DELTA:
        stance = "all_cash"
        headline = "오늘 할 일: 전량 현금화 (추세 OFF) — 매도 후 BIL/SGOV 파킹"
    else:
        stance = "rebalance"
        headline = f"오늘 할 일: 리밸런스 {len(trades)}건"

    # ── Why bullets (priority order, each one short sentence) ─────────────────
    why: list[str] = []
    tc = trend_core or {}
    spy_on, qqq_on = tc.get("coreSpyOn"), tc.get("coreQqqOn")
    dist = tc.get("distPct")
    dist_s = f" (SPY 200일선 대비 {dist:+.1f}%)" if isinstance(dist, (int, float)) else ""
    if spy_on and qqq_on:
        why.append(f"추세 ON — SPY·QQQ 모두 200일선 위{dist_s}")
    elif spy_on:
        why.append(f"SPY만 추세 ON — QQQ 슬리브는 현금{dist_s}")
    elif qqq_on:
        why.append("QQQ만 추세 ON — SPY 슬리브는 현금")
    elif spy_on is not None:
        why.append(f"추세 OFF — 양쪽 200일선 아래{dist_s}, 현금은 BIL/SGOV 파킹")

    fd_active = bool((fear_dip or {}).get("isEntry")) or \
        bool(((fear_dip or {}).get("paper") or {}).get("open", 0) > 0)
    if vix is not None:
        if fd_active:
            why.append(f"VIX {vix:.1f} + 공포매수 트리거 — SPXL 틸트 적용 중")
        elif tc.get("calmBoost"):
            why.append(f"VIX {vix:.1f} 저변동 + 양쪽 상승 — 캄-불 SPXL 부스트(20%) 작동")
        else:
            why.append(f"VIX {vix:.1f} — 평시 구성 (SPXL 기본 5%)" if spy_on
                       else f"VIX {vix:.1f}")

    sat = satellite or {}
    pick = sat.get("pick") or []
    if portfolio.get("enabled"):
        as_of = sat.get("pickAsOf")
        as_of_s = f" ({as_of} 금요일 종가 선정)" if as_of else ""
        if pick:
            why.append(f"섹터 슬리브 {portfolio.get('sleeveWeight', 0.15)*100:.0f}%: "
                       f"{'·'.join(pick)} — RSI-14 상위{as_of_s}, 주 1회 교체")
        else:
            why.append("섹터 슬리브: 전량 현금 — 상위 RSI 섹터가 모두 200일선 아래")
        # Rotation preview: today's RAW ranking vs the Friday-pinned pick. The
        # held pick stays until Friday's close, but flagging the divergence
        # early means Monday's swap never arrives as a surprise.
        ranked = sat.get("ranked") or []
        top_k = int(sat.get("topK", 2) or 2)
        daily_pick = [r["ticker"] for r in ranked[:top_k] if r.get("above200")]
        if pick and daily_pick and set(daily_pick) != set(pick):
            why.append(f"참고: 오늘 기준 일간 RSI 순위는 {'·'.join(daily_pick)} — "
                       f"금요일 종가로 확정되면 다음 주 교체 가능")

    # ── Stop-proximity alerts ─────────────────────────────────────────────────
    alerts: list[str] = []
    ex = tc.get("exec") or {}
    for leg, applies in (("spy", "SPY·SPXL"), ("qqq", "QQQ")):
        e = ex.get(leg)
        if not (e and e.get("price") and e.get("stop") and e["stop"] > 0):
            continue
        dist = (float(e["price"]) / float(e["stop"]) - 1.0) * 100
        if 0 <= dist < STOP_WARN_PCT:
            alerts.append(f"추세 손절선 근접 — {leg.upper()} 종가가 손절선(${e['stop']:.2f})까지 "
                          f"여유 {dist:.1f}%. 이탈 시 {applies} 청산 지시가 나옵니다")

    out_prices = {}
    for tk in target:
        v = (prices or {}).get(tk)
        if v is not None and v == v and float(v) > 0:
            out_prices[tk] = round(float(v), 2)

    return {
        "stance": stance,
        "headline": headline,
        "trades": trades,
        "nTrades": len(trades),
        "targetWeights": {k: round(v, 4) for k, v in target.items()},
        "cashWeight": round(cash_w, 4),
        "effExposure": round(eff, 2),
        "why": why,
        "alerts": alerts,
        "prices": out_prices,
        "prevDate": prev_date,
        "prevSource": ((prev or {}).get("source") or "tracker") if prev else None,
    }


def format_target_line(decision: dict) -> str:
    """One-line target allocation, weight-descending, leverage tagged."""
    parts = [f"{tk} {_fmt_pct(w)}{' (3x)' if tk in _LEV_TICKERS else ''}"
             for tk, w in sorted(decision.get("targetWeights", {}).items(),
                                 key=lambda kv: -kv[1]) if w > 0.001]
    if decision.get("cashWeight", 0) > 0.001:
        parts.append(f"현금/BIL {_fmt_pct(decision['cashWeight'])}")
    return " · ".join(parts) if parts else "현금 100%"
