# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

미국 인덱스/섹터/레버리지 ETF 당일 방향성 시그널 시스템.  
GitHub Actions가 매일 배치 실행 → HTML 리포트를 GitHub Pages에 발행 → 텔레그램으로 링크 발송 → **사용자 수동 실행** (토스증권, 한국 브로커, PDT 룰 비적용).
토스증권 Open API(2026-06 신규, 미국주식 주문 지원)를 통한 자동화는 `REVIEW_2026-06-12_auto_trading.md` 참조 — Tier-2 게이트(~2026-08-29) 통과 전 실자본 자동주문 금지, 그 전엔 읽기 전용(잔고 대조)까지만. 읽기 전용 통합은 `broker/`(2026-06-12 스캐폴딩): 우분투 서버가 `scripts/sync_broker_holdings.py`로 비중-only 스냅샷(`data/broker/holdings.json`)을 커밋 → 파이프라인이 신선하면(≤4일) decision의 '현재 보유'로 사용, 아니면 트래커 기록 폴백. **`broker/`에 주문 코드 추가 금지**(구조적 읽기 전용, `tests/broker/test_toss_readonly.py`가 강제). API 키는 서버 전용 — GH secrets에 두지 않는다. 엔드포인트 경로는 키 승인 후 실응답으로 확정 필요(env로 오버라이드 가능).

## Commands

```sh
# Install deps
pip install -r requirements-dev.txt

# Run full pipeline locally (phase 0 = data only)
SYSTEM_PHASE=0 FRED_API_KEY=xxx python scripts/run_daily.py

# Refresh historical data only
python -m data.collector

# Run all tests
pytest tests/ -v

# Run look-ahead bias test (run after every factor change)
pytest tests/data/test_lookahead.py -v
```

## Architecture

```
GitHub Actions (1일 1회 배치)
 ├── data/collector.py     — yfinance + FRED → data/raw/*.parquet
 ├── features/factors.py   — 기술적 팩터 (인과적, look-ahead 금지)
 ├── models/inference.py   — 저장된 가중치 → 방향 스코어 [-1,+1] + 신뢰도
 ├── signal/generator.py   — 진입가/TP/SL + 적중률/손익비/기대값 검증
 ├── report/builder.py     — HTML 템플릿에 REPORT JS 객체 주입 → docs/
 └── bot/notifier.py       — 텔레그램 단방향 푸시 (요약 + 리포트 링크)

scripts/run_daily.py       — Actions 진입점. 위 모듈 순서대로 호출.
config.py                  — 모든 상수의 단일 출처 (비용, 게이트, 임계값)
data/universe.py           — ETFMeta 목록 + Phase별 활성 여부
```

모델 학습은 Actions 외부(로컬/별도 환경)에서 수행 → `models/weights/`에 커밋 → Actions는 추론만.

## Critical constraints

**거래비용은 낮추지 않는다.**  
`config.TOTAL_COST_ROUNDTRIP` (≥0.25% 왕복, 토스증권 0.1%×2 + 슬리피지)을 모든 백테스트·기대값 계산에 사용한다. 이 값을 다른 곳에 하드코딩하지 않는다.

**시그널 유효 조건은 세 가지 동시 충족.**  
`Signal.is_valid()`: `winrate > 0` AND `payoff > 1.0` AND `expectancy > 0` (비용 차감 후). 하나라도 음수면 미발송.

**Look-ahead 편향 금지.**  
팩터 추가 후 반드시 `pytest tests/data/test_lookahead.py` 실행. 모든 팩터는 날짜 T의 종가까지만 사용.

**레버리지는 마지막.**  
`LEVERAGE_EDUCATION_DONE=true` AND `SYSTEM_PHASE >= 5` AND 신뢰도 최상위 버킷 입증 후에만 활성화. `data/universe.py`의 `requires_education=True` 항목 참조.

**IC 최소 기준: 0.01.**  
`config.IC_MIN_VIABLE` 미달 시 전략 보류. 거래비용(왕복 0.25%)이 신호를 잠식한다.

## Configuration

`config.py`가 단일 진실 출처. 주요 상수:

| 상수 | 값 | 의미 |
|------|----|------|
| `TOTAL_COST_ROUNDTRIP` | 0.0025 | 왕복 비용 (수수료+슬리피지) |
| `TIER1_SHARPE_MIN` | 0.7 | 페이퍼 진입 게이트 |
| `TIER1_MDD_MAX` | 0.25 | |
| `TIER1_MIN_TRADING_DAYS` | 120 | 최소 표본 |
| `TIER2_PAPER_SHARPE_MIN` | 0.5 | 실거래 진입 게이트 |
| `IC_MIN_VIABLE` | 0.01 | IC 하한 |

## Signal output contract

`report/builder.py:_signal_to_dict()` 반환 형태가 HTML 템플릿의 데이터 컨트랙트.  
변경 시 `report/templates/daily-signal-report-template.html`의 JS 렌더러도 함께 수정.

**Decision contract (2026-06-11, 강화 2026-06-12):** `signals/decision.py:build_decision()`이 만드는 `regime["decision"]` (stance/headline/trades/why/prevSource)이 1순위 표시 대상. '현재 보유'는 broker 스냅샷(신선 시) > 트래커 기록 순. **텔레그램은 지시-only**: 오늘 할 일 diff + 목표 포트폴리오 + 손절 + 링크 — 근거 불릿·페이퍼 진행은 리포트 전용(메시지 재추가 금지). 리포트 상단도 히어로+목표만, 검증 트랙은 접힌 참고 섹션. 보정승률(~57% 평탄, 변별력 0)은 어떤 표면에도 표시하지 않는다 — 컬럼 부활 금지, `calWin`/`estEv`는 페이로드에서도 제거됨(2026-06-12). 페이퍼 Sharpe는 95% CI(`sharpeCi`)와 함께 표시(소표본 과신 방지). `report/builder.py:write_index()`가 docs/index.html(최신 리포트 리다이렉트)을 매 실행 갱신.

## GitHub Actions cron

`daily_signal.yml`은 **`workflow_dispatch` 전용**이다. GitHub native `schedule`은 1~3시간 지연·누락이 잦아(미국 장 시작 전 도착 보장 불가) 제거됐다. 대신 외부 우분투 서버 cron이 매일 13:00 UTC(22:00 KST, 월~금)에 dispatch API를 POST해 트리거한다(`scripts/trigger_daily_signal.sh`). 중복 실행 방지: `concurrency.group: daily-signal`, `cancel-in-progress: false`.
`sentiment.yml`은 별개로 native `schedule: 0 13 * * 1-5`를 유지한다(v4용 사전수집, 코어 파이프라인 미편입).
`watchdog.yml`(2026-06-12): 평일 14:30 UTC native schedule 백스톱 — 당일 `docs/<날짜>.html` 미발행 시 텔레그램 경보. 우분투 cron·dispatch·워크플로 실패의 무감지 리스크용이며, native 지터(1~3h)는 "늦은 경보 > 무경보"로 허용.

## Phase roadmap

| Phase | 주요 작업 | 현재 |
|-------|-----------|------|
| 0 | 데이터 파이프라인 + 유니버스 확정 + look-ahead 검증 | ✅ 완료 |
| 1 | 단일 팩터 IC 검증, 무효 팩터 폐기 | ✅ 완료 |
| 2 | 베이스라인 규칙 모델 + 백테스트 | ✅ 완료 |
| 3 | LightGBM 추론 (v3, +macro, top-K) + walk-forward | ✅ 운영 중 |
| 4 | 페이퍼 트레이딩 (conviction Friday-K=1 + fear-dip) | ✅ 운영 중 |
| 5 | 주력 엔진: trend-core + 캄-불 부스트 + RSI 섹터 슬리브 블렌드(85/15) | **운영 중 (2026-05-24~)** |

**현재 운영 구성 (2026-05-31 기준):** 주력은 **추세코어 85% + RSI 섹터 슬리브 15% 블렌드** (`signals/portfolio.py` compose). 구성요소:
- **추세코어** (`signals/trend_core.py`): SPY/QQQ 50/50, VIX<22면 200SMA·≥22면 50&200 골든크로스. 추세-on 시 SPXL 5% 상시, fear-dip 활성 시 15% 틸트, **양쪽 상승+VIX<16(캄-불)이면 SPXL 20% 부스트**(`TREND_CORE_STRONG_SPXL`). 현금 구간 BIL/SGOV(IRX).
- **RSI 섹터 슬리브** (`signals/sector_rotation.py`): 6개 섹터 RSI-14 상위 2개(200SMA 위 조건)에 자본의 15%(`config.SECTOR_SLEEVE_WEIGHT`). LightGBM은 섹터 횡단면 스킬 0(IC≈0)이라 RSI로 대체. 검증: 블렌드 Full OOS 2021+ +124%/Sharpe1.23, Holdout +59%/1.51 (엔진단독 +114%/1.12 대비 위험조정 개선). **pick은 `evaluate_weekly()`로 마지막 금요일 종가에 고정**(주 1회 교체) — 백테스트(`sweep_blend_goal.py:159`)가 검증한 cadence. 2026-06-11 이전엔 매일 재계산되어 주중 4회 교체·비용 드래그가 발생했음(수정됨). 일간 `evaluate()`를 라이브 pick에 직접 쓰지 말 것.
- **블렌드 "goal" 노브 재현·상향 조건**: 슬리브/STRONG-SPXL 상향은 ① `scripts/sweep_blend_goal.py`(프로덕션 함수 day-by-day replay, 비용차감·look-ahead-safe) 재현으로 **Tier-1 게이트 PASS** 확인, ② **페이퍼 검증 윈도우(~2026-08-29) 완료 후** — 둘 다 충족 후에만 올린다. (검증 중 가중치 변경은 트래커가 대조하는 하드와이어 목표 `blendFull` Sharpe 1.23을 무효화한다.) 시세 호스트 차단 환경에선 `.github/workflows/blend_goal_sweep.yml`(workflow_dispatch)로 CI 실행. **미재현 수치로 실자본 노브 상향 금지**(부록 B). **재현 결과**(2026-06-01, data~05-29, 출처 `data/logs/blend_goal_sweep.csv`): 출하 0.15/0.20 = Full +121%/1.22·Holdout +58%/1.49 (docstring claim +124%/1.23·+59%/1.51을 ~3% 이내 근사). **0.25는 claim +131%/1.30이 재현 안 됨** → 실제 +124%/1.26·MDD -14%대로 악화. 상위 셀의 Sharpe 이득은 +0.02~0.05로 한계적이며 MDD도 동반 미세 악화(0.20도 Holdout MDD -12.6%→-13.4%, 0.25는 -14%대) → 결정규칙(Sharpe≥출하 AND MDD 불악화)상 **무료 상향 셀 없음**. **현 라이브는 슬리브 15% 유지.**
- **3개월 페이퍼 검증 중** (`paper/portfolio_tracker.py` NAV추적, ~2026-08-29 만기) — Tier-2 게이트 통과 전 실자본 미전환.
- **stale-data 가드**: 최신 종가가 4일 초과 지연 시 리포트·텔레그램 경고(`data.collector.data_freshness`).
- conviction/fear-dip 신호는 새틀라이트(참고용); 외부 우분투 cron이 22:00 KST에 workflow_dispatch 트리거. 메모: [[project_trend_core_engine]], [[project_model_skill_rsi_rotation]], [[project_improvement_loop_0531]].

Tier 1: Sharpe > 0.7, MDD < 25%, OOS/IS ≥ 40%, walk-forward 과반 양수, ≥ 120거래일.  
Tier 2: 페이퍼 3개월+, 실현성과/백테스트 괴리 < 40%, 페이퍼 Sharpe > 0.5.

## Honesty principles (부록 B)

1. 베이스라인을 못 이기는 ML은 도입하지 않는다.
2. 성과는 비용 차감 후 수치로만 보고한다.
3. 적중률은 손익비·기대값·표본과 함께만 표시한다.
4. 2단계 게이트를 건너뛰지 않는다.
5. 알파가 없다는 결론도 유효한 결과다.
6. 9개월 킬스위치를 감정으로 연장하지 않는다.
