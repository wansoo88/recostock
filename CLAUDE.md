# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

미국 인덱스/섹터/레버리지 ETF 당일 방향성 시그널 시스템.  
GitHub Actions가 매일 배치 실행 → HTML 리포트를 GitHub Pages에 발행 → 텔레그램으로 링크 발송 → **사용자 수동 실행** (토스증권, 한국 브로커, PDT 룰 비적용).

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

## GitHub Actions cron

DST 양쪽을 커버하기 위해 `20:30 UTC`와 `21:30 UTC` 두 개 cron 실행.  
중복 실행 방지: `concurrency.group: daily-signal`.

## Phase roadmap

| Phase | 주요 작업 | 현재 |
|-------|-----------|------|
| 0 | 데이터 파이프라인 + 유니버스 확정 + look-ahead 검증 | **In progress** |
| 1 | 단일 팩터 IC 검증, 무효 팩터 폐기 | — |
| 2 | 베이스라인 규칙 모델 + 백테스트 (왕복 0.2% 반영) | — |
| 3 | LightGBM → GRU + walk-forward | — |
| 4 | Tier 1 게이트 → 페이퍼 트레이딩 (1x ETF) | — |
| 5 | Tier 2 게이트 → 소액 실거래 + 레버리지 버킷 입증 | — |

Tier 1: Sharpe > 0.7, MDD < 25%, OOS/IS ≥ 40%, walk-forward 과반 양수, ≥ 120거래일.  
Tier 2: 페이퍼 3개월+, 실현성과/백테스트 괴리 < 40%, 페이퍼 Sharpe > 0.5.

## Honesty principles (부록 B)

1. 베이스라인을 못 이기는 ML은 도입하지 않는다.
2. 성과는 비용 차감 후 수치로만 보고한다.
3. 적중률은 손익비·기대값·표본과 함께만 표시한다.
4. 2단계 게이트를 건너뛰지 않는다.
5. 알파가 없다는 결론도 유효한 결과다.
6. 9개월 킬스위치를 감정으로 연장하지 않는다.
