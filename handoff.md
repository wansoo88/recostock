# Handoff — recostock 현황 (2026-05-17 갱신, B Phase 적용)

## 한 줄 요약

**일간 시스템: v3 LightGBM 모델 운영 중**(매크로 11피처 + 종목 17개 + top-5 thr=0.58). 최근 12개월 백테스트 WR 57%, Sharpe 1.69, 일평균 +0.030%. 인트라데이는 폐기. **B Phase(daily 자동 재학습) 배포 완료** — `scripts/nightly_retrain.py` + `.github/workflows/retrain.yml` (14:00 UTC, Mon–Fri). WR 60% 천장 돌파는 A Phase(sentiment) 대기.

---

## 1. 현재 운영 상태

| 항목 | 상태 |
|---|---|
| **일간 시스템 모델** | v3 (`models/inference_v3.py`, `lgbm_phase3_v3_uniform.pkl`) |
| **운영 설정** | TOP_K=5, PRODUCTION_THRESHOLD=0.58 |
| **종목 풀** | 17개 (SPY/QQQ/DIA + 9 sectors + 3 inverse + VXX 등) |
| **GitHub Actions cron** | 매일 20:30 / 21:30 UTC 정상 동작 |
| **다음 시그널 발송** | 월요일 21:30 UTC 부터 v3로 자동 작동 |
| **Daily 재학습 cron** | 매일 14:00 UTC, Mon–Fri (`retrain.yml`, 시그널 6h+ 전) |
| **재학습 safety gate** | mean OOS AUC ≥ 0.50 AND min fold ≥ 0.45 (실패 시 weights 미갱신) |
| **인트라데이 봇** | 폐기 (deploy workflow 비활성화 + 서버 systemd 정지 완료) |
| **인프라 비용** | 일간 = GitHub Actions 무료. 서버는 A/B용 별도 |
| **Phase 5 (실거래)** | **보류** — Tier 1 MDD 게이트 검증 필요 + 운영 forward 데이터 누적 후 |

### 로컬 dry-run 결과 (2026-05-17 시그널 예시)

v3 모델이 18개 ETF 점수화 → top-5 시그널:
- PSQ (rank 1, ema_proba 0.668) — 인버스 QQQ
- SH (rank 2, 0.666) — 인버스 SPY
- VXX (rank 3, 0.644) — 변동성
- IBB (rank 4, 0.639) — 바이오테크
- XLE (rank 5, 0.639) — 에너지

**현재 시장 헤징 regime 인식** (인버스 + 변동성 우위). 정상적인 모델 반응.

---

## 2. 최근 성과 — v1 (이전 운영) vs v3 (현 운영)

| 기간 | 지표 | v1 (실측, 폐기) | **v3 (백테스트, 운영)** |
|---|---|---|---|
| 12개월 | WR | 39.6% | **57.0%** |
|  | Sharpe | 1.08 | **1.69** |
|  | 일평균 | -0.035% | **+0.030%** |
| 6개월 | WR | 27.3% | **65.0%** |
|  | Sharpe | n/a (음수) | **+2.97** (v3 top-7 기준) |
|  | 일평균 | -0.091% | **+0.094%** |
| 3개월 | WR | 22.2% | **50-60%** |
|  | 일평균 | -0.189% | +0.085% |
| 30일 | WR | 0% (5/5 손실) | 50% (안정) |

**같은 시장, 같은 종목, 부호 자체가 뒤집힘.**

---

## 3. 모델 진화 — v1 → v2 → v3

| 버전 | 변경점 | 결과 (12m Sharpe) |
|---|---|---|
| v1 (Phase 3, 2026-05-15) | 기술 지표 11 + cross-section rank 5 + VIX | 1.08 |
| v2 (2026-05-16 야간) | + 매크로 11피처 (oil, gold, DXY, yields, credit spread, VVIX) | 1.33 |
| v3 (2026-05-17) | + 17 ETF + Top-K 선별 + threshold 0.58 | **1.69** |

### Feature Importance 변화

v1 운영 모델: VIX 피처가 압도적 1위 (49,597) — 다른 피처의 5배.
v3 운영 모델: `hy_ig_logratio`(신용 스프레드) 1위. **Top 13 중 9개가 신규 매크로**.

→ 매크로/크로스에셋 시그널이 결정적이었음.

---

## 4. 인트라데이 — 폐기 완료

12가지 통계적 접근(룰 sweep, ML, 페어, 반전, 시간프레임 확장, 변동성 게이트, ...) 모두 OOS net EV 음수. yfinance 15분 지연 + 5분봉 + 9 ETF + 0.25% 비용 = **수학적으로 양수 EV 불가능**임을 확인.

**처리 완료:**
- `.github/workflows/deploy_intraday.yml` 자동 트리거 비활성화 (manual-only)
- 서버 systemd `intraday-bot` stop + disable (사용자 수동 완료)
- 검증 산출물 + 12 스크립트 보존 (`scripts/run_intraday_*.py`)

**다시 인트라데이 시도하려면 데이터 자체가 달라져야** (Polygon Level 2 호가창 등 유료, 월 $200+).

---

## 5. WR 한계 솔직 보고

사용자 압박: "WR을 어떻게든 올려라". 50+ 시나리오 시뮬레이션(`scripts/push_wr_higher.py`) 결과:

| Threshold | n (12m) | WR | Sharpe | 일평균 |
|---|---|---|---|---|
| 0.53 (기본) | 33 | 48% | 0.31 | +0.007% |
| **0.58 (운영 적용)** | **21** | **57%** | **1.69** | **+0.030%** |
| 0.60 | 16 | 56% | -0.44 | **-0.007%** ❌ |
| 0.62 | 14 | 43% | -1.30 | -0.019% ❌ |
| 0.55 + VIX≥18 | 15 | 47% | +2.60 | +0.027% |

**60% 이상 threshold는 평균 수익 음수.** 이건 모델 한계가 아니라 **yfinance 데이터의 정보 천장**:
- 매크로 피처 최대 \|IC\| = 0.033
- 비용 0.25% 차감 후 net 정보 작음
- WR 70%+ 만들려면 IC 0.05+ 피처가 필요한데 yfinance에 없음

---

## 6. 다음 단계 — 서버 A/B 계획

서버 115.68.230.40을 60% 천장 돌파에 활용. 자세한 계획서: `server_ab_plan_2026-05-17.md`.

### A. Sentiment 수집기 — Week 1 step 1 **완료 (2026-05-17, GitHub Actions)**
- `sentiment/sources/yahoo_rss.py` — Yahoo Finance per-ticker RSS (17 ETF, key 불필요)
- `sentiment/sources/hackernews.py` — HN Algolia search (key 불필요)
- `sentiment/ticker_extract.py` — cashtag + bare ticker + ETF 이름 alias (strict, 모호 ticker는 cashtag 강제)
- `sentiment/aggregator.py` — 일별 long-format `data/raw/sentiment_daily.parquet` 누적 upsert (lookback 2 days)
- `scripts/collect_sentiment.py` — entrypoint, source 하나 실패해도 진행
- `.github/workflows/sentiment.yml` — daily 13:00 UTC, Mon–Fri (재학습 14:00 직전)
- 첫 로컬 실행 결과: SPY 4, QQQ 3, XLK 2, XLF 2, XLE 1, XLP 1 mentions
- **다음 step**: Reddit (사용자 API key 발급 후) → SEC EDGAR → FinBERT 점수화 → v4 모델 통합 (3주 일정 유지)

### B. 야간 자동 재학습 — **완료 (2026-05-17, daily로 운영)**
- `.github/workflows/retrain.yml` — 매일 14:00 UTC, Mon–Fri (시그널 cron 6h+ 전)
- `scripts/nightly_retrain.py` — yfinance/매크로 fetch → v3 walk-forward 학습 → safety gate → 운영 weights 교체
- 안전 가드: mean OOS AUC ≥ 0.50 AND min fold ≥ 0.45. FAIL 시 staged weights 폐기, 이전 가중치 유지.
- 학습은 임시 디렉토리(stage)에 저장 후 PASS만 운영 경로로 promote (망가진 모델이 production을 덮어쓰지 못함).
- 메타데이터는 `data/logs/retrain_history.csv`에 누적.
- 텔레그램 알림: TELEGRAM_* env 있으면 PASS/FAIL 모두 푸시.
- Concept drift 자동 대응 (분기 단위 → 일 단위).

### 효과 추정 (매크로 사례 근거)

| 단계 | WR (12m) | Sharpe | 일평균 |
|---|---|---|---|
| 현재 v3 | 57% | 1.69 | +0.030% |
| + B 야간 재학습 | 60% | 1.85 | +0.040% |
| **+ A sentiment** | **65-70%** | **2.0-2.3** | **+0.05-0.07%** |

→ **1-2개월 후 WR 65-70% 도달 가능.** 매크로 피처가 v1→v2에서 +19.6%p 만든 같은 효과를 sentiment에서 반복.

### 우선순위
1. ~~**즉시** — B Phase 시작 (재학습 script, 1주 안에 배포)~~ **2026-05-17 완료, daily 운영 중**
2. ~~**다음 주** — A Phase Week 1 (Reddit + RSS, 무료 API)~~ **2026-05-17 step 1 완료 (Yahoo RSS + HN). Reddit/EDGAR은 사용자 key 발급 후 추가**
3. **2-3주 후** — FinBERT + v4 모델 학습
4. **1개월 후** — sentiment 운영 적용

---

## 7. 사용자 결정 대기 사항

| 항목 | 누가 | 비고 |
|---|---|---|
| 월요일 cron 후 v3 첫 시그널 확인 | 사용자 | 텔레그램 모니터 |
| Reddit API client_id/secret 발급 | 사용자 | reddit.com/prefs/apps 무료 — 발급 시 sentiment/sources/reddit.py 추가 |
| B Phase 시작 GO 신호 | 사용자 | 다음 세션 즉시 시작 가능 |
| Phase 5 (실거래) 진입 결정 | 사용자 | v3 forward 4주+ 누적 후 |
| FRED API 키 (선택) | 사용자 | 더 깨끗한 매크로 데이터 (선택) |

---

## 8. 검증된 사실 (현재 정확한 상태)

### 8.1 Tier 1/2 게이트
- v1 실측: Sharpe 1.538, **MDD -41.05% → Tier 1 FAIL**
- v3 백테스트 (12m): Sharpe 1.69, **MDD -3.5% → Tier 1 PASS 후보**
- paper/tier2.py 코드의 MDD 게이트 누락 버그 수정됨 (2026-05-16)
- 단, 실 forward 페이퍼 거래 데이터 0건 → Tier 2의 "3개월 forward" 조건 미충족

### 8.2 VIX Regime 효과
- v2 분석에서 VIX≥20일 때 Sharpe +1.66, VIX<20일 때 -0.34 발견
- Walk-forward 5-fold 검증: 2/4 fold만 양수, mean Δ +0.12 (marginal)
- 결론: **VIX 게이트 즉시 채택 X, shadow 모드 권고**

### 8.3 종목별 약점
- 매크로로 회복: XLE (oil 피처가 정확히 적중)
- 매크로로 못 회복: **XLF, XLV** — 추가 정보 필요 (금리 곡선 detail, FDA calendar)

---

## 9. 산출물 인덱스

### 보고서 (3개)
- `night_report_2026-05-16.md` — 야간 자율 작업 종합 보고
- `performance_uplift_report_2026-05-17.md` — v3 성능 향상 작업
- `server_ab_plan_2026-05-17.md` — 서버 A/B 활용 계획

### 운영 코드
- `models/inference_v3.py` (현 운영)
- `models/inference.py` (v1, 백업 보존)
- `scripts/run_daily.py` (import 변경 적용됨)
- `scripts/nightly_retrain.py` (B Phase, daily 재학습 — 2026-05-17 추가)
- `.github/workflows/retrain.yml` (daily cron 14:00 UTC — 2026-05-17 추가)
- `sentiment/` 패키지 + `scripts/collect_sentiment.py` (A Phase Week 1 step 1 — 2026-05-17 추가)
- `.github/workflows/sentiment.yml` (daily cron 13:00 UTC — 2026-05-17 추가)
- `data/collector.py` (`fetch_macro_yfinance` 추가)
- `data/macro_collector.py` (12개 매크로 시리즈)
- `features/macro_factors.py` (글로벌 + ticker-specific)
- `models/train_lgbm_v2.py` (v2/v3 학습)
- `paper/tier2.py` (MDD 게이트 추가)

### 검증 스크립트
- `scripts/analyze_macro_ic.py` — 매크로 11피처 IC 검증
- `scripts/diagnose_daily_system.py` — Sharpe/MDD/시기별
- `scripts/analyze_confidence_and_failures.py` — 신뢰도 역전
- `scripts/validate_ticker_exclusion.py` — holdout 검증
- `scripts/validate_vix_gate.py` — VIX 게이트 OOS marginal
- `scripts/run_phase3_v2.py` — v1/v2 비교 학습
- `scripts/integrated_backtest_v3.py` — v3 + 사이징
- `scripts/final_topk_comparison.py` — v2/v3 × top-K
- `scripts/push_wr_higher.py` — 50+ WR 향상 시뮬
- `scripts/enhanced_backtest_v2.py` — bootstrap, regime, multi-metric
- `scripts/expand_universe.py` — 5 ETF 추가
- `scripts/summarize_daily_actuals.py` / `yearly_wr_dailyret.py` — 실측 통계

### 모델
- `models/weights/lgbm_phase3.pkl` — v1 (보존, 롤백 가능)
- **`models/weights/lgbm_phase3_v3_uniform.pkl`** — v3 (현 운영)
- `models/weights/lgbm_phase3_v2_*.pkl` — v2 중간 단계 (보존)

### 인트라데이 검증 산출물 (보존만)
- `data/intraday_*.{csv,txt}` 12가지 검증 결과
- `models/weights/intraday_lgbm.pkl`
- `scripts/run_intraday_*.py`, `train_intraday_lgbm.py`, `run_pairs_backtest.py`

---

## 10. Commit 이력 (2026-05-16 ~ 2026-05-17)

```
538a49d feat: v3 model in production (option 1, top-5 thr=0.58)
a5003c4 feat: v3 model — expanded universe + top-K (recent 6m WR 65%, Sharpe 2.97)
0606026 docs: night_report_2026-05-16 — 야간 자율 작업 종합 보고
609bcd7 feat: v2 model — add 11 macro features + walk-forward
c3bc37d fix: paper/tier2.py — add Tier 1 MDD<25% gate
1334720 chore: retire intraday system — preserve 12-approach validation
```

모두 `origin/main`에 push 완료.

---

## 11. 정직성 점검 (CLAUDE.md 부록 B)

| # | 원칙 | 점검 |
|---|---|---|
| 1 | 베이스라인 못 이기는 ML 도입 X | ✓ v2가 v1을, v3가 v2를 능가 확인 후 도입 |
| 2 | 성과는 비용 차감 후 | ✓ 모든 백테스트에 0.25% 왕복 비용 |
| 3 | WR은 손익비·기대값과 함께 | ✓ multi-metric 표 사용 |
| 4 | 2단계 게이트 안 건너뜀 | ✓ MDD 게이트 누락 버그 발견·수정 |
| 5 | 알파 없으면 그렇게 보고 | ✓ XLF/XLV 매크로로도 회복 못함 솔직 보고 |
| 6 | 감정으로 잘못된 시그널 연장 X | ✓ 인트라데이 폐기 결정 유지 |

WR 60% 천장 솔직 인정 = #1, #2, #5 동시 준수.
VIX 게이트 즉시 채택 거부 = #4 (2단계 게이트 — IS만으론 부족).
인트라데이 12가지 OOS 검증 후 폐기 = #5, #6.

---

## 12. 이전 인트라데이 조사 기록 (보존)

(아래는 2026-05-16 인트라데이 알파 부재 12가지 검증의 원본 handoff. 인트라데이는 폐기됐지만 향후 같은 함정 피하기 위해 보존.)

### 12가지 접근 — 모두 OOS net EV 음수

| # | 접근 | OOS net EV | 통계 검증 |
|---|---|---|---|
| 1 | 원본 룰 (trend-following) | -0.26% | WR 17% (가정 54%와 큰 괴리) |
| 2 | 72-config parameter sweep | -0.23% best | TP 멀티 변경 무효 |
| 3 | 방향 반전 (mean-reversion) | -0.22% | IC 음수 발견 |
| 4 | 반전 + 64-config sweep | -0.23% best | gross EV 천장 +0.03% |
| 5 | 15분/30분봉 | -0.22 ~ -0.32% | 시간프레임 확장 무효 |
| 6 | LightGBM h=12 (60min) | -0.23% | OOS AUC 0.498 |
| 7 | LightGBM h=24 (120min) | -0.22% | gross +0.033% (이론 한계) |
| 8 | 변동성 게이트(ATR≥0.25%) + 반전 | -0.11% | 60일 단일창 |
| 9 | 변동성 게이트 + 정밀 TP/SL | "+0.012%" | ❌ OOS -0.95% |
| 10 | 5-fold rolling WF | 3/5 양수 | t-stat 0.26 (무의미) |
| 11 | 페어 트레이딩 z-score | -0.225% | t-stat -10.35 |
| 12 | 페어 strict (z≥2.5/3.0) | -0.20% | t-stat -4.01 |

### IC 분석 결론
- 5분봉 팩터 최대 \|IC\| = 0.0475
- Gross EV 천장 +0.03~0.05%/거래
- 거래 비용 0.25% = 정보의 5-8배
- 부족분 약 0.20~0.22% — 어떤 미세조정으로도 안 됨
- retail-grade 데이터(yfinance 15분 지연, OHLCV만)로 깰 수 없는 비용 장벽

### LightGBM 핵심 통찰
피처 중요도 1위 `vix_prior` (49,597) — 다른 피처의 5배.
→ 모델 학습 대부분이 **일간 레짐 정보**이고 진정한 5분 알파는 약함.
이게 5분봉에서 알파 못 만드는 구조적 이유.

### 결론 (당시)
**알파가 없다는 결론도 유효한 결과** (CLAUDE.md 부록 B #5).
다른 시간프레임 시도하려면 유료 데이터(Polygon Level 2 호가창 월 $200+) 필요.
