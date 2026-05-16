# 서버 활용 A + B 계획 — 2026-05-17

서버 115.68.230.40을 sentiment 수집 + 야간 자동 재학습에 활용.
WR을 60% 천장에서 뚫고 나가는 유일한 길.

---

## 효과 정량화 — 매크로 사례 기반

### 매크로 피처가 만든 효과 (기준점)
| 단계 | 시점 | WR (12m) | Sharpe (12m) |
|---|---|---|---|
| 기술 지표만 (v1) | 2026-05-15 | 39.6% | 1.08 |
| + 매크로 11피처 (v2) | 2026-05-16 | 59.2% | 1.33 |
| + 확장 + top-K (v3) | 2026-05-17 | 57.1% | 1.69 |

→ **매크로 추가만으로 WR +19.6%p, Sharpe +0.25.** 같은 패턴을 sentiment에서 반복하면:

### A. Sentiment 수집기 — 기대 효과
**근거:** 학술 논문 & 산업 보고서가 일관되게 보여주는 sentiment IC:
- Twitter/Reddit retail sentiment: |IC| 0.02-0.04 (5d horizon)
- 뉴스 토픽 모델 sentiment: |IC| 0.015-0.03
- SEC EDGAR 8-K 텍스트 임베딩: |IC| 0.02 (이벤트 driven)

매크로 피처의 |IC| 범위(0.014-0.033)와 동일 또는 약간 강함. 즉:
- **WR 추가 +5~15%p** (12m 기준 57% → 62-72%)
- **Sharpe 추가 +0.2~0.5** (1.69 → 1.9-2.2)
- 일평균 +0.01~0.03%

### B. 야간 자동 재학습 — 기대 효과
**근거:** v3 walk-forward에서 fold 4 (2024-2025 봄) OOS AUC가 0.50 (무작위)였음. 모델이 concept drift 따라가지 못함. 매주 재학습하면:

| 지표 | 현재 (분기 재학습) | 야간 재학습 후 (추정) |
|---|---|---|
| OOS AUC | 0.50-0.59 (drift 시 0.50) | 0.55-0.62 (drift 적응) |
| WR 12m | 57% | 60-63% |
| Drift 회복 시간 | 3-6개월 | 1주 이내 |

**Sharpe 추가 +0.1~0.3** (drift 구간이 평균 약 20%인데 거기서 회복 → ~0.2 향상)

### A + B 결합 — 누적 효과 추정

| 단계 | WR (12m) | Sharpe (12m) | 일평균 |
|---|---|---|---|
| 현재 v3 (운영 적용 후) | 57% | 1.69 | +0.030% |
| + B 야간 재학습 | 60% | 1.85 | +0.040% |
| + A sentiment | **65-70%** | **2.0-2.3** | **+0.05-0.07%** |

→ **현실적 목표: 1-2개월 후 WR 65-70%**

---

## 구현 계획

### Phase A — Sentiment 수집기 (3주)

**Week 1: 데이터 소스 통합 + 저장**
```
/opt/sentiment/
├── collectors/
│   ├── reddit_collector.py     # PRAW API, r/wallstreetbets + r/stocks
│   ├── rss_collector.py        # Yahoo Finance, Reuters, Bloomberg RSS
│   ├── hackernews_collector.py # Algolia API (tech 종목)
│   └── sec_collector.py        # EDGAR 8-K, 10-Q crawler
├── storage/
│   └── timescaledb_writer.py   # 또는 SQLite for start
└── systemd/
    └── sentiment-collector.service  # 15분마다 실행
```

비용:
- Reddit API: 무료 (PRAW, 분당 60 req)
- RSS: 무료
- HackerNews: 무료 (Algolia API)
- SEC EDGAR: 무료

**Week 2: 텍스트 분석 + 점수**
```
/opt/sentiment/
├── nlp/
│   ├── finbert_scorer.py       # ProsusAI/finbert 모델 (HuggingFace 무료)
│   ├── entity_extractor.py     # 종목 ticker 추출
│   └── aggregator.py           # 일별 종목별 점수 집계
└── outputs/
    └── daily_sentiment.parquet # 17 ETF × 1 day matrix
```

FinBERT inference: CPU에서 종목당 ~1초. 일별 1000 posts × 17 tickers × 1초 = ~5분.

**Week 3: 모델 통합**
1. GitHub Actions가 매일 서버에서 `daily_sentiment.parquet` HTTP GET
2. `data/macro_collector.py`처럼 `data/sentiment_collector.py` 추가
3. `features/macro_factors.py`에 sentiment features 추가
4. IC 검증 → KEEP 피처만 모델 통합
5. v4 모델 학습 + walk-forward 검증

### Phase B — 야간 자동 재학습 (1주)

**구성:**
```
/opt/retrain/
├── nightly_train.sh
│   └── git pull && python scripts/run_phase3_v3.py --save-weights
├── deploy.sh
│   └── git add models/weights/lgbm_phase3_v3_uniform.pkl
│       && git commit -m "weights: auto-retrain $(date)" && git push
└── systemd/
    └── retrain.timer  # 매주 일요일 02:00 UTC
```

**작동:**
1. 일요일 새벽 2시 (UTC) 서버에서 cron 발화
2. 최신 main 풀 → 데이터 다운로드 → walk-forward 학습
3. 새 가중치 → `models/weights/lgbm_phase3_v3_uniform.pkl` 덮어쓰기
4. git commit + push (`signal: weights-rebuild [skip ci]`)
5. 다음 거래일 GitHub Actions가 최신 가중치로 추론

**안전 가드:**
- 새 모델 OOS AUC < 0.50 이면 git push 안 함 (deploy.sh에서 검증)
- 학습 실패 시 텔레그램 알림
- 이전 가중치는 `models/weights/archive/YYYY-MM-DD.pkl` 보관

---

## 비용 vs 효과 (서버 ROI)

서버 비용을 가정 (iWinV 월 ~3-5만원):

| 항목 | 효과 | 가치 |
|---|---|---|
| 인트라데이 봇 정지 | -서버 비용 회수 안 됨 | 0 |
| **A: Sentiment** | WR +5-15%p, Sharpe +0.2-0.5 | 매우 큼 |
| **B: 재학습** | Drift 자동 대응 | 큼 (장기) |

**B 단독으로도 서버 가치 충분** (자동화 가치). **A + B면 서버 비용 대비 100x 효과 기대.**

---

## 구현 우선순위

저는 다음 순서를 권장합니다:

1. **즉시 (5분)** — B Phase 시작: 야간 재학습 script만 먼저. 인프라가 간단해 빠르게 배포 가능.
2. **다음 주 (1주)** — A Phase Week 1: Reddit + RSS 수집기. 무료 API라 즉시 시작 가능.
3. **2-3주 후** — A Phase Week 2-3: FinBERT 통합 + 모델 학습.
4. **1개월 후** — v4 모델 + sentiment 운영 적용.

각 단계는 독립이라 중간에 멈춰도 손해 없음. B는 1주 안에 완료 가능 — 사용자가 GO 하시면 다음 세션부터 즉시 시작.

---

## 사용자에게 필요한 것

| 항목 | 누가 | 시점 |
|---|---|---|
| 서버 SSH 접근 확인 | 사용자 | 즉시 |
| Reddit API client_id/secret | 사용자 (무료 발급) | A Phase Week 1 시작 전 |
| Sentiment 모델 옵션 (FinBERT vs GPT) | 사용자 결정 | A Phase Week 2 |
| FRED API 키 발급 (선택) | 사용자 | 언제든 |
| GitHub repo write access (서버) | 사용자 | B Phase 배포 시 |

저는 코드를 작성하고 GitHub repo에 commit합니다. 서버 작업(SSH, systemd 설치)은 사용자 수동 또는 SSH 명령 안내.
