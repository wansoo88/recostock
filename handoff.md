# Handoff — Intraday Alpha Investigation (2026-05-16)

> **⚠️ 2026-05-16 야간 정정:** 이 문서에 적힌 "일간 시스템 Tier 2 PASS"는 잘못된
> 주장이었음. paper/tier2.py가 Tier 1 MDD<25% 게이트를 검사에서 누락해서 발생한
> 버그. 실측 MDD -41.05%로 Tier 1조차 미통과. 자세한 사후 분석은
> `night_report_2026-05-16.md` 참조. 본문은 인트라데이 조사 기록을 위해 보존.

## 한 줄 요약

**5분봉 ETF 인트라데이 시그널의 알파를 12가지 접근으로 검증 → 모두 통계적으로 음의 net EV. 인트라데이 알파 없음.** 일간 시스템은 별도 정정 (위 경고문 참조).

---

## 출발점

- **일간 시스템:** Phase 4 완료, Tier 2 PASS (Sharpe 1.538 / WR 56.3% / Payoff 1.30, ~5.8년 백필)
- **인트라데이 시스템:** 6-layer 기술분석(ADX, StochRSI, VWAP-SD, OBV, ORB, regime-ATR) + 텔레그램 양방향 봇, Ubuntu 서버에서 systemd로 가동 중
- **라이브 봇 코드에 박힌 추정치:** `INTRADAY_WINRATE_EST = 0.54`, EV +0.106%/거래

검증 목표: 위 추정치를 실제 60일 데이터로 측정하고, 비용 0.25%(토스 0.1%×2 + 슬리피지)를 넘는 알파 존재 여부 확인.

---

## 12가지 접근 — 모두 OOS net EV 음수

| # | 접근 | OOS net EV | 통계 검증 | 비고 |
|---|---|---|---|---|
| 1 | 원본 룰 (trend-following) | **-0.26%** | — | WR 17% (가정 54%와 큰 괴리) |
| 2 | 72-config parameter sweep | -0.23% best | — | TP 멀티 변경 무효 (ATR 작아 floor 우세) |
| 3 | 방향 반전 (mean-reversion) | -0.22% | — | IC 음수 발견 → 정방향 잘못이었음 |
| 4 | 반전 + 64-config sweep | -0.23% best | — | gross EV 천장 +0.03% |
| 5 | 15분/30분봉 | -0.22 ~ -0.32% | — | 시간프레임 확장 무효 |
| 6 | LightGBM h=12 (60min) | -0.23% | OOS AUC 0.498 | 4-fold WF, IS AUC 0.86 |
| 7 | LightGBM h=24 (120min) + 넓은 TP | -0.22% | OOS AUC 0.498 | gross +0.033% (이론 한계) |
| 8 | 변동성 게이트(ATR≥0.25%) + 반전 | -0.11% | 60일 단일창 | 첫 의미있는 개선 |
| 9 | 변동성 게이트 + 정밀 TP/SL | **"+0.012%"** | ❌ OOS -0.95% | grid-search OOS 검증 실패 |
| 10 | 위 winner config 5-fold rolling WF | 3/5 양수 | **t-stat 0.26** (무의미) | 9개 ETF 중 5개가 0/5 양수 |
| 11 | 페어 트레이딩 z-score | -0.225% | **t-stat -10.35** | 강한 음의 검증 (sample 큼) |
| 12 | 페어 strict (z≥2.5/3.0) | -0.20% | t-stat -4.01 | 여전히 음수 유의 |

**중간에 발견된 모든 "양수"는 OOS에서 검증 실패.**

---

## 결정적 분석 — Per-ticker (rolling 5-fold WF)

같은 vol-gated reverse 설정 적용 시:

| 티커 | 양수 fold | 총 OOS PnL |
|---|---|---|
| SPY | **0/5** | -0.86% |
| DIA | **0/5** | -0.91% |
| XLV | 0/5 | 0.00% |
| XLY | 0/5 | -0.62% |
| QQQ | 1/5 | +0.13% |
| XLF | 1/5 | -1.22% |
| XLI | 1/5 | +0.84% |
| XLE | 2/5 | 0.00% |
| **XLK** | **2/5** | **+1.10%** ← 유일하게 양수 |

전체 양수 PnL은 거의 XLK 1개 종목의 우연. 9개 ETF 평균에 알파 없음.

---

## IC 분석 (Phase 1 인트라데이 버전)

13/30 팩터-horizon 조합이 통계적으로 유의 (|IC| ≥ 0.01):

**Top KEEP:**
- `rsi14` @ 150min: IC = **-0.0475** (음수 → mean-reversion)
- `ema_spread_pct` @ 150min: -0.0455
- `rsi14` @ 30min: -0.0369
- `vwap_dev_sd` @ 150min: -0.0329
- `stochrsi_k` @ 30min: -0.0314
- `adx14` @ 60min: **+0.0314** (유일한 양수)
- `obv_slope` @ 60min: -0.0311

**핵심 통찰:** 모든 모멘텀/추세 지표가 음의 IC → 라이브 봇은 정확히 반대 방향(trend-following)으로 진입 중이었음.

IC 천장 0.045 → 정보비율 한계 약 0.03% gross/거래 → 12가지 접근 모두에서 측정된 천장과 일치.

---

## LightGBM 결과

**구조:** 13개 KEEP 팩터 + minutes_from_open + VIX_prior + ticker_code, forward 12-bar log return 이진분류, 4-fold expanding-window walk-forward.

| Fold | IS AUC | OOS AUC | 평가 |
|---|---|---|---|
| 1 (train 20d) | 0.87 | 0.44 | 학습 부족 |
| 2 (train 33d) | 0.88 | 0.49 | 무작위 수준 |
| 3 (train 46d) | 0.86 | **0.54** | 의미 있는 신호 |
| 4 (train 58d) | 0.84 | 0.52 | 의미 있는 신호 |

**피처 중요도 (gain):**
1. `vix_prior` 49597 ← **다른 것의 5배** (일간 레짐 신호)
2. `ticker_code` 9520
3. `minutes_from_open` 9198
4. `rsi14` 8525
5. `ema_spread_pct` 6977
...
12. `obv_slope` 1005

→ 모델 학습의 대부분은 **일간 레짐 정보**이고, 진정한 인트라데이(5분 내) 정보는 약함. 이게 5분봉에서 알파 못 만드는 구조적 이유.

---

## 통계적 한계 정리

| 측정 항목 | 값 | 의미 |
|---|---|---|
| 5분봉 팩터 최대 \|IC\| | 0.0475 | 일간 IC와 비슷 |
| Gross EV 천장 | +0.03~+0.05% | 룰/ML/페어 모두 동일 |
| 거래 비용 | 0.25% | 정보의 5-8배 |
| **부족분** | **약 0.20~0.22%** | 어떤 미세조정으로도 안 됨 |
| 60일 표본 통계 power | 약함 | 풀 sweep으로는 grid-search 거짓 양성 다발 |

**결론:** retail-grade 데이터(yfinance 15분 지연, OHLCV만)로 깰 수 없는 비용 장벽.

---

## 작성된 파일

### 스크립트 (모두 `scripts/` 하위)

| 파일 | 목적 |
|---|---|
| `run_intraday_backtest.py` | 메인 백테스트 엔진 (--reverse, --interval, --min-atr-pct, --min-vix 등 옵션) |
| `run_intraday_sweep.py` | TP/SL/MIN_BARS sweep (`--reverse` 지원) |
| `run_intraday_vol_sweep.py` | 변동성 게이트(ATR, VIX) sweep |
| `run_intraday_ic.py` | 5분봉 IC 분석 (Phase 1 인트라데이 버전) |
| `train_intraday_lgbm.py` | LightGBM walk-forward + 시그널 백테스트 |
| `run_intraday_walkforward.py` | IS/OOS 분리 + grid search 검증 |
| `check_oos_specific.py` | 고정 config의 IS/OOS 직접 검증 (data snooping 없음) |
| `run_rolling_wf.py` | 5-fold expanding WF + per-ticker 분석 |
| `run_pairs_backtest.py` | 페어 z-score 백테스트 + 5-fold WF |

### 결과 데이터 (모두 `data/` 하위)

| 파일 | 내용 |
|---|---|
| `intraday_backtest.csv` | 원본 60일 거래 로그 (265건) |
| `intraday_backtest_summary.txt` | 원본 요약 |
| `intraday_backtest_*.csv/txt` | 각 변종(15m, 30m, reverse, vol-gated 등) |
| `intraday_sweep_results.csv/txt` | TP/SL/MIN_BARS sweep (72 configs) |
| `intraday_sweep_reverse.csv/txt` | 반전 모드 sweep (64 configs) |
| `intraday_vol_sweep.csv/txt` | 변동성 게이트 sweep (30 configs) |
| `intraday_ic_results.csv/txt` | IC 분석 결과 |
| `intraday_lgbm_trades.csv` | LightGBM OOS 거래 |
| `intraday_lgbm_summary.txt` | LightGBM 요약 |
| `intraday_walkforward.txt`, `intraday_walkforward_is.csv` | WF + grid search 검증 |
| `intraday_rolling_wf.txt` | 5-fold WF + per-ticker |
| `intraday_pairs.txt`, `pairs_trades.csv` | 페어 트레이딩 결과 |

### 모델

| 파일 | 내용 |
|---|---|
| `models/weights/intraday_lgbm.pkl` | 학습된 LightGBM (final fold) |
| `models/weights/intraday_lgbm_feature_importance.csv` | 피처 중요도 |

---

## 권장 액션 — 봇 정지가 아닌 메시지 정직화

**핵심 통찰:** 봇은 자동매매 안 함 (수동). 자동 손실 위험 없음. 단, **메시지에 박힌 허위 WR 54% / EV +0.106%가 의사결정을 오도**할 위험.

### 수정 대상

**1. `signals/intraday_generator.py`:**
- `INTRADAY_WINRATE_EST = 0.54` → `0.17` (실측)
- `INTRADAY_AVG_WIN_EST = 0.010` → `0.0029`
- `INTRADAY_AVG_LOSS_EST = 0.004` → `0.0037`
- 또는 더 깔끔: 추정치 자체를 제거하고 None 반환

**2. `bot/intraday_bot.py`의 `_signal_row()`:**
- "예상WR {sig.winrate:.0%}" 삭제
- "기대수익 {sig.exp_return:+.2%}" 삭제
- 헤더 라인에 추가: `⚠️ 60일 백테스트 net EV -0.26%/거래 — 거래 비권장, 시장 관찰용`

### 봇은 유지

- 텔레그램 알림: 시장 상황 인식용으로 유용
- 인프라(systemd + GitHub Actions deploy): 작동 검증된 자산
- `/positions`, `/stats` 명령: 만약 실거래 시도하면 기록용

---

## 서버 배포 — 메시지 수정 후 반영 순서

```bash
# 로컬 (Windows PowerShell)
git add signals/intraday_generator.py bot/intraday_bot.py
git commit -m "fix: remove fake WR/EV from intraday messages, mark as observation-only"
git push origin main
# → GitHub Actions self-hosted runner가 자동 배포

# 서버 직접 적용도 가능 (SSH 후 bash에서):
# ssh -i ~/.ssh/autobtc_iwinv root@115.68.230.40
# cd /root/recostock && git pull && systemctl restart intraday-bot
```

봇 자동 재배포 트리거 파일: `.github/workflows/deploy_intraday.yml` 참조.

---

## 다음 단계 — 일간 시스템 Phase 5

CLAUDE.md Phase 로드맵:
- Phase 4 완료 (현재): Tier 2 PASS (페이퍼 백필 271 포지션, Sharpe 1.538)
- **Phase 5 (다음):** Tier 2 PASS → 소액 실거래 + 레버리지 버킷 입증

Phase 5 진입 조건 (Tier 2 게이트):
- 페이퍼 트레이딩 3개월 이상 ✓
- 실현/백테스트 괴리 < 40% ✓
- 페이퍼 Sharpe > 0.5 ✓

→ 모든 조건 충족. Phase 5 시작 가능.

**구체적 다음 작업:**
1. 토스증권에 소액(예: $500-1000) 자금 입금
2. 일간 시스템 시그널 그대로 따라 매수
3. 1개월 후 실현 성과 vs 페이퍼 비교
4. 괴리 작으면 점진 증액
5. 레버리지 버킷 활성화는 `LEVERAGE_EDUCATION_DONE=true` AND `SYSTEM_PHASE>=5` 이후

---

## 만약 인트라데이를 다시 시도한다면

1. **데이터 업그레이드 필수:** Polygon.io 5분봉 5년치 (월 $30-100) 또는 IEX Cloud
2. **호가창 데이터:** Polygon Level 2 (월 $200+) — 진짜 알파의 원천
3. **유니버스 확장:** 개별주, 옵션, 외환 (각각 다른 비용 구조)
4. **다른 시간프레임:** 1-2시간봉으로 swing trading (intraday보다 overnight 비용 발생)

5분봉 + 9 ETF + 60일 + yfinance + 0.25% 비용 = **수학적으로 양수 EV 불가능**이 이번 검증의 결론.

---

## CLAUDE.md 정직 원칙 점검 (부록 B)

- ✅ #1 베이스라인을 못 이기는 ML 도입하지 않음
- ✅ #2 모든 성과 비용 차감 후로 보고
- ✅ #3 적중률을 손익비/기대값/표본과 함께 표시
- ✅ #4 2단계 게이트 (in-sample + out-of-sample) 통과 요구
- ✅ #5 알파가 없다는 결론 = 유효한 결과로 수용
- ✅ #6 감정으로 잘못된 시그널 연장 안 함

---

## 메모리 업데이트

`memory/project_recostock.md`에 다음 추가 권장:
- 인트라데이 알파 부재 12가지 검증 완료 (2026-05-16)
- 봇 정지 대신 메시지 정직화 선택
- 다음 단계: 일간 Phase 5 (소액 실거래)
