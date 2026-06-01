# 리포트 시각화 개선 계획

대상: `report/templates/daily-signal-report-template.html` (GitHub Pages 일일 리포트) + `bot/notifier.py` (텔레그램 푸시).

## 설계 제약 (불변)
- **의존성 0**: 외부 차트 라이브러리 금지. GitHub Pages 정적 서빙 + 오프라인 + 단일 주입 JS 객체(`REPORT`) 구조 유지 → 시각화는 **인라인 SVG/CSS**로만.
- **데이터 컨트랙트 결합**: `report/builder.py:_signal_to_dict()` 및 `build_report`의 `REPORT` 형태가 템플릿 렌더러와 결합. 새 데이터가 필요한 시각화는 **builder + 템플릿 양쪽**을 함께 수정해야 함.
- **정직성(부록 B)**: 표시 수치는 `signals/sector_rotation.py`의 `BACKTEST` dict / 재현 스윕(`data/logs/blend_goal_sweep.csv`)에서만. 템플릿에 수치를 새로 하드코딩하지 않는다.

## 완료 (2026-06-01)
- **텔레그램 메시지 재정렬**: 헤더 → **오늘의 포지션(액션, 최상단으로 이동)** → 개별 시그널 → 선택 레이어(🛰️) → 검증·실험(🧪 그룹) → 맥락 참고(🧭 RSI순 · 📋 conviction) → 푸터. `📋` 중복 제거(워치리스트는 🧭). 모든 접근 `.get()` 가드 유지.
- **HTML 배분 막대(allocation bar)**: '오늘의 포지션' 패널에 가중치 누적 막대 + 범례(SPXL 3x 표시). 기존 텍스트 배분을 시각화.
- **HTML RSI 막대**: 섹터 로테이션 표의 RSI-14를 0~100 미니 막대로(50 기준 색상).
- **정직성 수정**: 백테스트 설명문의 stale `슬리브 25%`→`15%`, 재현 안 된 `(슬리브 25%→ +131%/1.30)` 제거(실제 +124%/1.26·MDD 악화 명시). 죽은 코드(`newSpyW`) 제거.

## 로드맵 (우선순위)

### P1 — 높은 가치
1. **NAV 자본곡선 스파크라인** (페이퍼 검증 패널). 현재는 누적/Sharpe 숫자만. 인라인 SVG polyline로 NAV 추이.
   - **컨트랙트 변경 필요**: `portfolioPaper`가 `data/paper/portfolio_nav.parquet`의 NAV 시계열(예: `navSeries: [{date, nav}]`)을 노출하도록 `run_daily`/`portfolio_tracker.metrics()` 확장.
2. **유효노출 게이지**: `effExposure`(예 1.11x)를 1.0x 기준선 대비 막대/게이지로. 기존 데이터로 가능(컨트랙트 변경 없음).

### P2 — 중간
3. **섹터 RSI 히트-스트립**: 6개 섹터 RSI를 한 줄 히트바로(현재는 표). 기존 `sectorSatellite.ranked` 사용.
4. **레짐 타임라인**: 최근 N일 추세-on/off·캄-불 부스트 상태를 작은 띠로. **컨트랙트 변경 필요**(regime 이력 노출).
5. **팩터 기여 다이버징 바**: 시그널 카드의 팩터 막대를 +/− 양방향 다이버징으로(현재 절대값 단방향).

### P3 — 마감
6. 모바일 반응형 점검(표 가로 스크롤), 인쇄 스타일, 접근성(대비/aria).

## 검증 루틴 (변경 시 필수)
- `pytest tests/bot tests/` (notifier 계약 + 회귀).
- 노드 DOM-스텁 렌더 하니스로 템플릿 JS를 실데이터형 `REPORT`에 대해 **실행**해 런타임 에러/시각화 누락 확인(이번 변경에 사용).
- 컨트랙트 변경 시 `report/builder.py`와 템플릿을 **동시** 수정.
