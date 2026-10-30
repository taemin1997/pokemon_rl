# 통합 실행기(headless run_story)가 매크로 뒤 지난 화면으로 예측

- 날짜: 2026-10-20
- 분류: 버그

## 내용
envs/heal.py RENDER=False라 규칙·매크로가 화면을 안 그림 → 직후 model.predict가 매크로 전 화면을 봄 (판당 약 550번). 고치니 A_s1 낮 8 → 18/20, M2B 낮 10 → 0/20. 옛 판정 3 숫자 무효 (runs/story/INVALID_stale_obs.md). --fresh-obs 기본

## 갱신 기록
- 2026-10-21: 파일 생성
