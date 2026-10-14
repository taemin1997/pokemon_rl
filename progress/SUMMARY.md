# 정리 문서 색인

> 날짜마다, 모델마다, 특이사항마다 새 파일. 이 파일은 목록만

| 종류 | 위치 | 언제 새로 만드나 / 고치나 |
|---|---|---|
| **최신 요약** | [summary/latest.md](summary/latest.md) | 날마다 고쳐 씀 (결과, 비계, RL, 모델링 한 장) |
| 일별 정리 | [summary/daily/](summary/daily/) `YYYY-MM-DD.md` | 작업한 날마다 새 파일 |
| 주별 정리 | [summary/weekly/](summary/weekly/) `YYYY-Www.md` | 주가 바뀌면 새 파일 (월요일 시작) |
| 월별 정리 | [summary/monthly/](summary/monthly/) `YYYY-MM.md` | 달이 바뀌면 새 파일 |
| 작업 일지 | [log/](log/) `YYYY-MM-DD.md` | 날마다 새 파일. 학습 시작/끝, 검증, 특이사항, 사소한 것까지 시각과 한 줄 |
| 모델 카드 | [models/](models/README.md) `<모델 이름>.md` | 모델마다 새 파일 (학습 시작 때 만들고, 진단/판정마다 갱신) |
| 특이사항 | [events/](events/README.md) `YYYY-MM-DD_<요약>.md` | 버그, 발견, 환경 문제, 미해결마다 새 파일 |
| 실험 기록 (자세히) | [notes/experiments/](../notes/experiments/) `E<번호>_<이름>.md`, 목록 [notes/04_experiments.md](../notes/04_experiments.md) | 실험마다 새 파일 (가설과 판정 기준을 학습 전에) |
| 단계별 현황 | `step1_cartpole.md` … `step4_training.md`, `battle.md`, `segments.md` | 계속 갱신. 새 단계가 생기면 새 파일 |
| 끊겼을 때 다시 시작 | [RESUME.md](RESUME.md) | 학습/세션 재개 절차와 현재 E26 재개 지점 |
| notes 전체 목록 | [notes/README.md](../notes/README.md), M2a 스토리 경로·이벤트 [notes/07_m2a_story.md](../notes/07_m2a_story.md) | 새 notes 파일이 생길 때 |

## 지금 있는 파일 (10-14 기준)
- 최신 요약: [summary/latest.md](summary/latest.md) (기준일 2026-10-14)
- 일별: [10-02](summary/daily/2026-10-02.md), [10-05](summary/daily/2026-10-05.md), [10-06](summary/daily/2026-10-06.md), [10-07](summary/daily/2026-10-07.md), [10-08](summary/daily/2026-10-08.md), [10-09](summary/daily/2026-10-09.md), [10-10](summary/daily/2026-10-10.md), [10-11](summary/daily/2026-10-11.md), [10-13](summary/daily/2026-10-13.md), [10-14](summary/daily/2026-10-14.md)
- 주별: [2026-W40](summary/weekly/2026-W40.md) (09-28 ~ 10-04), [2026-W41](summary/weekly/2026-W41.md) (10-05 ~ 10-11, 내용은 10-08까지만 — 아직 안 고침)
- 월별: [2026-10](summary/monthly/2026-10.md) (진행 중, 내용은 10-08까지만 — 아직 안 고침)
- 일지: [10-07](log/2026-10-07.md), [10-08](log/2026-10-08.md), [10-09](log/2026-10-09.md), [10-10](log/2026-10-10.md), [10-11](log/2026-10-11.md), [10-12](log/2026-10-12.md), [10-13](log/2026-10-13.md), [10-14](log/2026-10-14.md) (그 전은 `README.md` 갱신 기록)
- 실험 기록: E01 ~ E27 ([notes/04_experiments.md](../notes/04_experiments.md)). 최근: [E23 M2A](../notes/experiments/E23_M2A.md), [E24 ZS30](../notes/experiments/E24_ZS30.md), [E25 NEWMAP](../notes/experiments/E25_NEWMAP.md), [E26 SELFREPAIR (1단계 완료)](../notes/experiments/E26_SELFREPAIR.md), [E27 M2B (구간 통과·통합 미달)](../notes/experiments/E27_M2B.md)
- 모델 카드 36개 ([models/README.md](models/README.md)), 특이사항 16개 ([events/README.md](events/README.md)) — `progress/make_docs.py`로 색인 다시 만듦
