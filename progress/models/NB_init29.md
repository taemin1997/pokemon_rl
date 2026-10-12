# NB_init29

> 모델 카드. 이 모델의 설정, 결과, 상태를 여기에 계속 갱신 (아래 '갱신 기록'에 날짜와 한 줄)

| 항목 | 내용 |
|---|---|
| 상태 | ⚠️ 7.4만 (프로세스 꺼짐) |
| 시작 | 2026-10-07 |
| 환경 / 커리큘럼 | `newbark_demo` |
| 보상 | goal_only |
| 바꾼 것 (직전 실험 대비) | 연두마을 구간, DEMO_TILE 가중치에서 |
| 설정 (env_config.json) | 조작 tile, 기억 X, 목표 X, 시드 0, 시작 가중치 models/DEMO_TILE_route29/final.zip |
| 결과 | 5만에 24% (scratch 100%), 끝까지 28% |
| 판정 / 메모 | 부정적 전이 + 잊음 (T1) |
| 파일 | `models/NB_init29/` (live.zip, ppo_49998_steps.zip) |
| 자세한 기록 | [notes/experiments/E17_CHAIN1.md](../../notes/experiments/E17_CHAIN1.md) |

## 갱신 기록
- 2026-10-08: 카드 생성 (기존 기록에서 옮김)
