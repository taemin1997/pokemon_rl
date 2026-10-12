# A_NM1B_route30rev_seed1

> 모델 카드. 이 모델의 설정, 결과, 상태를 여기에 계속 갱신 (아래 '갱신 기록'에 날짜와 한 줄)

| 항목 | 내용 |
|---|---|
| 상태 | ✅ 200k |
| 시작 | 2026-10-09 |
| 환경 / 커리큘럼 | `route30_rev_demo_ge` |
| 보상 | goal_only |
| 바꾼 것 (직전 실험 대비) | [A] NM1-B, PPO 시드 1 (같은 Go-Explore 경로) |
| 설정 (env_config.json) | 조작 tile, 기억 X, 목표 O, 시드 1, 시작 가중치 models/GOAL2_route29/final.zip |
| 결과 | 고정 출발 100% (5만부터), 경로 밖 100/88/84/91% (전체 91%) |
| 판정 / 메모 | 판정: final 고정 출발 ≥ 80% (재현) |
| 파일 | `models/A_NM1B_route30rev_seed1/` (final.zip, ppo_149994_steps.zip, ppo_199992_steps.zip, ppo_49998_steps.zip, ppo_99996_steps.zip) |
| 자세한 기록 | `notes/04_experiments.md`의 A_NM1B_route30rev_seed1 블록 |

## 갱신 기록
- 2026-10-08: 카드 생성 (기존 기록에서 옮김)
