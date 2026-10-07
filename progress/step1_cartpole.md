# 1단계: CartPole로 강화학습 기본 루프

**상태**: ✅ 완료

## 목표
포켓몬에 들어가기 전에, "되는 게 확실한 문제"에서 상태 → 행동 → 보상 루프와 PPO가 제대로 도는지 확인
→ 나중에 포켓몬에서 학습이 안 될 때 "PPO 세팅 문제는 아님"을 빼고 생각할 수 있게

## 완료한 것
- [x] 랜덤 행동 루프 (`step1_cartpole/01_random_loop.py`): 5판 평균 약 24점
- [x] PPO 5만 스텝 학습 → 평가 → 저장 → 창으로 확인 (`step1_cartpole/02_ppo_train.py`)
  - 평가 20판 **500.0 ± 0.0** (만점)
- [x] TensorBoard 로그 (`runs/cartpole`)

## 여기서 익힌 것 (4단계에서 그대로 다시 씀)
- `reset → step → (obs, reward, terminated, truncated, info)` 규격
- 학습할 땐 렌더링 끄기, 평가는 `deterministic=True`로 따로
- `seed` 고정, `tensorboard_log`로 곡선 남기기

## 부족한 점
| 우선순위 | 내용 | 왜 필요한가 |
|---|---|---|
| 낮음 | `notes/01_cartpole.md`가 없음 (결과가 대화에만 있음) | 포폴에서 "출발점"으로 한 줄 쓸 때 근거 |

## 관련 파일
- `step1_cartpole/01_random_loop.py`, `step1_cartpole/02_ppo_train.py`
