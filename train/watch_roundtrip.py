"""
왕복 시청: 정책 하나로 29번도로 입구 → 무궁시티 → (목표 바꿈) → 연두마을을 한 판에

왜 필요한가
- GOAL2는 같은 도로를 목표에 따라 양방향으로 가는 정책. 스토리의 왕복 구간이 이 형태라, 한 판 안에서
  목표가 바뀌어도 이어서 가는지를 눈으로 확인
- 다음 단계(구간 통합: 중간 목표 달성 ≠ 판 종료)의 미리보기 겸 사전 점검

동작
1. 29번도로 입구(S1)에서 목표 = 무궁시티로 시작
2. 무궁시티 맵에 들어서면 목표를 연두마을로 바꿈
   - 비계: GOAL2는 무궁시티 안 화면을 본 적이 없음(도착 = 판 종료였음) → 무궁시티에 있는 동안은
     규칙으로 "오른쪽"을 눌러 29번도로로 되돌아감 (상위 계획의 구간 전환 역할). 화면/터미널에 표시
3. 연두마을에 들어서면 끝. 전투는 학습과 같이 자동

실행
- uv run python -m train.watch_roundtrip models/GOAL2_route29/final.zip
- 빠르게: --speed 6 / 판 수: --episodes 3 / 가장 확률 높은 행동만: --deterministic
- 창 없이 성공률만: --headless --episodes 10
- 집 1층부터 (연두마을 → 무궁시티 → 연두마을): --from-house models/NB_scratch/final.zip
"""

import argparse
import signal
from pathlib import Path

import numpy as np
import torch

from envs import memory_map as mm
from envs.gold_env import ACTIONS, GOAL_GROUPS, GOAL_NUMBERS, GoldEnv
from envs.map_names import map_name
from envs.segments import SEGMENTS
from train.diagnose import _load
from train.train_ppo import load_env_config

stop = False


def _on_ctrl_c(signum, frame):
    global stop
    stop = True


def goal_vec(m):
    v = np.zeros(GOAL_GROUPS + GOAL_NUMBERS, dtype=np.float32)
    v[m[0]] = 1.0
    v[GOAL_GROUPS + m[1]] = 1.0
    return v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--speed", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--headless", action="store_true", help="창 없이 (왕복 성공률 재기용)")
    # 집 1층에서 출발: 집/연두마을 구간은 이 모델(목표 정보 없는 모델도 됨)로, 29번도로부터 목표 조건 모델로
    parser.add_argument("--from-house", default=None, help="연두마을 구간 모델 (예: models/NB_scratch/final.zip). 주면 집 1층에서 출발")
    args = parser.parse_args()
    signal.signal(signal.SIGINT, _on_ctrl_c)

    cfg = load_env_config(Path(args.model).parent)
    assert cfg.get("obs_goal"), "목표 정보(--goal)로 학습한 모델이어야 함"
    model = _load(args.model)
    route, cherry, newbark, house = (24, 3), (26, 3), (24, 4), (24, 6)
    nb_model = _load(args.from_house) if args.from_house else None
    start = "states/start.state" if nb_model else SEGMENTS["route29"].start_state
    # 판 종료 조건을 직접 다룸: 허용 맵 = 29번도로 + 무궁시티 + 연두마을, 목표 맵 없음 (도착해도 안 끝나게)
    env = GoldEnv(render_mode=None if args.headless else "human", state_path=start, max_steps=args.max_steps,
                  auto_battle=True, allowed_maps=[route, cherry, newbark, house], goal_maps=[cherry],
                  move_mode=cfg.get("move_mode", "tile"), obs_memory=cfg.get("obs_memory", False), obs_goal=True)
    env.goal_maps = set()   # 무궁시티 도착으로 판이 끝나지 않게
    env.pyboy.set_emulation_speed(0 if args.headless else args.speed)

    try:
        for ep in range(args.episodes):
            torch.manual_seed(ep)
            obs, info = env.reset()
            env.goal_vec = goal_vec(cherry)
            obs = env._get_obs()
            phase, steps_out, steps_back, scripted = ("집 → 29번도로" if nb_model else "가는 중"), None, None, 0
            print(f"\n[{ep + 1}] 출발: {'집 1층' if nb_model else '29번도로 입구'}, 목표 = 무궁시티")
            done = False
            while not done and not stop and getattr(env, "window_open", True):
                here = info["position"][:2]
                if phase == "집 → 29번도로" and here == route:
                    phase = "가는 중"
                    print(f"[{ep + 1}] 29번도로 진입 ({env.step_count}걸음). 이제 목표 조건 모델로")
                if phase == "가는 중" and here == cherry:
                    phase, steps_out = "무궁시티 → 되돌아가기 (규칙)", env.step_count
                    env.goal_vec = goal_vec(newbark)
                    print(f"[{ep + 1}] 무궁시티 도착 ({steps_out}걸음). 목표 = 연두마을로 바꿈, 29번도로로 나올 때까지 규칙으로 오른쪽")
                if phase.startswith("무궁시티") and here == route:
                    phase = "돌아오는 중"
                    print(f"[{ep + 1}] 29번도로 복귀. 이제 정책이 연두마을로")
                if here == newbark and phase == "돌아오는 중":
                    steps_back = env.step_count - steps_out
                    print(f"[{ep + 1}] 연두마을 도착! 갈 때 {steps_out}걸음 + 올 때 {steps_back}걸음 "
                          f"(규칙으로 누른 걸음 {scripted}), 전투 {info['battles']}번, 남은 HP {mm.read_party_hp_total(env.pyboy)}")
                    break
                if phase.startswith("무궁시티"):
                    action = ACTIONS.index("right")
                    scripted += 1
                elif phase == "집 → 29번도로":
                    action, _ = nb_model.predict(env._get_obs()["screen"], deterministic=args.deterministic)
                else:
                    obs = env._get_obs()
                    action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            else:
                if not stop and getattr(env, "window_open", True):
                    why = "전멸" if info["whiteouts"] else ("시간 초과" if env.step_count >= args.max_steps else "다른 맵으로 이탈")
                    print(f"[{ep + 1}] 실패 ({why}) — {phase}, {env.step_count}걸음, 마지막 위치 {map_name(info['position'][:2])}")
            if stop or not getattr(env, "window_open", True):
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
