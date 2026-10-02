"""
GoldEnv 점검 스크립트 (학습 전에 반드시 돌려볼 것)

왜 필요한가
- 환경이 틀리면 PPO는 에러 없이 몇 시간 돌고 "아무것도 못 배움"으로 끝남
  → 원인이 환경인지 학습인지 구분이 안 됨
- 그래서 학습을 붙이기 전에 아래 4가지를 따로 확인함
  1) Gym 규격 검사 (SB3의 check_env): 관찰 모양/타입, reset/step 반환값
  2) reset 재현성: 두 번 reset 하면 똑같은 화면에서 시작하는지
  3) 행동이 실제로 먹히는지: 방향키를 누르면 좌표가 바뀌는지
  4) 랜덤 행동 기준점 + 속도: CartPole의 "랜덤 24점"처럼 비교 기준을 만들고,
     초당 몇 스텝인지 재서 학습에 얼마나 걸릴지 예상함

실행 (프로젝트 루트에서, -m 으로 실행해야 `from envs ...` import가 됨)
- 창 없이 점검:          uv run python -m envs.check_gold_env
- 랜덤 에이전트 눈으로:  uv run python -m envs.check_gold_env --watch
"""

import sys
import time

import numpy as np
from stable_baselines3.common.env_checker import check_env

from envs.gold_env import ACTIONS, GoldEnv


def main():
    watch = "--watch" in sys.argv

    if watch:
        # 창 띄워서 랜덤 에이전트가 뭘 하는지 구경만 함
        env = GoldEnv(render_mode="human", max_steps=300)
        env.reset()
        done = False
        total = 0.0
        while not done:
            _, r, term, trunc, info = env.step(env.action_space.sample())
            total += r
            done = term or trunc
        print(f"랜덤 300스텝: 보상 {total:.1f}, info {info}")
        env.close()
        return

    env = GoldEnv()

    # 1) 규격 검사. 문제 있으면 여기서 에러/경고가 남
    check_env(env, warn=True)
    print("[1] check_env 통과")

    # 2) reset 재현성
    obs_a, info_a = env.reset()
    env.step(3)  # 아무 행동이나 해서 상태를 바꿔놓고
    obs_b, info_b = env.reset()
    same = np.array_equal(obs_a, obs_b) and info_a["position"] == info_b["position"]
    print(f"[2] reset 재현성: {'OK' if same else '다름!'}  시작 위치 {info_a['position']}")

    # 3) 방향키가 좌표를 바꾸는지. 벽에 막힐 수 있어서 4방향 다 시도
    moved = False
    for a in range(4):
        env.reset()
        before = env._get_info()["position"]
        for _ in range(2):  # 첫 입력은 방향 전환만 될 수 있어서 2번
            _, _, _, _, info = env.step(a)
        print(f"    {ACTIONS[a]:>5}: {before} → {info['position']}")
        moved |= info["position"] != before
    print(f"[3] 이동 반영: {'OK' if moved else '안 움직임!'}")

    # 4) 랜덤 기준점 + 속도
    n_steps = 1000
    env.reset()
    total = 0.0
    t0 = time.time()
    for _ in range(n_steps):
        _, r, term, trunc, info = env.step(env.action_space.sample())
        total += r
        if term or trunc:
            env.reset()
    sps = n_steps / (time.time() - t0)
    print(f"[4] 랜덤 {n_steps}스텝: 보상 {total:.1f}, 탐험 칸 {info['visited_tiles']}, 맵 {info['visited_maps']}")
    print(f"    속도 {sps:.0f} 스텝/초 → 100만 스텝 약 {1_000_000 / sps / 3600:.1f}시간 (환경 1개 기준)")

    env.close()


if __name__ == "__main__":
    main()
