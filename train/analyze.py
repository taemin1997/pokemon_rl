"""
학습된 모델을 창 없이 여러 판 돌려서 "어느 맵에서 시간을 썼는지" 요약

왜 필요한가
- watch.py는 한 판을 눈으로 보는 거라 운이 섞이고, 2048스텝을 다 보기도 힘듦
- 이 스크립트는 여러 판을 빠르게 돌려서 맵별 체류 스텝 수와 이동 순서를 숫자로 보여줌
  → "연두마을에서 1800스텝을 썼다" 같은 근거로 다음 보상 설계를 정할 수 있음
- notes/04_experiments.md의 "행동 관찰" 칸을 채울 때 씀

실행
- uv run python -m train.analyze models/v1_explore/final.zip
- 판 수 바꾸기:  ... --episodes 8
"""

import argparse
from collections import Counter

from stable_baselines3 import PPO

from envs import memory_map as mm
from envs.gold_env import GoldEnv
from envs.map_names import map_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--episodes", type=int, default=4)
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    env = GoldEnv()  # 창 없음 → 빠름

    for ep in range(args.episodes):
        obs, info = env.reset()
        dwell = Counter()  # 맵별 머문 스텝 수
        route = []         # 맵 이동 순서 (같은 맵 연속은 하나로)
        done = False
        while not done:
            # 학습 때와 같은 확률적 행동. 판마다 결과가 달라서 여러 판을 보는 것
            action, _ = model.predict(obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            here = mm.read_position(env.pyboy)[:2]
            dwell[here] += 1
            if not route or route[-1] != here:
                route.append(here)

        print(f"[{ep}] 칸 {info['visited_tiles']} | 맵 {info['visited_maps']} | 배지 {info['badges']}")
        print("    체류: " + ", ".join(f"{map_name(m)} {n}" for m, n in dwell.most_common()))
        # 순서가 길면 앞부분만 (같은 문을 들락날락하면 끝없이 길어짐)
        shown = " > ".join(map_name(m) for m in route[:12])
        print(f"    순서: {shown}{' ...' if len(route) > 12 else ''}  (맵 이동 {len(route) - 1}번)")

    env.close()


if __name__ == "__main__":
    main()
