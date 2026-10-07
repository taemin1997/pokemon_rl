"""
학습된 모델이 실제로 뭘 하는지 창으로 보기

왜 필요한가
- TensorBoard 숫자(탐험 칸 수 등)는 "얼마나"만 알려주고 "어떻게"는 안 알려줌
- 같은 탐험 칸 수라도 마을 밖으로 나간 건지, 집 안을 구석구석 훑은 건지는 봐야 앎
  → 4단계 notes 기록의 "행동 관찰" 칸은 이 스크립트로 채움

실행
- uv run python -m train.watch models/v1_explore/final.zip
- 가장 확률 높은 행동만:  ... --deterministic
"""

import argparse

from stable_baselines3 import PPO

from envs.gold_env import GoldEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--steps", type=int, default=2048)
    # 기본은 확률적 행동 (학습 때와 같은 방식)
    # CartPole에선 deterministic=True가 실력이었지만, 포켓몬에선 한 행동만 고르면
    # 벽에 대고 같은 버튼만 누르며 갇히는 경우가 많아서 둘 다 볼 수 있게 옵션으로 둠
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    env = GoldEnv(render_mode="human", max_steps=args.steps)
    model = PPO.load(args.model)

    obs, info = env.reset()
    total = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        done = terminated or truncated

    print(f"보상 합계 {total:.1f} | 탐험 칸 {info['visited_tiles']} | 맵 {info['visited_maps']} | 배지 {info['badges']}")
    env.close()


if __name__ == "__main__":
    main()
