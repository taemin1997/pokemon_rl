"""
E30 D군 상태 묶음: 기존 16구간 BFS 시연 경로 칸 중 옛 정책(M2B) 최빈 행동 확률 ≥ 0.8인 칸의 관찰

- 관찰은 E29와 같은 방법 (시계 lock, 상태 불러온 뒤 1프레임 그림, 목표 = 그 구간 목표)
- 뒤집힘(E29 K)은 학습 전엔 모르므로 '옛 정책이 확신한 칸'만 씀 (E29 K 규칙의 앞 절반)
- 출력: runs/e30/anchor_states.npz (screen, goal, segment, tile, old_max)

실행: POKE_CLOCK=lock uv run python -m train.e30_anchor_states
"""

import io
import os
import pickle
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from envs import clock
from envs.gold_env import GoldEnv
from envs.segments import SEGMENTS
from train.e29_critical import _unz, probs
from train.run_story import goal_vec
from train.selfrepair import M2B_PARTS

OLD = "models/M2B_multi/final.zip"
OUT = Path("runs/e30/anchor_states.npz")


def main():
    assert os.environ.get("POKE_CLOCK", "lock") == "lock"
    os.environ["POKE_CLOCK"] = "lock"
    old = PPO.load(OLD, device="cpu")
    screens, goals, segs, tiles, maxp = [], [], [], [], []
    total = 0
    for name in M2B_PARTS:
        seg = SEGMENTS[name]
        demo = pickle.load(open(seg.demo_path, "rb"))
        env = GoldEnv(**{**seg.env_kwargs(), "move_mode": "tile", "obs_goal": True, "fast_battle": True, "auto_event": True})
        env.reset()
        g = goal_vec(seg.goal_maps[0])
        kept = 0
        for pos, st in zip(demo["positions"], demo["states"]):
            env.pyboy.load_state(io.BytesIO(_unz(st)))
            clock.after_load(env.pyboy)
            env.pyboy.tick(1, True)
            o = {**env._get_obs(), "goal": g}
            p = probs(old, o)
            total += 1
            if p.max() >= 0.8:
                screens.append(o["screen"]); goals.append(g); segs.append(name); tiles.append(list(pos[:4])); maxp.append(p.max())
                kept += 1
        env.close()
        print(f"{name}: 시연 {len(demo['positions'])}칸 중 {kept}칸")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, screen=np.array(screens, dtype=np.uint8), goal=np.array(goals, dtype=np.float32),
                        segment=np.array(segs), tile=np.array(tiles), old_max=np.array(maxp, dtype=np.float32))
    print(f"저장: {OUT} ({len(screens)}/{total}칸)")


if __name__ == "__main__":
    main()
