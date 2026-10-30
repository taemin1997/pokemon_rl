"""
E29 길목(뒤집힌 칸) 자동 추출 — 학습 없음, 관찰만

깨진 구간의 BFS 시연 경로 칸마다 옛·새 정책의 행동 확률을 재고 (불러온 뒤 1프레임 그린 관찰, E28 F2와 같음)
규칙 (notes/experiments/E29_CRITICAL.md, 결과 보기 전 고정):
  K      = 옛 최빈 확률 ≥ 0.8, 새 최빈 ≠ 옛 최빈, 새가 옛 최빈에 주는 확률 ≤ 0.5
  K_ctrl = 옛 최빈 확률 ≥ 0.8, 새 최빈 = 옛 최빈 인 칸 중 |K|개 (시드 0으로 뽑음)
출력: runs/e29/<사례>.json (K·K_ctrl 칸 = [맵 그룹, 맵 번호, x, y], 칸별 확률, 시연 남은 거리)

실행: POKE_CLOCK=lock uv run python -m train.e29_critical
"""

import io
import json
import os
import pickle
import zlib
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from envs import clock
from envs.gold_env import GoldEnv
from envs.segments import SEGMENTS
from train.run_story import goal_vec

OLD = "models/M2B_multi/final.zip"
CASES = {"K1": ("route29", "models/A_E26_s0_r1/final.zip"),
         "K2": ("route29", "models/A_E26_s1_r1/final.zip"),
         "K3": ("cherrygrove", "models/A_E26_s0_r2/final.zip")}
OUT = Path("runs/e29")


def _unz(b):
    try:
        return zlib.decompress(b)
    except Exception:
        return b


def probs(model, obs):
    t, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        return model.policy.get_distribution(t).distribution.probs[0].numpy()


def main():
    assert os.environ.get("POKE_CLOCK", "lock") == "lock", "길목은 시계 고정(lock) 관찰로 정함"
    os.environ["POKE_CLOCK"] = "lock"
    OUT.mkdir(parents=True, exist_ok=True)
    old = PPO.load(OLD, device="cpu")
    for case, (segname, new_path) in CASES.items():
        seg = SEGMENTS[segname]
        new = PPO.load(new_path, device="cpu")
        demo = pickle.load(open(seg.demo_path, "rb"))
        env = GoldEnv(**{**seg.env_kwargs(), "move_mode": "tile", "obs_goal": True, "fast_battle": True, "auto_event": True})
        env.reset()
        g = goal_vec(seg.goal_maps[0])
        rows = []
        for i, (pos, st) in enumerate(zip(demo["positions"], demo["states"])):
            env.pyboy.load_state(io.BytesIO(_unz(st)))
            clock.after_load(env.pyboy)
            env.pyboy.tick(1, True)
            o = {**env._get_obs(), "goal": g}
            po, pn = probs(old, o), probs(new, o)
            rows.append({"i": i, "tile": list(pos[:4]), "dist": int(demo["dist"][i]), "old": po.round(3).tolist(), "new": pn.round(3).tolist()})
        env.close()
        conf = [r for r in rows if max(r["old"]) >= 0.8]
        K = [r for r in conf if int(np.argmax(r["new"])) != int(np.argmax(r["old"])) and r["new"][int(np.argmax(r["old"]))] <= 0.5]
        same = [r for r in conf if int(np.argmax(r["new"])) == int(np.argmax(r["old"]))]
        pick = np.random.default_rng(0).choice(len(same), size=min(len(K), len(same)), replace=False)
        K_ctrl = [same[j] for j in sorted(pick)]
        rec = {"case": case, "segment": segname, "old": OLD, "new": new_path, "rule": "old max ≥ 0.8, new argmax ≠ old argmax, new[old argmax] ≤ 0.5",
               "K": [r["tile"] for r in K], "K_ctrl": [r["tile"] for r in K_ctrl],
               "K_min_dist": min((r["dist"] for r in K), default=None), "n_demo": len(rows), "n_confident": len(conf), "rows": rows}
        (OUT / f"{case}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        print(f"{case} {segname}: 시연 {len(rows)}칸, 옛 확신 {len(conf)}칸, 뒤집힌 K {len(K)}칸 "
              f"{[tuple(r['tile'][2:]) for r in K]} (최소 남은 거리 {rec['K_min_dist']}), 대조 {[tuple(r['tile'][2:]) for r in K_ctrl]}")


if __name__ == "__main__":
    main()
