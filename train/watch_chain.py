"""
이어 붙이기 시청: 구간 경계에서 모델을 바꿔 끼우며 한 판을 창으로 보여줌

왜 필요한가
- live_watch는 모델 하나만 봄. "집 1층 → 무궁시티"는 연두마을 모델 → 29번도로 모델로 바뀌어야 해서
  train.diagnose의 chain: 정책과 같은 방식(지금 맵이 속한 구간의 모델로 행동)을 창으로 보는 스크립트
- 화면 아래 터미널에 구간이 바뀌는 순간과 판 결과를 찍음

실행
- uv run python -m train.watch_chain newbark_cherrygrove "chain:newbark=models/NB_scratch/final.zip,route29=models/LAT1_route29/final.zip"
- 빠르게: --speed 6 / 판 수: --episodes 3 / 가장 확률 높은 행동만: --deterministic
- 끄기: 게임 창 닫기 또는 Ctrl+C
"""

import argparse
import signal
from pathlib import Path

import torch

from envs import memory_map as mm
from envs.gold_env import GoldEnv
from envs.map_names import map_name
from envs.segments import SEGMENTS
from train.diagnose import _load, parse_chain
from train.train_ppo import load_env_config

stop = False


def _on_ctrl_c(signum, frame):
    global stop
    stop = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("segment", choices=SEGMENTS.keys())
    parser.add_argument("chain", help='"chain:구간=모델,구간=모델" (train.diagnose와 같은 형식)')
    parser.add_argument("--speed", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    signal.signal(signal.SIGINT, _on_ctrl_c)

    chain = parse_chain(args.chain)
    cfg = load_env_config(Path(chain[0][1]).parent)
    seg = SEGMENTS[args.segment]
    env = GoldEnv(render_mode="human", **{**seg.env_kwargs(), "move_mode": cfg.get("move_mode", "tap"),
                                          "obs_memory": cfg.get("obs_memory", False),
                                          "obs_goal": cfg.get("obs_goal", False)})
    env.pyboy.set_emulation_speed(args.speed)
    models = [(name, set(map(tuple, SEGMENTS[name].allowed_maps)), _load(path)) for name, path in chain]

    try:
        for ep in range(args.episodes):
            torch.manual_seed(ep)
            obs, info = env.reset()
            current = None
            done = False
            while not done and not stop and getattr(env, "window_open", True):
                here = info["position"][:2]
                hit = next(((n, m) for n, maps, m in models if here in maps), None)
                if hit and hit[0] != current:
                    current = hit[0]
                    print(f"[{ep + 1}] {map_name(here)}: '{current}' 모델로 ({env.step_count}걸음째)")
                model = hit[1] if hit else model
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            if stop or not getattr(env, "window_open", True):
                break
            result = "도착" if info["reached_goal"] else ("전멸" if info["whiteouts"] else ("이탈" if info["left_area"] else "시간 초과"))
            print(f"[{ep + 1}] {result} — {env.step_count}걸음 (최단 128), 전투 {info['battles']}번, "
                  f"경험치 +{info['exp_gain']}, 남은 HP {mm.read_party_hp_total(env.pyboy)}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
