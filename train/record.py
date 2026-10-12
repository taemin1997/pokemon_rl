"""
플레이 녹화: 모델 한 판을 MP4로 저장 + 영상 두 개 나란히 붙이기 (포트폴리오용)

왜 필요한가 (사용자 요청 10-08)
- 전당등록에 성공하면 "첫날 헤매던 PPO 모델 vs 마지막 모델"을 나란히 보여주는 영상을 만들 것
- 환경 코드가 계속 바뀌어서(조작 방식, 관찰, 리셋 처리 …) 몇 달 뒤엔 첫날 모델을 예전 조건 그대로 재생하기 어려울 수 있음
  → **이정표 모델은 그때그때 녹화해서 보관** (`videos/`, models/처럼 git엔 안 올라감 → 따로 백업)

녹화 방식
- 걸음마다 마지막 프레임 1장 (게임 시간 40프레임 / 걸음 → 15fps면 약 2.7배속). 자동 전투 중 화면은 건너뜀 (전투 횟수는 자막에)
- 화면 2배 확대 + 아래 자막 (모델, 걸음, 맵, 전투 수). 자막은 영어/숫자만 (OpenCV 기본 글꼴이 한글을 못 씀)

실행
- 한 판 녹화: uv run python -m train.record models/v1_explore/final.zip --env home --steps 2048 --label "Day 1: v1_explore (exploration reward)"
- 구간: uv run python -m train.record models/LAT1_route29/final.zip --env route29 --label "LAT1"
- 무작위 정책: ... random --env home
- 나란히: uv run python -m train.record --side-by-side videos/a.mp4 videos/b.mp4 --out videos/compare.mp4
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from envs import memory_map as mm
from envs.gold_env import GoldEnv
from envs.map_names import map_name
from train.diagnose import _load
from train.train_ppo import ENV_PRESETS, load_env_config

SCALE = 2
CAPTION_H = 56


def frame_with_caption(env, lines):
    rgb = env.pyboy.screen.ndarray[:, :, :3]
    img = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    cap = np.full((CAPTION_H, img.shape[1], 3), 20, dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(cap, line, (6, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([img, cap])


def ascii_map(group_number):
    # 맵 이름이 한글이라 자막엔 번호로 (예: 24:3)
    return f"map {group_number[0]}:{group_number[1]}"


def record(model_path, env_name, steps, label, out, fps, seed, deterministic):
    cfg = {} if model_path == "random" else load_env_config(Path(model_path).parent)
    kwargs = dict(ENV_PRESETS[env_name])
    if steps:
        kwargs["max_steps"] = steps
    env = GoldEnv(**{**kwargs, "move_mode": cfg.get("move_mode", "tap"), "obs_memory": cfg.get("obs_memory", False),
                     "obs_goal": cfg.get("obs_goal", False)})
    model = None if model_path == "random" else _load(model_path)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs, info = env.reset()
    first = frame_with_caption(env, [label, "step 0"])
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (first.shape[1], first.shape[0]))
    writer.write(first)
    done = False
    while not done:
        if model is None:
            action = int(rng.integers(0, env.action_space.n))
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        status = "GOAL!" if info.get("reached_goal") else ("left area" if info.get("left_area") else "")
        writer.write(frame_with_caption(env, [label, f"step {env.step_count}  {ascii_map(info['position'][:2])}  "
                                                     f"battles {info['battles']}  {status}"]))
    for _ in range(fps * 2):   # 마지막 화면 2초 유지
        writer.write(frame_with_caption(env, [label, f"END  step {env.step_count}  maps visited {info['visited_maps']}  "
                                                     f"{'GOAL' if info.get('reached_goal') else ''}"]))
    writer.release()
    env.close()
    print(f"저장: {out} ({env.step_count}걸음, 맵 {info['visited_maps']}개, 도착 {bool(info.get('reached_goal'))}, "
          f"마지막 위치 {map_name(info['position'][:2])})")


def side_by_side(a, b, out, fps):
    ca, cb = cv2.VideoCapture(str(a)), cv2.VideoCapture(str(b))
    fa, fb = None, None
    writer = None
    while True:
        ra, na = ca.read()
        rb, nb = cb.read()
        if not ra and not rb:
            break
        fa = na if ra else fa   # 짧은 쪽은 마지막 화면 유지
        fb = nb if rb else fb
        h = max(fa.shape[0], fb.shape[0])
        pad = lambda f: np.vstack([f, np.zeros((h - f.shape[0], f.shape[1], 3), np.uint8)]) if f.shape[0] < h else f
        frame = np.hstack([pad(fa), np.full((h, 8, 3), 255, np.uint8), pad(fb)])
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    writer.release()
    print(f"저장: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", help="모델 .zip 또는 random")
    parser.add_argument("--env", default="home", choices=list(ENV_PRESETS))
    parser.add_argument("--steps", type=int, default=None, help="판 길이 (기본: 환경 값)")
    parser.add_argument("--label", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--side-by-side", nargs=2, metavar=("A", "B"))
    args = parser.parse_args()
    if args.side_by_side:
        side_by_side(*args.side_by_side, Path(args.out or "videos/compare.mp4"), args.fps)
        return
    name = "random" if args.model == "random" else Path(args.model).parent.name
    out = Path(args.out or f"videos/{name}_{args.env}_{time.strftime('%m%d_%H%M')}.mp4")
    record(args.model, args.env, args.steps, args.label or name, out, args.fps, args.seed, args.deterministic)


if __name__ == "__main__":
    main()
