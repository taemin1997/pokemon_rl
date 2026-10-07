"""
목표 바꿔치기 시험 (GOAL1 핵심 진단): 같은 화면에 목표만 바꿔 넣으면 정책이 다른 방향을 고르나

왜 필요한가
- 양방향 도착률만 보면 "목표 정보를 써서" 됐는지, 다른 단서(시작 위치 차이 등)로 됐는지 구분이 안 됨
- 같은 게임 상태(화면 동일)에서 관찰의 goal만 바꿔 넣고 정책의 행동 확률을 비교하면 직접 보임
  - 목표 정보를 안 쓰는 정책: 두 목표에서 행동이 같음
  - 쓰는 정책: 각 목표의 거리표 기준 "맞는 방향"으로 바뀜
- 학습 없음, 게임을 진행하지도 않음 (상태 불러오기 → 관찰 한 번 → 정책 출력만) → 빠름

재는 것 (칸마다, 두 목표 각각)
- 맞는 방향 확률: 그 목표의 거리표에서 거리가 줄어드는 이웃 방향들에 정책이 준 확률 합
- 가장 높은 행동이 맞는 방향인가
- 바뀜: 두 목표에서 가장 높은 행동이 다른 비율
- 목표 정보가 없는 모델(LAT1 등)도 돌릴 수 있음 (그땐 관찰이 같아서 바뀜 0%가 기준선)

칸: 두 구간의 경로 밖 상태(offpath.pkl)에서 두 거리표에 다 있는 칸, 경로 거리 구간별로 골고루

실행
- uv run python -m train.goal_probe models/GOAL1_route29/final.zip
- 구간 쌍 바꾸기: --segments route29 route29_rev
"""

import argparse
import io
import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from envs.gold_env import ACTIONS, GoldEnv
from envs.offpath_tiles import load_offpath
from envs.segments import SEGMENTS
from train.diagnose import _load, policy_label
from train.train_ppo import load_env_config

DIR_STEP = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def correct_dirs(seg, pos, dist, adj):
    """pos에서 그 구간 거리표 기준 거리가 줄어드는 방향들 (턱은 edges에 2칸 이동으로 들어 있음)"""
    out = set()
    for nxt in adj.get(pos, []):
        if nxt in dist and dist[nxt] < dist[pos] and nxt[:2] == pos[:2]:
            dx = (nxt[2] > pos[2]) - (nxt[2] < pos[2])
            dy = (nxt[3] > pos[3]) - (nxt[3] < pos[3])
            for d, step in DIR_STEP.items():
                if step == (dx, dy):
                    out.add(d)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--segments", nargs=2, default=["route29", "route29_rev"])
    parser.add_argument("--per-band", type=int, default=20, help="경로 거리 구간당 칸 수")
    args = parser.parse_args()

    cfg = load_env_config(Path(args.model).parent)
    model = _load(args.model)
    segs = [SEGMENTS[n] for n in args.segments]
    dists = [{tuple(r[:4]): r[4] for r in json.loads(Path(s.dist_path).read_text())} for s in segs]
    adjs = []
    for s in segs:
        adj = {}
        for e in json.loads(Path(s.edges_path).read_text()):
            adj.setdefault(tuple(e[:4]), []).append(tuple(e[4:]))
        adjs.append(adj)
    envs = [GoldEnv(**{**s.env_kwargs(), "move_mode": cfg.get("move_mode", "tap"),
                       "obs_memory": cfg.get("obs_memory", False), "obs_goal": cfg.get("obs_goal", False)})
            for s in segs]

    # 칸 고르기: 첫 구간 경로 밖 상태 중 두 거리표에 다 있고, 두 목표 모두 "맞는 방향"이 있는 칸
    tiles = load_offpath(segs[0])
    rng = np.random.default_rng(0)
    bands = [(0, 0), (1, 2), (3, 5), (6, 99)]
    picks = []
    for lo, hi in bands:
        cand = [p for p, t in tiles.items() if lo <= t["path_dist"] <= hi and all(p in d for d in dists)
                and all(correct_dirs(s, p, d, a) for s, d, a in zip(segs, dists, adjs))]
        cand.sort()
        for i in rng.permutation(len(cand))[:args.per_band]:
            picks.append((f"{lo}-{hi}" if hi < 99 else f"{lo}+", cand[i]))

    rows = []
    for band, pos in picks:
        state = zlib.decompress(tiles[pos]["state"])
        out = {"band": band, "pos": list(pos)}
        argmaxes = []
        for k, (env, s, d, a) in enumerate(zip(envs, segs, dists, adjs)):
            env._load_start = lambda st=state, e=env: e.pyboy.load_state(io.BytesIO(st))
            obs, _ = env.reset()
            obs_t, _ = model.policy.obs_to_tensor(obs)
            with torch.no_grad():
                probs = model.policy.get_distribution(obs_t).distribution.probs[0].numpy()
            good = correct_dirs(s, pos, d, a)
            p_good = float(sum(probs[ACTIONS.index(g)] for g in good))
            top = ACTIONS[int(probs.argmax())]
            argmaxes.append(top)
            out[s.name] = {"p_correct": p_good, "top": top, "top_correct": top in good, "correct": sorted(good)}
        out["flipped"] = argmaxes[0] != argmaxes[1]
        rows.append(out)
    for env in envs:
        env.close()

    label = policy_label(args.model)
    print(f"\n{label} (목표 정보 {'있음' if cfg.get('obs_goal') else '없음'}), 칸 {len(rows)}개")
    print(f"{'경로 거리':>8}{'칸':>5}" + "".join(f"{s.name + ' 맞는 방향 확률/최고':>28}" for s in segs) + f"{'바뀜':>8}")
    summary = {"policy": label, "obs_goal": bool(cfg.get("obs_goal")), "bands": {}}
    for band in dict.fromkeys(b for b, _ in picks):
        rs = [r for r in rows if r["band"] == band]
        cols = []
        for s in segs:
            pc = float(np.mean([r[s.name]["p_correct"] for r in rs]))
            tc = float(np.mean([r[s.name]["top_correct"] for r in rs]))
            cols.append((pc, tc))
        flip = float(np.mean([r["flipped"] for r in rs]))
        summary["bands"][band] = {"tiles": len(rs), **{s.name: {"p_correct": c[0], "top_correct": c[1]}
                                                       for s, c in zip(segs, cols)}, "flipped": flip}
        print(f"{band:>8}{len(rs):>5}" + "".join(f"{c[0]:>20.0%} / {c[1]:<5.0%}" for c in cols) + f"{flip:>8.0%}")
    allc = {s.name: float(np.mean([r[s.name]["top_correct"] for r in rows])) for s in segs}
    summary["all"] = {**allc, "flipped": float(np.mean([r["flipped"] for r in rows]))}
    print(f"{'전체':>8}{len(rows):>5}  최고 행동이 맞는 방향: " +
          ", ".join(f"{k} {v:.0%}" for k, v in allc.items()) + f", 바뀜 {summary['all']['flipped']:.0%}")
    out = Path("runs/diagnose")
    out.mkdir(parents=True, exist_ok=True)
    stem = f"goalprobe_{'+'.join(args.segments)}_{time.strftime('%m%d_%H%M')}_{label}"
    (out / f"{stem}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    (out / f"{stem}_raw.json").write_text(json.dumps(rows, ensure_ascii=False))
    print(f"\n저장: {out / (stem + '_summary.json')}")


if __name__ == "__main__":
    main()
