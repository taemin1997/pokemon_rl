"""
전투 시작 상태 모으기 (전투 모듈 B0 재료)

왜 필요한가
- 전투 모듈(실행 층 매크로 → 규칙 → RL)을 길찾기와 따로 만들고 평가하려면 "전투가 막 시작된 상태"가 여러 개 필요함
  (길찾기에서 세이브 상태로 구간을 잘랐던 것과 같은 이유: 매번 풀숲을 걸어 들어가면 시간이 대부분 걷기에 씀)
- 사람이 플레이해서 만들지 않고, 구간 안을 무작위로 걷다가 인카운터가 시작된 순간을 저장

방법
- 시작점: 경로 밖 상태(offpath.pkl)가 있으면 그 칸들, 없으면 시연 상태들 → 풀숲 칸이 골고루 섞임
- 거기서 1칸 조작(tile)으로 무작위로 최대 60걸음, 전투가 시작되면 그 순간 저장
- HP: 시작점 상태는 전투를 피해서 만든 거라 HP가 거의 가득(18~20) → 전투 시작 HP도 비슷

결과
- states/battles/<구간>/NNN.state + index.json (적 종/레벨/최대 HP, 내 HP, 위치)

실행
- uv run python -m train.build_battles route29 --n 30
"""

import argparse
import io
import json
import pickle
import zlib
from pathlib import Path

import numpy as np

from envs import memory_map as mm
from envs.gold_env import GoldEnv
from envs.offpath_tiles import offpath_file
from envs.segments import SEGMENTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("segment", choices=SEGMENTS.keys())
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    seg = SEGMENTS[args.segment]

    if offpath_file(seg).exists():
        starts = [zlib.decompress(t["state"]) for t in pickle.loads(offpath_file(seg).read_bytes())["tiles"].values()]
    else:
        starts = [zlib.decompress(z) for z in pickle.loads(Path(seg.demo_path).read_bytes())["states"]]

    # auto_battle=False: 전투가 시작돼도 처리하지 않고 그대로 둬야 그 순간을 저장할 수 있음
    env = GoldEnv(**{**seg.env_kwargs(), "auto_battle": False, "move_mode": "tile", "max_steps": 60})
    rng = np.random.default_rng(args.seed)
    out = Path("states/battles") / seg.name
    out.mkdir(parents=True, exist_ok=True)
    index, tries = [], 0
    while len(index) < args.n and tries < args.n * 50:
        tries += 1
        state = starts[rng.integers(len(starts))]
        env._load_start = lambda s=state: env.pyboy.load_state(io.BytesIO(s))
        env.reset()
        env.pyboy.tick(int(rng.integers(0, 60)), False)   # 인카운터 타이밍을 판마다 다르게
        for _ in range(60):
            _, _, terminated, truncated, info = env.step(int(rng.integers(0, 4)))
            if mm.in_battle(env.pyboy):
                buf = io.BytesIO()
                env.pyboy.save_state(buf)
                name = f"{len(index):03d}.state"
                (out / name).write_bytes(buf.getvalue())
                index.append({
                    "file": name, "position": list(info["position"]),
                    "enemy_species": mm.read_byte(env.pyboy, mm.ENEMY_SPECIES),
                    "enemy_level": mm.read_byte(env.pyboy, mm.ENEMY_LEVEL),
                    "enemy_maxhp": mm.read_enemy_hp(env.pyboy)[1],
                    "party_hp": mm.read_party_hp_total(env.pyboy),
                })
                break
            if terminated or truncated:
                break
    env.close()
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
    species = {}
    for e in index:
        species[(e["enemy_species"], e["enemy_level"])] = species.get((e["enemy_species"], e["enemy_level"]), 0) + 1
    print(f"[전투 시작 상태] {len(index)}개 ({tries}번 시도) → {out}")
    print(f"  (종 번호, 레벨): 개수 = {species}")
    print(f"  내 HP 합: {sorted(e['party_hp'] for e in index)}")


if __name__ == "__main__":
    main()
