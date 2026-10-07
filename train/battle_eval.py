"""
전투 진단 (전투 모듈 B0~): 전투 시작 상태들에서 전투 정책을 돌려 결과를 잼

왜 필요한가
- 길찾기는 train.diagnose(도착률)로 비교하듯, 전투도 정책끼리 같은 표로 비교해야 함
- B0: 지금 비계인 A 연타가 얼마나 하는지 = 앞으로 규칙(B2), RL(B3)이 넘어야 할 기준선
- 학습 보상과 무관한 실제 결과만 잼

재는 것 (전투 하나마다)
- 승리: 전투가 끝났고 경험치가 늘었음 / 전멸: 파티 HP 합이 0까지 내려감 / 시간 초과: 6000프레임 안에 안 끝남
- 턴 수: 메인 메뉴(싸우다/가방/포켓몬/도망)가 열린 횟수 (`battle_menu()`)
- 잃은 HP: 전투 전 파티 HP 합 − 전투 후 (전멸이면 전부)
- 프레임: 전투에 걸린 게임 시간 (60프레임 = 1초)

정책
- mash: A 연타 (GoldEnv._run_auto_battle과 같은 타이밍: A 4프레임 + 20프레임). 커서가 기본 위치라
  "싸우다 → 첫 번째 기술"만 계속 고름. 기술 2번, 도망, 교체는 절대 안 함
- 실행 층 매크로(envs/battle.py)를 쓰는 정책: 메인 메뉴가 열릴 때마다 묶은 행동 하나를 고름
  - move1 / move2: 늘 그 기술 (move1은 A 연타와 같은 결과여야 함 = 매크로 확인용)
  - run: 늘 도망
  - random: 기술(가진 것 중)/도망을 무작위
  - best: B2 규칙. 위력 × 명중 × 자속 × 상성이 가장 큰 기술 (ROM 표에서 읽음)

실행
- uv run python -m train.battle_eval route29 mash --runs 5
결과: 터미널 표 + runs/diagnose/battle_<구간>_<날짜시각>_<정책>_{summary,raw}.json
"""

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
from pyboy import PyBoy

from envs import battle
from envs import memory_map as mm

MAX_FRAMES = 6000   # GoldEnv 자동 전투와 같은 안전장치 (약 100초)


def mash_step(pb):
    """A 연타 한 번 (GoldEnv._run_auto_battle과 같은 타이밍)"""
    pb.button_press("a")
    pb.tick(4, False)
    pb.button_release("a")
    pb.tick(20, False)
    return 24


def _random_decide(rng):
    def decide(pb):
        n = battle.num_moves(pb)
        return [f"move{i + 1}" for i in range(n)][rng.integers(n)] if rng.random() < 0.8 else "run"
    return decide


# 매크로 정책: 판단 함수를 만드는 함수 (판마다 rng를 받아서)
DECIDERS = {
    "move1": lambda rng: (lambda pb: "move1"),
    "move2": lambda rng: (lambda pb: "move2"),
    "run": lambda rng: (lambda pb: "run"),
    "random": _random_decide,
    "best": lambda rng: battle.decide_best_move,   # B2 규칙: 기대 데미지 최대 기술 (ROM 표)
}
POLICIES = ["mash", *DECIDERS]


def run_battle(pb, state, policy, rng):
    pb.load_state(io.BytesIO(state))
    pb.tick(int(rng.integers(0, 30)), False)   # 난수(데미지, 급소, 선공) 타이밍을 판마다 다르게
    hp0 = mm.read_party_hp_total(pb)
    exp0 = mm.read_mon1_exp(pb)
    level0 = mm.read_byte(pb, mm.PARTY_MON1_LEVEL)
    enemy = (mm.read_byte(pb, mm.ENEMY_SPECIES), mm.read_byte(pb, mm.ENEMY_LEVEL))
    frames, turns, lowest = 0, 0, hp0
    rejected, actions = 0, []
    if policy == "mash":
        prev_menu = None
        while mm.in_battle(pb) and frames < MAX_FRAMES:
            menu = mm.battle_menu(pb)
            if menu == "main" and prev_menu != "main":
                turns += 1
            prev_menu = menu
            frames += mash_step(pb)
            lowest = min(lowest, mm.read_party_hp_total(pb))
    else:
        decide = DECIDERS[policy](rng)

        def watch(pb_):
            nonlocal lowest
            lowest = min(lowest, mm.read_party_hp_total(pb_))
            return decide(pb_)
        out = battle.run_battle(pb, watch)
        frames, turns, rejected, actions = out["frames"], out["turns"], out["rejected"], out["actions"]
        lowest = min(lowest, mm.read_party_hp_total(pb))
    # 전투가 끝나고 필드로 돌아오는 연출이 끝나게 조금 더 흘림
    pb.tick(60, False)
    whiteout = lowest == 0
    hp1 = 0 if whiteout else mm.read_party_hp_total(pb)
    won = not whiteout and frames < MAX_FRAMES and mm.read_mon1_exp(pb) > exp0
    # 도망: 전투가 끝났는데 이기지도 전멸하지도 않음
    fled = not won and not whiteout and frames < MAX_FRAMES
    return {
        "policy": policy, "enemy": list(enemy), "won": won, "whiteout": whiteout,
        "timeout": frames >= MAX_FRAMES, "fled": fled, "turns": turns, "frames": frames,
        "rejected": rejected, "actions": actions,
        "hp_before": hp0, "hp_lost": hp0 - hp1, "level_up": mm.read_byte(pb, mm.PARTY_MON1_LEVEL) > level0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("segment")
    parser.add_argument("policies", nargs="+", choices=POLICIES)
    parser.add_argument("--runs", type=int, default=5, help="전투 시작 상태당 판 수")
    args = parser.parse_args()

    folder = Path("states/battles") / args.segment
    index = json.loads((folder / "index.json").read_text())
    pb = PyBoy("roms/gold_rom.gb", window="null")
    pb.set_emulation_speed(0)
    results = []
    for policy in args.policies:
        for e in index:
            state = (folder / e["file"]).read_bytes()
            for seed in range(args.runs):
                r = run_battle(pb, state, policy, np.random.default_rng(seed))
                r["file"] = e["file"]
                results.append(r)
    pb.stop(save=False)

    rows = []
    print(f"\n전투 시작 상태 {len(index)}개 × {args.runs}판")
    print(f"{'정책':<10}{'판':>5}{'승리':>7}{'도망':>7}{'전멸':>7}{'시간초과':>9}{'턴':>6}{'초':>7}{'잃은 HP':>9}{'레벨업':>8}{'거절':>6}")
    for policy in args.policies:
        rs = [r for r in results if r["policy"] == policy]
        row = {"policy": policy, "runs": len(rs),
               "win": float(np.mean([r["won"] for r in rs])),
               "whiteout": float(np.mean([r["whiteout"] for r in rs])),
               "timeout": float(np.mean([r["timeout"] for r in rs])),
               "fled": float(np.mean([r["fled"] for r in rs])),
               "rejected": float(np.mean([r["rejected"] for r in rs])),
               "turns": float(np.mean([r["turns"] for r in rs])),
               "seconds": float(np.mean([r["frames"] for r in rs])) / 60,
               "hp_lost": float(np.mean([r["hp_lost"] for r in rs])),
               "level_up": float(np.mean([r["level_up"] for r in rs]))}
        rows.append(row)
        print(f"{policy:<10}{row['runs']:>5}{row['win']:>7.0%}{row['fled']:>7.0%}{row['whiteout']:>7.0%}{row['timeout']:>9.0%}"
              f"{row['turns']:>6.1f}{row['seconds']:>7.1f}{row['hp_lost']:>9.1f}{row['level_up']:>8.0%}{row['rejected']:>6.2f}")
        # 적별
        for enemy in sorted({tuple(r["enemy"]) for r in rs}):
            er = [r for r in rs if tuple(r["enemy"]) == enemy]
            print(f"  적 {enemy}: {len(er)}판, 승리 {np.mean([r['won'] for r in er]):.0%}, "
                  f"턴 {np.mean([r['turns'] for r in er]):.1f}, 잃은 HP {np.mean([r['hp_lost'] for r in er]):.1f}")

    out = Path("runs/diagnose")
    out.mkdir(parents=True, exist_ok=True)
    stem = f"battle_{args.segment}_{time.strftime('%m%d_%H%M')}_{'+'.join(args.policies)}"
    (out / f"{stem}_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    (out / f"{stem}_raw.json").write_text(json.dumps(results, ensure_ascii=False))
    print(f"\n저장: {out / (stem + '_summary.json')}")


if __name__ == "__main__":
    main()
