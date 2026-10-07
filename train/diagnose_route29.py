"""
29번도로 진단 평가: 학습 없이, 이미 학습한 모델을 고정된 시작 위치에서 여러 번 돌려서 "어디서 왜 막히나"를 잼

왜 필요한가
- 새로 1시간 학습하기 전에, 가지고 있는 모델(v1, v2)로 가설을 먼저 걸러내면 실험 비용이 거의 0
- 학습 중 지표(ep_rew_mean 등)는 "보상을 얼마나 먹었나"라서, 실제 진행(목표까지 거리)과 다를 수 있음
- 같은 시작 상태 + 같은 측정 방식 → 앞으로의 모든 실험을 이 표 하나로 비교 가능

비교 대상
- 학습된 모델들 + random(무작위 행동)
- random을 같이 넣는 이유: 모델이 random보다 못하면 "못 배운 것"이 아니라 "피하는 걸 배운 것"

시작 위치 (train/build_route29_eval.py로 생성)
- S1_entry  : 29번도로 입구 (목표까지 90걸음)
- S2_deadend: 윗길 x=44 (76걸음). 막다른 곳처럼 보이지만 아래로 6칸 내려가면 (43,14) 틈으로 빠져나감
- S3_fork   : (53,9) (84걸음)

재는 것 (한 판 = 최대 512스텝)
- progress    : 시작 칸의 남은 거리 - 판 중 가장 가까웠던 거리 (클수록 목표 쪽으로 감)
- exit        : 풀숲 필드를 빠져나감 = 남은 거리 69 이하 칸 도달 ((43,14)를 지났다는 뜻)
- cherry      : 무궁시티 도착
- grass       : 풀숲 줄(29번도로 x44~53, y10~13)에 있던 스텝 비율
- battles     : 전투 수, 전투에 쓴 스텝
- back_to_town: 연두마을로 되돌아간 비율

실행: uv run python -m train.diagnose_route29 models/v1_explore/final.zip models/v2_ent003/final.zip random
옵션: --runs 25 (위치당 판 수), --steps 512, --auto-battle (전투 자동 처리, route29 커리큘럼 모델 평가용)
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from envs import memory_map as mm
from envs.gold_env import GoldEnv

STARTS = ["S1_entry", "S2_deadend", "S3_fork"]
EXIT_DIST = 69          # (43,14)를 지난 칸들의 거리. 이하면 필드 탈출
ROUTE29 = (24, 3)
TOWN = (24, 4)

_dist = None
_models = {}


def load_dist():
    rows = json.loads(Path("envs/data/route29_dist.json").read_text())
    return {tuple(r[:4]): r[4] for r in rows}


def in_grass_rows(pos):
    g, n, x, y = pos
    return (g, n) == ROUTE29 and 44 <= x <= 53 and 10 <= y <= 13


def run_one(args):
    """한 판 실행. 프로세스 여러 개가 나눠서 돌림"""
    policy, start, seed, max_steps, auto_battle = args
    global _dist
    if _dist is None:
        _dist = load_dist()

    # auto_battle: 학습을 전투 자동 처리로 했으면 평가도 같은 조건이어야 공정함
    env = GoldEnv(state_path=f"states/eval/{start}.state", max_steps=max_steps, auto_battle=auto_battle)
    rng = np.random.default_rng(seed)
    model = None
    if policy != "random":
        if policy not in _models:
            _models[policy] = PPO.load(policy, device="cpu")
        model = _models[policy]
        # 주의: PPO.load가 저장된 시드로 난수를 다시 맞춰버려서, 시드는 반드시 load 뒤에 설정
        # (안 그러면 판마다 똑같은 행동이 나와서 25판이 사실상 1판이 됨)
        torch.manual_seed(seed)

    obs, info = env.reset()
    # 같은 시작 상태라도 몇 프레임 흘려서 인카운터 타이밍이 판마다 달라지게 함
    env.pyboy.tick(int(rng.integers(0, 30)), False)

    d0 = _dist[info["position"]]
    best = d0
    exit_step = None
    grass = battles = battle_steps = 0
    in_battle = False
    reached_cherry = went_town = False
    exp0 = mm.read_mon1_exp(env.pyboy)

    for t in range(max_steps):
        if model is None:
            action = int(rng.integers(0, env.action_space.n))
        else:
            action, _ = model.predict(obs)
        obs, _, term, trunc, info = env.step(action)

        pos = info["position"]
        d = _dist.get(pos)
        if d is not None:
            best = min(best, d)
            if d <= EXIT_DIST and exit_step is None:
                exit_step = t + 1
        if pos[:2] == (26, 3):
            reached_cherry = True
        if pos[:2] == TOWN:
            went_town = True
        grass += in_grass_rows(pos)

        if auto_battle:
            battles = env.battles  # 자동 처리된 전투는 환경이 셈 (전투 스텝은 에이전트 스텝이 아니라 0)
        else:
            b = mm.in_battle(env.pyboy)
            battles += b and not in_battle
            battle_steps += b
            in_battle = b

        if reached_cherry or term or trunc:
            break

    result = {
        "policy": policy, "start": start,
        "progress": d0 - best, "exit": exit_step is not None, "exit_step": exit_step,
        "cherry": reached_cherry, "town": went_town,
        "grass_frac": grass / (t + 1), "battles": battles, "battle_steps": battle_steps,
        "exp": mm.read_mon1_exp(env.pyboy) - exp0, "steps": t + 1,
    }
    env.close()
    return result


def summarize(results):
    print(f"\n{'정책':<14}{'시작':<12}{'진행(평균/최대)':>16}{'탈출률':>8}{'탈출까지':>9}{'무궁':>6}{'마을복귀':>9}{'풀숲%':>7}{'전투/판':>8}{'전투스텝':>9}")
    rows = []
    for policy in dict.fromkeys(r["policy"] for r in results):
        for start in STARTS:
            rs = [r for r in results if r["policy"] == policy and r["start"] == start]
            if not rs:
                continue
            prog = [r["progress"] for r in rs]
            exits = [r["exit_step"] for r in rs if r["exit"]]
            name = "random" if policy == "random" else Path(policy).parent.name
            row = {
                "policy": name, "start": start,
                "progress_mean": float(np.mean(prog)), "progress_max": int(max(prog)),
                "exit_rate": len(exits) / len(rs),
                "exit_step_mean": float(np.mean(exits)) if exits else None,
                "cherry_rate": np.mean([r["cherry"] for r in rs]),
                "town_rate": np.mean([r["town"] for r in rs]),
                "grass_frac": np.mean([r["grass_frac"] for r in rs]),
                "battles": np.mean([r["battles"] for r in rs]),
                "battle_steps": np.mean([r["battle_steps"] for r in rs]),
            }
            rows.append(row)
            es = f"{row['exit_step_mean']:.0f}" if exits else "-"
            print(f"{name:<14}{start:<12}{row['progress_mean']:>10.1f} / {row['progress_max']:<3}"
                  f"{row['exit_rate']:>8.0%}{es:>9}{row['cherry_rate']:>6.0%}{row['town_rate']:>9.0%}"
                  f"{row['grass_frac']:>7.0%}{row['battles']:>8.2f}{row['battle_steps']:>9.0f}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policies", nargs="+", help="모델 .zip 경로들, 또는 random")
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--auto-battle", action="store_true", help="전투를 A 연타로 자동 처리 (route29 커리큘럼 조건)")
    args = parser.parse_args()

    jobs = [(p, s, seed, args.steps, args.auto_battle) for p in args.policies for s in STARTS for seed in range(args.runs)]
    with ProcessPoolExecutor(4) as ex:
        results = list(ex.map(run_one, jobs))

    rows = summarize(results)
    out = Path("runs/diagnose")
    out.mkdir(parents=True, exist_ok=True)
    # 전투 자동 처리 여부에 따라 파일을 나눔 (조건이 다른 결과가 서로 덮어쓰지 않게)
    tag = "_auto" if args.auto_battle else ""
    (out / f"route29{tag}_raw.json").write_text(json.dumps(results, default=float))
    (out / f"route29{tag}_summary.json").write_text(json.dumps(rows, default=float, ensure_ascii=False, indent=1))
    print(f"\n원자료: {out / f'route29{tag}_raw.json'}")


if __name__ == "__main__":
    main()
