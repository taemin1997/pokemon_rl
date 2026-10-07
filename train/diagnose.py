"""
구간 진단 평가 (일반화 버전): 학습한 모델을 구간의 고정 시작 상태에서 여러 번 돌려서 실제 진행을 잼

왜 필요한가 (확장성)
- diagnose_route29.py는 29번도로 전용(풀숲 좌표, 탈출 기준이 하드코딩)이라 다른 구간에 못 씀
- 이 버전은 envs/segments.py의 구간 정의만 보고 동작 → 구간이 늘어도 같은 표로 비교 가능
- 학습 보상(ep_rew_mean)은 보상 설계마다 크기가 달라서 실험끼리 비교가 안 됨
  → 보상과 무관한 "실제 진행"(도착률, 걸음 수, 거리)으로만 비교

재는 것 (시작 상태별, 정책별)
- 도착률, 도착까지 평균 걸음 (최단 경로 걸음과 비교)
- 진행: 시작 칸의 남은 거리 − 판 중 가장 가까웠던 거리 (평균 / 최대)
- 이탈률: 구간 밖으로 나가서 실패한 비율
- 전투 수, 경험치 증가

실행
- uv run python -m train.diagnose route29 models/B_route29/final.zip models/A2_route29/final.zip random
- 이어 붙이기: uv run python -m train.diagnose newbark_cherrygrove "chain:newbark=models/NB/final.zip,route29=models/DEMO_TILE_route29/final.zip"
  (지금 맵이 속한 구간의 모델로 행동. 구간 경계에서 모델 교체)
- 옵션: --runs 25 (시작 상태당 판 수), --workers 4 (학습과 같이 돌릴 땐 2)
결과: 터미널 표 + runs/diagnose/<구간>_<날짜시각>_<정책들>_summary.json (실행마다 따로 저장)
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from envs.gold_env import GoldEnv
from envs.segments import SEGMENTS
from train.train_ppo import load_env_config

_models = {}


def parse_chain(policy):
    """'chain:newbark=models/A/final.zip,route29=models/B/final.zip' → [(구간 이름, 모델 경로), ...]

    이어 붙이기 정책: 지금 서 있는 맵이 어느 구간의 allowed_maps에 속하는지 보고 그 구간 모델로 행동을 고름
    = 사람이 정한 상위 계획(구간 순서) + 구간마다 RL 정책. 구간 경계에서 모델이 바뀜
    """
    return [tuple(part.split("=", 1)) for part in policy[len("chain:"):].split(",")]


def _load(path):
    if path not in _models:
        _models[path] = PPO.load(path, device="cpu")
    return _models[path]


def policy_label(policy):
    """표에 쓸 이름: final.zip이면 실험 이름, 체크포인트면 실험 이름 + 스텝 (같은 실험의 체크포인트끼리 비교할 때 안 섞이게)"""
    if policy == "random":
        return "random"
    if policy.startswith("chain:"):
        return "chain(" + "+".join(policy_label(path) for _, path in parse_chain(policy)) + ")"
    p = Path(policy)
    if p.stem == "final":
        return p.parent.name
    return f"{p.parent.name}@{p.stem.replace('ppo_', '').replace('_steps', '')}"


def run_one(job):
    segment, policy, start_name, start_path, seed = job
    seg = SEGMENTS[segment]
    # 조작 방식은 모델 폴더의 env_config.json을 따름 (학습과 같은 조건으로 평가). random은 tap
    chain = parse_chain(policy) if policy.startswith("chain:") else None
    first = chain[0][1] if chain else policy
    cfg = {} if policy == "random" else load_env_config(Path(first).parent)
    if chain:
        # 이어 붙이는 모델끼리 조작 방식, 관찰 모양이 같아야 같은 환경에서 돌릴 수 있음
        cfgs = [load_env_config(Path(path).parent) for _, path in chain]
        assert all(c.get("move_mode", "tap") == cfg.get("move_mode", "tap") and
                   c.get("obs_memory", False) == cfg.get("obs_memory", False) and
                   c.get("obs_goal", False) == cfg.get("obs_goal", False) for c in cfgs), "chain 모델 설정이 다름"
    move_mode = cfg.get("move_mode", "tap")
    env = GoldEnv(**{**seg.env_kwargs(), "state_path": start_path, "move_mode": move_mode,
                     "obs_memory": cfg.get("obs_memory", False), "obs_goal": cfg.get("obs_goal", False)})
    rng = np.random.default_rng(seed)

    model = None
    if chain:
        # 구간 이름 → (허용 맵, 모델)
        chain_models = [(set(map(tuple, SEGMENTS[name].allowed_maps)), _load(path)) for name, path in chain]
        model = chain_models[0][1]
        torch.manual_seed(seed)
    elif policy != "random":
        model = _load(policy)
        # PPO.load가 저장된 시드로 난수를 되돌리기 때문에 반드시 load 뒤에 시드 설정
        torch.manual_seed(seed)

    obs, info = env.reset()
    # 같은 시작 상태라도 프레임을 조금 흘려서 인카운터 타이밍이 판마다 달라지게 함
    env.pyboy.tick(int(rng.integers(0, 30)), False)

    done = False
    while not done:
        if chain:
            # 지금 맵이 속한 구간의 모델로 (어느 구간에도 없으면 직전 모델 유지)
            here = info["position"][:2]
            model = next((m for maps, m in chain_models if here in maps), model)
        if model is None:
            action = int(rng.integers(0, env.action_space.n))
        else:
            action, _ = model.predict(obs)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    result = {
        "policy": policy_label(policy),
        "move_mode": move_mode,
        "start": start_name,
        "goal": bool(info["reached_goal"]),
        "goal_step": info.get("goal_step"),
        "left": bool(info["left_area"]),
        "progress": info.get("progress", 0),
        "battles": info["battles"],
        "exp": info["exp_gain"],
        "whiteout": info.get("whiteouts", 0) > 0,
    }
    env.close()
    return result


def summarize(results, shortest):
    # 진행은 평균/최대보다 중앙값(p50)과 상위 10%(p90)가 더 믿을 만함
    # (최대는 운 좋은 한 판에 끌려가고, 평균은 아주 못한 몇 판에 끌려감)
    header = f"{'정책':<30}{'시작':<12}{'도착률':>7}{'도착 걸음':>10}{'진행 p50/p90/최대':>18}{'이탈':>6}{'전멸':>6}{'전투':>6}{'경험치':>7}"
    print(f"\n최단 경로(시작 상태 'start' 기준): {shortest}걸음\n{header}")
    rows = []
    for policy in dict.fromkeys(r["policy"] for r in results):
        for start in dict.fromkeys(r["start"] for r in results):
            rs = [r for r in results if r["policy"] == policy and r["start"] == start]
            if not rs:
                continue
            steps = [r["goal_step"] for r in rs if r["goal"]]
            prog = [r["progress"] for r in rs]
            row = {
                "policy": policy, "start": start, "runs": len(rs),
                "goal_rate": float(np.mean([r["goal"] for r in rs])),
                "goal_step_mean": float(np.mean(steps)) if steps else None,
                "progress_mean": float(np.mean(prog)), "progress_max": int(max(prog)),
                "progress_p50": float(np.percentile(prog, 50)), "progress_p90": float(np.percentile(prog, 90)),
                "left_rate": float(np.mean([r["left"] for r in rs])),
                "whiteout_rate": float(np.mean([r["whiteout"] for r in rs])),
                "battles": float(np.mean([r["battles"] for r in rs])),
                "exp": float(np.mean([r["exp"] for r in rs])),
            }
            rows.append(row)
            gs = f"{row['goal_step_mean']:.0f}" if steps else "-"
            print(f"{policy:<30}{start:<12}{row['goal_rate']:>7.0%}{gs:>10}"
                  f"{row['progress_p50']:>8.0f} / {row['progress_p90']:<3.0f}/ {row['progress_max']:<4}{row['left_rate']:>6.0%}{row['whiteout_rate']:>6.0%}"
                  f"{row['battles']:>6.1f}{row['exp']:>7.0f}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("segment", choices=SEGMENTS.keys())
    parser.add_argument("policies", nargs="+", help="모델 .zip 경로들, 또는 random")
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--workers", type=int, default=4, help="병렬 프로세스 수 (학습 중이면 2)")
    args = parser.parse_args()

    seg = SEGMENTS[args.segment]
    if not seg.ready():
        print(f"구간 '{args.segment}' 준비 안 됨 (시작 상태 또는 거리표 없음)")
        return

    jobs = [
        (args.segment, p, name, path, seed)
        for p in args.policies
        for name, path in seg.all_eval_states().items()
        for seed in range(args.runs)
    ]
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(run_one, jobs))

    # 최단 걸음: 거리표에서 시작 칸의 거리
    dist = {tuple(r[:4]): r[4] for r in json.loads(Path(seg.dist_path).read_text())}
    probe = GoldEnv(**seg.env_kwargs())
    _, info = probe.reset()
    shortest = dist.get(info["position"])
    probe.close()

    rows = summarize(results, shortest)
    # 진단할 때마다 따로 저장 (예전엔 같은 파일을 덮어써서 마지막 결과만 남았음)
    # 파일명: <구간>_<날짜시각>_<정책들>
    import time
    names = "+".join(dict.fromkeys(policy_label(p).split("@")[0] for p in args.policies))[:120]
    stem = f"{args.segment}_{time.strftime('%m%d_%H%M')}_{names}"
    out = Path("runs/diagnose")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    (out / f"{stem}_raw.json").write_text(json.dumps(results, ensure_ascii=False))
    print(f"\n저장: {out / f'{stem}_summary.json'}")


if __name__ == "__main__":
    main()
