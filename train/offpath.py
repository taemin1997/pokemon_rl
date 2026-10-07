"""
경로 밖 진단: "시연 경로 한 줄을 외웠나, 길 찾기를 배웠나"를 재는 도구

왜 필요한가
- DEMO_TILE은 학습 중 모든 판이 시연 경로 위에서 시작함. 진단(start/S2/S3)도 경로 위나 바로 옆
  → 경로 위 96%가 "길 찾기"인지 "줄 따라 걷기"인지 지금 지표로는 구분이 안 됨
- 근거: 29번도로 모델을 집 1층에 넣으면 진행 p50 2 (무작위 5보다 못함), 막다른 곳에서 같은 벽 47번 연속
- 앞으로 모든 모델(경로 옆 시작, 기억, 목표 정보 …)을 같은 지표로 비교하려고 따로 만듦

재는 것 세 가지
1. on-path: 시연 경로 위 칸에서 시작
2. off-path: 경로에서 k걸음 떨어진 칸에서 시작 (k 구간별)
   - 혼동 변수: 멀리 떨어진 칸은 목표까지도 멀거나 지형이 어려울 수 있음
   → off-path 칸마다 "목표까지 BFS 거리가 가장 비슷한 경로 위 칸"을 짝(대조군)으로 같이 돌림
     같은 목표 거리에서 경로 위 vs 밖의 차이 = 경로에서 떨어진 효과
3. perturbation recovery: 입구에서 정상 출발 → 목표까지 남은 거리가 T 이하가 되는 순간
   k걸음을 강제로 경로 밖으로 끌어냄 (경로에서 멀어지는 방향을 규칙으로 고름) → 다시 정책에게 → 도착하나
   "실수해서 벗어나도 돌아오나". 끌어낸 그 자리는 정책이 스스로 만든 상태라 off-path 시작과 성격이 다름

경로에서의 거리 (path_dist)
- 시연 경로 칸들에서 "앞으로 걸어서" 몇 걸음 만에 그 칸에 닿나 (BFS. 턱은 한 방향이라 거리표 edges를 그대로 씀)
- 상태는 시연 상태에서 실제로 걸어가서 만듦 (전투를 피하는 타이밍으로 → HP는 시연과 같게 유지)

실행
- 상태 만들기 (한 번):  uv run python -m train.offpath build route29
- 진단:                 uv run python -m train.offpath eval route29 models/DEMO_TILE_route29/final.zip models/B2_route29/final.zip
- 교란 복구:            uv run python -m train.offpath perturb route29 models/DEMO_TILE_route29/final.zip models/B2_route29/final.zip
- 벽 밀기 측정:         uv run python -m train.offpath stuck route29 models/LAT1_route29/final.zip --workers 2
결과: 터미널 표 + runs/diagnose/offpath_<구간>_<날짜시각>_<정책들>_{summary,raw}.json
"""

import argparse
import json
import pickle
import time
import zlib
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from envs import memory_map as mm
from envs.gold_env import ACTIONS, GoldEnv
from envs.offpath_tiles import BANDS, band_name, in_holdout, load_offpath, offpath_file, sample_starts
from envs.segments import SEGMENTS
from train.build_segment import DIRS, SETTLE, Explorer
from train.diagnose import _load, policy_label
from train.train_ppo import load_env_config

MAX_RADIUS = 14   # 이보다 멀리는 상태를 안 만듦 (29번도로 폭을 생각하면 충분)


# ---------------------------------------------------------------
# 1. 상태 만들기
# ---------------------------------------------------------------
def build(seg):
    """시연 경로 칸들에서 BFS로 걸어 나가며 칸마다 상태 하나씩 저장 (전투 피하는 타이밍)"""
    demo = pickle.loads(Path(seg.demo_path).read_bytes())
    dist = {tuple(r[:4]): r[4] for r in json.loads(Path(seg.dist_path).read_text())}
    area = set(map(tuple, seg.allowed_maps))

    ex = Explorer(seg.start_state)
    states, path_dist, nearest = {}, {}, {}
    queue = deque()
    for i, (pos, sz) in enumerate(zip(demo["positions"], demo["states"])):
        pos = tuple(pos)
        if pos not in states:
            states[pos] = zlib.decompress(sz)
            path_dist[pos] = 0
            nearest[pos] = i          # 가장 가까운 시연 칸 번호 (분석용)
            queue.append(pos)
    t0 = time.time()
    failed = 0
    while queue:
        cur = queue.popleft()
        if path_dist[cur] >= MAX_RADIUS:
            continue
        for d in DIRS:
            # settle=SETTLE: 걸음 뒤 늦게 시작되는 인카운터까지 피함 (--demo와 같은 조건 → HP 유지)
            nxt = ex.try_move(states[cur], d, settle=SETTLE)
            if nxt is None:
                failed += 1
                continue
            if nxt == cur or nxt in states or nxt[:2] not in area or nxt not in dist:
                continue
            states[nxt] = ex.snap()
            path_dist[nxt] = path_dist[cur] + 1
            nearest[nxt] = nearest[cur]
            queue.append(nxt)
    ex.stop()
    out = offpath_file(seg)
    out.write_bytes(pickle.dumps({
        "segment": seg.name,
        "tiles": {p: {"path_dist": path_dist[p], "goal_dist": dist[p], "nearest_demo": nearest[p],
                      "state": zlib.compress(states[p])} for p in states},
    }))
    counts = {band_name(lo, hi): sum(lo <= v <= hi for v in path_dist.values()) for lo, hi in BANDS}
    print(f"[경로 밖 상태] {len(states)}칸 (구간 칸 {sum(k[:2] in area for k in dist)}개 중), "
          f"전투 못 피한 걸음 {failed}, {time.time() - t0:.0f}초 → {out}")
    print(f"  경로에서 거리별 칸 수: {counts}")


_tiles = {}


def load_tiles(seg):
    # 작업(판)마다 다시 읽지 않게 프로세스 안에서 한 번만
    if seg.name not in _tiles:
        _tiles[seg.name] = load_offpath(seg)
    return _tiles[seg.name]


# ---------------------------------------------------------------
# 2. 한 판 돌리기 (off-path 시작 / 교란 공용)
# ---------------------------------------------------------------
def make_env(seg, policy, state_bytes):
    cfg = {} if policy == "random" else load_env_config(Path(policy).parent)
    env = GoldEnv(**{**seg.env_kwargs(), "move_mode": cfg.get("move_mode", "tap"),
                     "obs_memory": cfg.get("obs_memory", False), "obs_goal": cfg.get("obs_goal", False)})
    if state_bytes is not None:
        # GoldEnv는 파일 경로에서 불러오니까, 판 시작 상태만 메모리에서 불러오게 바꿈
        import io
        env._load_start = lambda: env.pyboy.load_state(io.BytesIO(state_bytes))
    return env, cfg.get("move_mode", "tap")


def run_episode(env, model, rng, on_step=None):
    """on_step(env, info) → None이면 정책 행동, 버튼 번호면 그 방향으로 강제 이동 (교란용)

    강제 이동은 칸이 바뀔 때까지 최대 3번 누름: tap 모델(B2)은 다른 방향을 보고 있으면 첫 누름이 회전뿐이라서.
    tile은 한 번에 걷고, 막혀 있으면 3번 다 막힘 (away_actions가 갈 수 있는 이웃만 고르니 거의 없음)
    강제 걸음도 판 걸음 수(최대 512)에 포함됨"""
    obs, info = env.reset()
    env.pyboy.tick(int(rng.integers(0, 30)), False)   # 판마다 인카운터 타이밍을 다르게 (diagnose와 같음)
    done = False
    while not done:
        forced = on_step(env, info) if on_step else None
        if forced is None:
            action = int(rng.integers(0, env.action_space.n)) if model is None else model.predict(obs)[0]
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            continue
        before = info["position"]
        for _ in range(3):
            obs, _, terminated, truncated, info = env.step(forced)
            done = terminated or truncated
            if done or info["position"] != before:
                break
    return info


def result_row(info, **extra):
    return {**extra, "goal": bool(info["reached_goal"]), "goal_step": info.get("goal_step"),
            "left": bool(info["left_area"]), "whiteout": info.get("whiteouts", 0) > 0,
            "battles": info["battles"], "timeout": not info["reached_goal"] and not info["left_area"]}


def run_start(job):
    segment, policy, band, pos, role, seed = job
    seg = SEGMENTS[segment]
    tile = load_tiles(seg)[pos]
    env, move_mode = make_env(seg, policy, zlib.decompress(tile["state"]))
    model = None if policy == "random" else _load(policy)
    torch.manual_seed(seed)    # PPO.load 뒤에 시드 (diagnose와 같은 이유)
    info = run_episode(env, model, np.random.default_rng(seed))
    env.close()
    return result_row(info, policy=policy_label(policy), band=band, role=role, pos=list(pos),
                      path_dist=tile["path_dist"], goal_dist=tile["goal_dist"],
                      holdout=in_holdout(segment, pos))


# ---------------------------------------------------------------
# 3. 교란: 경로에서 멀어지는 방향으로 k걸음 강제
# ---------------------------------------------------------------
def push_plan(here, k, tiles, adj, rng):
    """here에서 k걸음 안에 닿는 칸 중 경로에서 가장 먼 칸까지의 방향 목록 (같으면 무작위)
    한 걸음씩 "가장 먼 이웃"을 고르면 좁은 길에서 금방 막혀서(3걸음 시켜도 1칸), 미리 BFS로 목적지를 정함
    adj: 거리표 edges (턱 같은 일방통행 반영)"""
    prev = {here: None}
    depth = {here: 0}
    queue = deque([here])
    while queue:
        u = queue.popleft()
        if depth[u] >= k:
            continue
        for v in adj.get(u, []):
            if v not in prev and v in tiles:
                prev[v], depth[v] = u, depth[u] + 1
                queue.append(v)
    best = max(tiles[v]["path_dist"] for v in prev)
    target = [v for v in prev if tiles[v]["path_dist"] == best]
    v = target[rng.integers(len(target))]
    route = []
    while prev[v] is not None:
        route.append(v)
        v = prev[v]
    dirs = []
    cur = here
    for nxt in reversed(route):
        dx, dy = (nxt[2] > cur[2]) - (nxt[2] < cur[2]), (nxt[3] > cur[3]) - (nxt[3] < cur[3])
        dirs.append({(0, -1): "up", (0, 1): "down", (-1, 0): "left", (1, 0): "right"}[(dx, dy)])
        cur = nxt
    return dirs


def run_perturb(job):
    segment, policy, trigger, k, seed = job
    seg = SEGMENTS[segment]
    tiles = load_tiles(seg)
    env, move_mode = make_env(seg, policy, None)
    model = None if policy == "random" else _load(policy)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    adj = {}
    for e in json.loads(Path(seg.edges_path).read_text()):
        adj.setdefault(tuple(e[:4]), []).append(tuple(e[4:]))
    st = {"done": k == 0, "plan": None, "triggered": False, "pushed_to": None}

    def on_step(env, info):
        if st["done"]:
            return None
        here = info["position"]
        if st["plan"] is None:
            if here in tiles and tiles[here]["goal_dist"] <= trigger:
                st["triggered"] = True
                st["plan"] = push_plan(here, k, tiles, adj, rng)
            else:
                return None
        if not st["plan"]:
            # 다 밀었음 → 정책에게 돌려줌. 실제로 경로에서 얼마나 떨어졌는지 기록
            st["done"] = True
            st["pushed_to"] = tiles.get(mm.read_position(env.pyboy), {}).get("path_dist")
            return None
        return ACTIONS.index(st["plan"].pop(0))

    info = run_episode(env, model, rng, on_step)
    env.close()
    return result_row(info, policy=policy_label(policy), trigger=trigger, k=k,
                      triggered=st["triggered"], pushed_path_dist=st["pushed_to"])


# ---------------------------------------------------------------
# 4. 벽 밀기 측정: 막혔을 때 다른 걸 해보나 (MEM 실험 기준선)
# ---------------------------------------------------------------
def run_stuck(job):
    """한 판 돌리면서 걸음마다 "방향 버튼을 눌렀는데 안 움직였나"를 기록
    전투가 끼어든 걸음은 뺌 (자동 전투 뒤 좌표 변화는 정책 행동과 무관)"""
    segment, policy, start, seed = job
    seg = SEGMENTS[segment]
    tile = load_tiles(seg)[start]
    env, _ = make_env(seg, policy, zlib.decompress(tile["state"]))
    model = None if policy == "random" else _load(policy)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs, info = env.reset()
    env.pyboy.tick(int(rng.integers(0, 30)), False)
    moves = blocked = repeat_after_block = after_block = 0
    streak = longest = 0
    prev_blocked_action = None
    done = False
    while not done:
        action = int(rng.integers(0, env.action_space.n)) if model is None else int(model.predict(obs)[0])
        before, battles = info["position"], info["battles"]
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if prev_blocked_action is not None:
            after_block += 1
            repeat_after_block += action == prev_blocked_action
        prev_blocked_action = None
        if action < 4 and info["battles"] == battles:      # 방향 버튼, 전투 없음
            moves += 1
            if info["position"] == before:
                blocked += 1
                prev_blocked_action = action
                streak = streak + 1 if streak and action == last else 1
                longest = max(longest, streak)
            else:
                streak = 0
            last = action
    env.close()
    return {"policy": policy_label(policy), "start": list(start), "goal": bool(info["reached_goal"]),
            "moves": moves, "blocked": blocked, "after_block": after_block,
            "repeat_after_block": repeat_after_block, "longest_streak": longest}


def summarize_stuck(results):
    print(f"\n{'정책':<26}{'판':>5}{'도착':>7}{'막힌 걸음 비율':>14}{'막힌 직후 같은 버튼':>18}{'최장 연속(중앙/최대)':>20}")
    rows = []
    for policy in dict.fromkeys(r["policy"] for r in results):
        rs = [r for r in results if r["policy"] == policy]
        mv, bl = sum(r["moves"] for r in rs), sum(r["blocked"] for r in rs)
        ab, rab = sum(r["after_block"] for r in rs), sum(r["repeat_after_block"] for r in rs)
        ls = [r["longest_streak"] for r in rs]
        row = {"policy": policy, "runs": len(rs), "goal": rate(rs, "goal"), "blocked_frac": bl / max(mv, 1),
               "repeat_after_block": rab / max(ab, 1), "longest_p50": float(np.median(ls)), "longest_max": int(max(ls))}
        rows.append(row)
        print(f"{policy:<26}{len(rs):>5}{row['goal']:>7.0%}{row['blocked_frac']:>14.0%}{row['repeat_after_block']:>18.0%}"
              f"{row['longest_p50']:>12.0f} / {row['longest_max']:<5}")
    return rows


# ---------------------------------------------------------------
# 표
# ---------------------------------------------------------------
def rate(rs, key):
    return float(np.mean([r[key] for r in rs])) if rs else float("nan")


def summarize_eval(results):
    print(f"\n{'정책':<26}{'경로 거리':>8}{'칸':>5}{'판':>5}{'도착(밖)':>9}{'도착(짝)':>9}{'차이':>7}{'목표거리':>9}{'시간초과':>9}{'전멸':>6}")
    rows = []
    for policy in dict.fromkeys(r["policy"] for r in results):
        for lo, hi in BANDS[1:]:
            b = band_name(lo, hi)
            off = [r for r in results if r["policy"] == policy and r["band"] == b and r["role"] == "off"]
            twin = [r for r in results if r["policy"] == policy and r["band"] == b and r["role"] == "twin"]
            if not off:
                continue
            row = {"policy": policy, "band": b, "tiles": len({tuple(r["pos"]) for r in off}), "runs": len(off),
                   "goal_off": rate(off, "goal"), "goal_twin": rate(twin, "goal"),
                   "goal_dist_off": float(np.mean([r["goal_dist"] for r in off])),
                   "goal_dist_twin": float(np.mean([r["goal_dist"] for r in twin])),
                   "timeout_off": rate(off, "timeout"), "whiteout_off": rate(off, "whiteout"),
                   "left_off": rate(off, "left")}
            rows.append(row)
            print(f"{policy:<26}{b:>8}{row['tiles']:>5}{row['runs']:>5}{row['goal_off']:>9.0%}{row['goal_twin']:>9.0%}"
                  f"{(row['goal_off'] - row['goal_twin']) * 100:>+6.0f}p{row['goal_dist_off']:>9.0f}"
                  f"{row['timeout_off']:>9.0%}{row['whiteout_off']:>6.0%}")
    # 지역별: LAT1부터 학습에서 통째로 뺀 지역(envs/offpath_tiles.py HOLDOUT_REGIONS) vs 나머지
    #   빼둔 지역의 off-path = "안 가본 지역으로 번지나", 나머지 = 칸 단위로만 빼둔 곳
    print(f"\n{'정책':<26}{'지역':>10}{'칸':>5}{'판':>5}{'도착(밖)':>9}{'시간초과':>9}")
    for policy in dict.fromkeys(r["policy"] for r in results):
        for name, flag in (("빼둔 지역", True), ("나머지", False)):
            off = [r for r in results if r["policy"] == policy and r["role"] == "off" and r.get("holdout") == flag]
            if not off:
                continue
            keys = {(r["band"], tuple(r["pos"])) for r in off}
            row = {"policy": policy, "region": name, "tiles": len(keys), "runs": len(off),
                   "goal_off": rate(off, "goal"), "timeout_off": rate(off, "timeout")}
            rows.append(row)
            print(f"{policy:<26}{name:>10}{row['tiles']:>5}{row['runs']:>5}{row['goal_off']:>9.0%}"
                  f"{row['timeout_off']:>9.0%}")
    return rows


def summarize_perturb(results):
    print(f"\n{'정책':<26}{'끌어낸 지점':>10}{'k':>4}{'판':>5}{'도착':>7}{'실제 밀린 거리':>14}{'시간초과':>9}{'전멸':>6}")
    rows = []
    for policy in dict.fromkeys(r["policy"] for r in results):
        for trig in dict.fromkeys(r["trigger"] for r in results):
            for k in sorted({r["k"] for r in results}):
                rs = [r for r in results if r["policy"] == policy and r["trigger"] == trig and r["k"] == k]
                if not rs:
                    continue
                pushed = [r["pushed_path_dist"] for r in rs if r["pushed_path_dist"] is not None]
                row = {"policy": policy, "trigger": trig, "k": k, "runs": len(rs), "goal": rate(rs, "goal"),
                       "triggered": rate(rs, "triggered"),
                       "pushed_mean": float(np.mean(pushed)) if pushed else None,
                       "timeout": rate(rs, "timeout"), "whiteout": rate(rs, "whiteout")}
                rows.append(row)
                pm = f"{row['pushed_mean']:.1f}" if pushed else "-"
                print(f"{policy:<26}{'≤' + str(trig):>10}{k:>4}{len(rs):>5}{row['goal']:>7.0%}{pm:>14}"
                      f"{row['timeout']:>9.0%}{row['whiteout']:>6.0%}")
    return rows


def save(kind, segment, policies, rows, results):
    names = "+".join(dict.fromkeys(policy_label(p).split("@")[0] for p in policies))[:120]
    out = Path("runs/diagnose")
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{kind}_{segment}_{time.strftime('%m%d_%H%M')}_{names}"
    (out / f"{stem}_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    (out / f"{stem}_raw.json").write_text(json.dumps(results, ensure_ascii=False))
    print(f"\n저장: {out / (stem + '_summary.json')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["build", "eval", "perturb", "stuck"])
    parser.add_argument("segment", choices=SEGMENTS.keys())
    parser.add_argument("policies", nargs="*")
    parser.add_argument("--per-band", type=int, default=8, help="eval: 경로 거리 구간당 off-path 칸 수")
    parser.add_argument("--runs", type=int, default=4, help="eval: 칸당 판 수 / perturb: 조건당 판 수(기본 25)")
    parser.add_argument("--triggers", default="80,50,25", help="perturb: 목표까지 남은 거리가 이 이하가 되면 끌어냄")
    parser.add_argument("--ks", default="0,3,6", help="perturb: 끌어내는 걸음 수 (0 = 교란 없음 기준)")
    parser.add_argument("--workers", type=int, default=4, help="병렬 프로세스 수 (학습 중이면 2)")
    args = parser.parse_args()
    seg = SEGMENTS[args.segment]

    if args.cmd == "build":
        build(seg)
        return
    tiles = load_tiles(seg)
    if args.cmd == "stuck":
        # 시작: 경로 밖 진단 칸 32개 (빼둔 지역의 aliasing 통로 포함) + 경로 위 짝. 칸당 1판
        picks = sample_starts(tiles, args.per_band)
        starts = [p for _, off, twin in picks for p in (off, twin)]
        jobs = [(args.segment, p, s, 0) for p in args.policies for s in starts]
        print(f"시작 칸 {len(starts)}개, 판 {len(jobs)}개")
        with ProcessPoolExecutor(args.workers) as ex:
            results = list(ex.map(run_stuck, jobs))
        save("stuck", args.segment, args.policies, summarize_stuck(results), results)
        return
    if args.cmd == "eval":
        picks = sample_starts(tiles, args.per_band)
        jobs = [(args.segment, p, band, pos, role, seed)
                for p in args.policies for band, off, twin in picks
                for role, pos in (("off", off), ("twin", twin)) for seed in range(args.runs)]
        print(f"off-path 칸 {len(picks)}개 (+ 짝 경로 칸), 판 {len(jobs)}개")
        with ProcessPoolExecutor(4) as ex:
            results = list(ex.map(run_start, jobs))
        save("offpath", args.segment, args.policies, summarize_eval(results), results)
    else:
        runs = args.runs if args.runs != 4 else 25
        triggers = [int(x) for x in args.triggers.split(",")]
        ks = [int(x) for x in args.ks.split(",")]
        # k=0은 끌어낸 지점과 무관하니 한 번만
        conds = [(t, k) for k in ks for t in (triggers if k else triggers[:1])]
        jobs = [(args.segment, p, t, k, seed) for p in args.policies for t, k in conds for seed in range(runs)]
        print(f"조건 {len(conds)}개, 판 {len(jobs)}개")
        with ProcessPoolExecutor(4) as ex:
            results = list(ex.map(run_perturb, jobs))
        save("perturb", args.segment, args.policies, summarize_perturb(results), results)


if __name__ == "__main__":
    main()
