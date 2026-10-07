"""
Go-Explore 1단계: 에이전트 쪽 탐색으로 시작 → 목표 경로 찾기 (BFS 거리표 없이)

왜 필요한가 (DEMO_TILE에서 배운 것)
- DEMO_TILE은 거리 보상 없이 무궁시티 88~100%를 달성했지만, 연습 순서를 정한 시연 경로는
  BFS 거리표에서 뽑은 **최단 경로**였음 = 정답 지도를 아는 선생님이 한 번 보여준 것
- 이 스크립트는 그 경로를 정답 지도 없이 찾음:
  가본 칸을 저장해두고(아카이브), 그중 하나로 순간이동해서 무작위로 더 걸어보는 걸 반복하다가
  우연히 목표 맵에 닿으면 그 경로를 시연으로 씀 (Ecoffet 등 2019 "Go-Explore"의 1단계)
- 2단계(튼튼하게 만들기)는 지금 DEMO_TILE 커리큘럼 그대로 (시연 파일만 이걸로 바꿈)

BFS와 뭐가 다른가
- BFS: 모든 칸에서 4방향을 빠짐없이 시험 + 목표에서 거꾸로 계산한 거리로 최단 경로를 뽑음
- 여기: 덜 가본 칸을 골라 **무작위로** 걸음. 목표 방향, 거리, 지도 모두 모름. 경로는 최단이 아닐 수 있음
- 남는 비계 (명시): 칸 = 좌표 (맵, x, y) (원 논문의 "도메인 지식 칸"과 같음), 세이브 상태 되돌리기,
  전투 회피 (지금 시연 기록과 같은 방식 → HP 조건이 같아서 "경로 출처"만 바뀜), 행동은 방향 4개만

아카이브 규칙 (원 논문 단순화)
- 칸마다 [세이브 상태, 시작부터 걸음 수, 직전 칸, 뽑힌 수]
- 고르기: 1 / √(뽑힌 수 + 1)  → 덜 뽑힌 칸 우선 (원 논문의 카운트 기반 선택)
- 갱신: 처음 온 칸이거나, 아는 칸에 더 적은 걸음으로 왔으면 덮어씀 → 경로가 점점 짧아짐
- 직전 칸을 따라 거꾸로 가면 경로가 나옴. 걸음 수가 직전 칸 쪽으로 갈수록 항상 줄어서 고리가 생기지 않음

실행
- uv run python -m train.go_explore route29                     (기본: 목표 찾은 뒤 500번 더 다듬기)
- uv run python -m train.go_explore route29 --iters 5000 --after-goal 1000
결과: states/segments/<구간>/demo_goexplore.pkl (build_segment --demo와 같은 형식)
"""

import argparse
import json
import pickle
import time
import zlib
from pathlib import Path

import numpy as np

from envs.segments import SEGMENTS
from train.build_segment import DIRS, Explorer


def explore(seg, iters, after_goal, walk, repeat, seed):
    rng = np.random.default_rng(seed)
    ex = Explorer(seg.start_state)
    area = set(map(tuple, seg.allowed_maps))
    goals = set(map(tuple, seg.goal_maps))

    start = ex.pos()
    # 칸 → [세이브 상태, 걸음 수, 직전 칸, 뽑힌 수]
    archive = {start: [ex.snap(), 0, None, 0]}
    best_goal = None          # (걸음 수, 목표 직전 칸, 목표 칸)
    found_at = None
    moves = 0                 # 에뮬레이터에서 실제로 시도한 걸음 수 (탐색 비용)
    log = []
    t0 = time.time()

    for it in range(iters):
        if found_at is not None and it - found_at >= after_goal:
            break
        # 1) 고르기: 덜 뽑힌 칸 우선
        keys = list(archive)
        w = np.array([1.0 / np.sqrt(archive[k][3] + 1) for k in keys])
        cur = keys[int(rng.choice(len(keys), p=w / w.sum()))]
        archive[cur][3] += 1
        state, length = archive[cur][0], archive[cur][1]

        # 2) 거기서 무작위로 걷기 (같은 방향을 이어 누르는 경향: repeat 확률로 직전 방향 유지)
        d = DIRS[int(rng.integers(0, 4))]
        for _ in range(walk):
            if rng.random() > repeat:
                d = DIRS[int(rng.integers(0, 4))]
            nxt = ex.try_move(state, d, settle=60, tries=2)   # 전투가 안 나는 타이밍을 찾아 걸음 (2번 다 전투면 None → 다른 방향)
            moves += 1
            if nxt is None or nxt == cur:
                continue                              # 전투만 나는 칸이거나 벽 → 다른 방향 시도
            if nxt[:2] in goals:
                if best_goal is None or length + 1 < best_goal[0]:
                    best_goal = (length + 1, cur, nxt)
                    found_at = it if found_at is None else found_at
                    log.append((it, moves, length + 1))
                    print(f"  [목표] 반복 {it}, 누적 걸음 {moves}, 경로 {length + 1}걸음")
                break
            if nxt[:2] not in area:
                break                                 # 구간 밖 (마을 등) → 이번 탐색 끝
            new_state = ex.snap()
            length += 1
            c = archive.get(nxt)
            if c is None:
                archive[nxt] = [new_state, length, cur, 0]
            elif length < c[1]:
                c[0], c[1], c[2] = new_state, length, cur   # 더 짧게 옴 → 경로 갱신 (뽑힌 수는 유지)
            else:
                # 이미 더 짧은 길이 있는 칸: 그 칸의 기록을 이어서 씀 (지금 걸음 수는 버림)
                new_state, length = c[0], c[1]
            state, cur = new_state, nxt

        if it % 50 == 0:
            print(f"반복 {it}: 아카이브 {len(archive)}칸, 누적 걸음 {moves}, {time.time() - t0:.0f}초")

    ex.stop()
    return start, archive, best_goal, {"iters": it + 1, "moves": moves, "archive": len(archive),
                                        "seconds": round(time.time() - t0), "goal_log": log}


def extract_path(archive, best_goal):
    """목표 직전 칸에서 직전 칸을 따라 시작까지 거꾸로 → 시작부터의 칸 순서"""
    path = []
    c = best_goal[1]
    while c is not None:
        path.append(c)
        c = archive[c][2]
    return path[::-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("segment", choices=SEGMENTS.keys())
    parser.add_argument("--iters", type=int, default=20000, help="최대 반복 수")
    parser.add_argument("--after-goal", type=int, default=500, help="목표를 처음 찾은 뒤 경로를 다듬는 반복 수")
    parser.add_argument("--walk", type=int, default=20, help="한 번 고른 칸에서 무작위로 걷는 최대 걸음")
    parser.add_argument("--repeat", type=float, default=0.7, help="직전 방향을 유지할 확률")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    seg = SEGMENTS[args.segment]

    start, archive, best_goal, stats = explore(seg, args.iters, args.after_goal, args.walk, args.repeat, args.seed)
    print(f"[탐색] {stats}")
    if best_goal is None:
        print("[중단] 목표를 못 찾음 → --iters를 늘려보면 됨")
        return

    path = extract_path(archive, best_goal)
    # 분석용: BFS 거리표로 경로가 최단과 얼마나 다른지 (경로 찾기엔 안 씀)
    dist = {}
    if Path(seg.dist_path).exists():
        dist = {tuple(r[:4]): r[4] for r in json.loads(Path(seg.dist_path).read_text())}
    out = Path(seg.demo_path).with_name("demo_goexplore.pkl")
    out.write_bytes(pickle.dumps({
        "segment": seg.name,
        "source": "go_explore",
        "positions": path,
        "states": [zlib.compress(archive[c][0]) for c in path],
        "dist": [dist.get(p) for p in path],       # 분석용 (학습엔 안 씀)
        "stats": {**stats, "seed": args.seed, "walk": args.walk, "repeat": args.repeat},
    }))
    shortest = dist.get(start)
    print(f"[시연] {len(path)}칸 (목표까지 {best_goal[0]}걸음, BFS 최단 {shortest}) → {out}")


if __name__ == "__main__":
    main()
