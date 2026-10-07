"""
29번도로 진단용 재료 만들기: 목표까지의 거리표 + 평가용 시작 상태(S1, S2, S3)

왜 필요한가
- "에이전트가 얼마나 진행했나"를 칸 수가 아니라 "목표(무궁시티)까지 남은 실제 걸음 수"로 재고 싶음
  → x좌표만 보면 아랫길로 내려가는 구간(x는 그대로, y만 변함)을 진행으로 못 셈
- 같은 거리표를 나중에 potential-based shaping 보상에도 그대로 씀
- 평가용 시작 상태를 사람 손으로 만들면 매번 위치가 조금씩 달라짐 → BFS 중에 정확한 칸에서 저장

방법
1. 시작 상태에서 상하좌우를 눌러보며 갈 수 있는 칸을 전부 탐색 (BFS). "A칸 → B칸으로 갈 수 있음"을 다 기록
2. 기록한 이동을 거꾸로 따라가며 무궁시티의 모든 칸에서 BFS → 각 칸의 "목표까지 최소 걸음 수"
   (거꾸로 하는 이유: 턱(ledge)은 한쪽으로만 내려갈 수 있어서, 갈 수 있는 방향을 그대로 써야 함)
3. S1(29번도로 입구), S2(x=44 막다른 곳), S3(아랫길로 꺾는 갈림길) 칸에 도착했을 때 상태를 저장

결과
- envs/data/route29_dist.json  (맵그룹, 맵번호, x, y, 거리) 목록. ROM 데이터가 아니라 직접 측정한 값
- envs/data/route29_edges.json (출발 칸 4개 값 + 도착 칸 4개 값) 목록. 한 걸음으로 갈 수 있는 이동
- states/eval/S1_entry.state, S2_deadend.state, S3_fork.state

실행: uv run python -m train.build_route29_eval   (약 3~5분, 창 없음)
- 평가 시작 상태는 이미 있으면 유지함. 새로 만들려면 --force-states
"""

import io
import json
import time
from collections import deque
from pathlib import Path

from pyboy import PyBoy

from envs import memory_map as mm

import sys

FORCE_STATES = "--force-states" in sys.argv
GOAL_MAP = (26, 3)                 # 무궁시티
ALLOWED_GROUPS = {24}              # 연두마을 일대(집, 건물, 29번도로) + 목표 맵만 탐색
DIRS = ["up", "down", "left", "right"]

# 평가용 시작 상태로 쓸 칸 (2026-10-02 최단 경로 분석 결과 기준)
EVAL_TILES = {
    "S1_entry": (24, 3, 59, 9),    # 마을에서 29번도로로 막 들어온 칸
    "S2_deadend": (24, 3, 44, 8),  # 윗길 막다른 곳 (v1이 항상 멈추던 곳)
    "S3_fork": (24, 3, 53, 9),     # 여기서 아래로 꺾어야 아랫길
}


def main():
    pb = PyBoy("roms/gold_rom.gb", window="null")
    pb.set_emulation_speed(0)
    with open("states/start.state", "rb") as f:
        pb.load_state(f)
    pb.tick()

    def snap():
        buf = io.BytesIO()
        pb.save_state(buf)
        return buf.getvalue()

    def move(direction, idle=0):
        # idle: 누르기 전에 흘려보낼 프레임. 풀숲 인카운터는 타이밍에 따라 달라져서, 전투가 나면 타이밍을 바꿔 재시도
        if idle:
            pb.tick(idle, False)
        # 16프레임 누르기: 다른 방향을 보고 서 있을 때 짧게 누르면 방향만 바뀌고 안 걸어서 길게 누름
        pb.button_press(direction)
        pb.tick(16, False)
        pb.button_release(direction)
        pb.tick(24, False)
        # 턱 점프는 일반 걸음보다 오래 걸려서, 위치가 멈출 때까지 더 기다림
        # (안 기다리면 점프 도중 좌표를 읽어서 대각선 이동 같은 가짜 이동이 기록됨)
        for _ in range(8):
            before = mm.read_position(pb)
            pb.tick(8, False)
            if mm.read_position(pb) == before:
                break

    start = mm.read_position(pb)
    states = {start: snap()}
    edges = {start: set()}         # 칸 → 그 칸에서 한 번에 갈 수 있는 칸들
    queue = deque([start])
    t0 = time.time()

    # 1) 앞으로 BFS: 갈 수 있는 칸과 이동 기록
    while queue:
        cur = queue.popleft()
        for d in DIRS:
            # 풀숲 인카운터는 이동 가능 여부와 상관없는 우연이라, 전투가 나면 타이밍을 바꿔 재시도
            # (재시도 없이 건너뛰면 하필 중요한 이동이 빠져서 목표까지 경로가 끊길 수 있음 → 실제로 겪음)
            for idle in (0, 3, 7, 13, 21, 34):
                pb.load_state(io.BytesIO(states[cur]))
                move(d, idle)
                if not mm.in_battle(pb):
                    break
            else:
                continue
            nxt = mm.read_position(pb)
            if nxt == cur:
                continue
            if nxt[0] not in ALLOWED_GROUPS and nxt[:2] != GOAL_MAP:
                continue
            edges[cur].add(nxt)
            if nxt not in states:
                states[nxt] = snap()
                edges[nxt] = set()
                # 목표 맵 안쪽까지 넓힐 필요는 없음 (도착만 하면 됨)
                if nxt[:2] != GOAL_MAP:
                    queue.append(nxt)
    print(f"탐색한 칸 {len(states)}개, {time.time() - t0:.0f}초")

    # 2) 거꾸로 BFS: 목표 맵의 칸들에서 출발해서 "들어오는 이동"을 따라가며 거리 계산
    reverse = {n: set() for n in edges}
    for u, vs in edges.items():
        for v in vs:
            reverse[v].add(u)
    dist = {n: 0 for n in states if n[:2] == GOAL_MAP}
    queue = deque(dist)
    while queue:
        v = queue.popleft()
        for u in reverse[v]:
            if u not in dist:
                dist[u] = dist[v] + 1
                queue.append(u)

    # 안전장치: 시작 칸에서 목표까지 경로가 없으면 기존 파일을 덮어쓰지 않고 멈춤
    # (한 번 빈 거리표로 덮어써버린 적이 있음)
    if start not in dist:
        print("시작 칸에서 목표까지 경로를 못 찾음 → 기존 파일을 그대로 두고 종료")
        pb.stop(save=False)
        return

    out = Path("envs/data/route29_dist.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps([[*k, d] for k, d in sorted(dist.items())]))
    print(f"거리표 저장: {out} ({len(dist)}칸). 시작 칸 거리 {dist.get(start)}")

    # 이동 기록도 저장 (턱처럼 한쪽으로만 갈 수 있는 곳을 분석할 때 씀)
    edge_out = Path("envs/data/route29_edges.json")
    edge_out.write_text(json.dumps([[*u, *v] for u, vs in edges.items() for v in vs]))
    print(f"이동 기록 저장: {edge_out}")
    unreachable = len(states) - len(dist)
    if unreachable:
        print(f"  목표로 못 가는 칸 {unreachable}개 (막힌 방 등)")

    # 3) 평가용 시작 상태 저장
    eval_dir = Path("states/eval")
    eval_dir.mkdir(parents=True, exist_ok=True)
    for name, tile in EVAL_TILES.items():
        path = eval_dir / f"{name}.state"
        # 이미 있으면 덮어쓰지 않음: 평가 시작 상태가 바뀌면 이전 실험 결과와 비교가 안 됨
        if path.exists() and not FORCE_STATES:
            print(f"  {name}: 이미 있음 → 유지 (다시 만들려면 --force-states)")
            continue
        if tile not in states:
            print(f"  {name} {tile}: BFS에서 못 찾음!")
            continue
        path.write_bytes(states[tile])
        print(f"  {name} {tile}: 저장, 목표까지 {dist.get(tile)}걸음")

    pb.stop(save=False)


if __name__ == "__main__":
    main()
