"""
구간 하나의 거리표 만들기 (+ 다음 구간 시작 상태 자동 생성)

왜 필요한가 (확장성)
- 29번도로 때는 거리표 스크립트를 따로 짰음 → 구간마다 새로 짜면 전당등록까지 못 감
- 이 스크립트는 envs/segments.py의 구간 정의만 보고 동작함 → 새 구간은 정의 한 줄 + 이 명령 하나

하는 일
1. 시작 상태에서 PyBoy로 상하좌우를 눌러보며 갈 수 있는 칸을 전부 탐색 (BFS)
   - 범위: 구간의 allowed_maps 안 + goal_maps에 처음 들어간 칸
   - 풀숲 인카운터가 나면 타이밍을 바꿔 재시도 (전투는 이동 가능 여부와 무관한 우연)
   - 턱 점프는 끝날 때까지 기다린 뒤 좌표를 읽음
2. 목표 맵 칸들에서 거꾸로 BFS → 각 칸의 "목표까지 남은 걸음" (턱 같은 일방통행도 정확히 반영)
3. --harvest: 거리표를 따라 실제로 걸어가서 목표 맵에 들어간 순간을 저장
   → 다음 구간의 시작 상태로 그대로 씀 (사람이 플레이해서 만들 필요 없음)
   --demo: 같은 길을 전투를 피해 걸으면서 지나간 칸의 상태를 전부 저장 (역방향 시연 학습용)
4. --check: 같은 탐색을 두 번 해서 거리표가 얼마나 다른지 확인
   (돌아다니는 NPC 등 때문에 같은 시작 상태라도 결과가 다를 수 있음 → 29번도로에서 115/128 경험)

결과
- envs/data/<구간>_dist.json, <구간>_edges.json
- (--harvest) states/segments/<구간>/reached_goal.state
- (--demo) states/segments/<구간>/demo.pkl

실행
- uv run python -m train.build_segment cherrygrove
- uv run python -m train.build_segment route29 --harvest     (거리표 있으면 탐색 생략하고 도착 상태만)
- uv run python -m train.build_segment cherrygrove --check
- uv run python -m train.build_segment route29 --demo        (reached_goal.state는 안 건드림)
"""

import argparse
import io
import json
import pickle
import time
import zlib
from collections import deque
from pathlib import Path

from pyboy import PyBoy

from envs import memory_map as mm
from envs.segments import SEGMENTS

DIRS = ["up", "down", "left", "right"]
# 걸음 뒤 몇 프레임 기다렸다가 "전투가 안 났다"고 볼지
#   인카운터는 걸음 직후가 아니라 화면 전환 연출 뒤에 전투 모드가 켜짐: 걸음 뒤 약 150~170프레임 (10-07 측정)
#   너무 짧으면 "곧 전투가 시작될 상태"를 칸 상태로 저장함 → 그 칸에서 모든 방향이 전투 → 막다른 곳으로 오판
#   (30번도로 탐색이 18칸에서 멈춤, 29번도로 경로 밖 상태 483개 중 12개가 이런 상태였음)
SETTLE = 240


class Explorer:
    """PyBoy 하나로 세이브 상태 저장/불러오기 + 한 걸음 이동"""

    def __init__(self, start_state):
        self.pb = PyBoy("roms/gold_rom.gb", window="null")
        self.pb.set_emulation_speed(0)
        with open(start_state, "rb") as f:
            self.pb.load_state(f)
        self.pb.tick()

    def snap(self):
        buf = io.BytesIO()
        self.pb.save_state(buf)
        return buf.getvalue()

    def load(self, data):
        self.pb.load_state(io.BytesIO(data))

    def pos(self):
        return mm.read_position(self.pb)

    def move(self, direction, idle=0):
        pb = self.pb
        if idle:
            pb.tick(idle, False)
        # 16프레임 누르기: 다른 방향을 보고 서 있을 때 짧게 누르면 방향만 바뀌어서 길게 누름
        pb.button_press(direction)
        pb.tick(16, False)
        pb.button_release(direction)
        pb.tick(24, False)
        # 턱 점프 등 이동이 끝날 때까지 기다림
        for _ in range(8):
            before = self.pos()
            pb.tick(8, False)
            if self.pos() == before:
                break

    def try_move(self, state, direction, settle=0, tries=6):
        """state에서 direction으로 한 걸음. 전투가 나면 타이밍을 바꿔 재시도. 실패하면 None
        settle: 걸음 뒤 몇 프레임 더 기다렸다가 전투 여부를 봄
          (인카운터는 걸음이 끝나고 조금 뒤에 시작될 수 있음. 거리표 탐색은 이동 가능 여부만 보면 돼서 0,
           --demo는 HP가 깎이면 안 돼서 60)
        tries: 전투가 나면 타이밍을 바꿔 몇 번까지 다시 해볼지 (go_explore는 속도 때문에 2)"""
        # 기본 6가지 타이밍. tries를 더 크게 주면 타이밍 후보를 늘림 (--demo에서 전투를 못 피할 때)
        idles = (0, 3, 7, 13, 21, 34) + tuple(range(40, 40 + 9 * max(0, tries - 6), 9))
        for idle in idles[:tries]:
            self.load(state)
            self.move(direction, idle)
            if settle:
                self.pb.tick(settle, False)
            if not mm.in_battle(self.pb):
                return self.pos()
        return None

    def finish_battle(self):
        """전투가 나면 A 연타로 끝냄 (GoldEnv의 자동 전투와 같은 방식)"""
        frames = 0
        while mm.in_battle(self.pb) and frames < 6000:
            self.pb.button_press("a")
            self.pb.tick(4, False)
            self.pb.button_release("a")
            self.pb.tick(20, False)
            frames += 24

    def stop(self):
        self.pb.stop(save=False)


def explore(seg):
    """앞으로 BFS → (states, edges)"""
    ex = Explorer(seg.start_state)
    area = set(map(tuple, seg.allowed_maps))
    goals = set(map(tuple, seg.goal_maps))

    start = ex.pos()
    states = {start: ex.snap()}
    edges = {start: set()}
    queue = deque([start])
    t0 = time.time()
    while queue:
        cur = queue.popleft()
        for d in DIRS:
            # settle 120 + 타이밍 10가지: 걸음 뒤 늦게 시작되는 인카운터를 거름 (10-07, route30)
            #   예전(settle 0)엔 "곧 전투가 시작될 상태"를 칸 상태로 저장 → 그 칸에서 모든 방향이 전투 → 막다른 곳으로 오판
            #   30번도로는 입구부터 풀숲이라 18칸에서 탐색이 끝났음. 29번도로/연두마을은 운 좋게 안 걸렸음
            nxt = ex.try_move(states[cur], d, settle=SETTLE, tries=10)
            if nxt is None or nxt == cur:
                continue
            if nxt[:2] not in area and nxt[:2] not in goals:
                continue
            edges[cur].add(nxt)
            if nxt not in states:
                states[nxt] = ex.snap()
                edges[nxt] = set()
                if nxt[:2] not in goals:  # 목표 맵 안쪽까지 넓힐 필요는 없음
                    queue.append(nxt)
    ex.stop()
    print(f"[탐색] {len(states)}칸, {time.time() - t0:.0f}초")
    return start, states, edges


def distances(states, edges, goals):
    """목표 맵 칸들에서 거꾸로 BFS"""
    reverse = {n: set() for n in edges}
    for u, vs in edges.items():
        for v in vs:
            reverse[v].add(u)
    dist = {n: 0 for n in states if n[:2] in goals}
    queue = deque(dist)
    while queue:
        v = queue.popleft()
        for u in reverse[v]:
            if u not in dist:
                dist[u] = dist[v] + 1
                queue.append(u)
    return dist


def compose(seg):
    """평가용 구간의 거리표를 앞뒤 구간 거리표로 이어 붙여 만듦 (BFS를 다시 안 돌림)

    왜: 집 1층에서 무궁시티까지 한 번에 BFS를 돌리면 결과가 흔들림 (594칸 / 721칸, 경로를 못 찾는 경우도 있음)
        → 이미 결정성을 확인한 구간별 거리표를 쓰는 게 안전함. 평가에선 진행 지표(진행 p50 등)에만 쓰는 값
    방법: 뒤 구간부터. 앞 구간 칸의 거리 = 앞 구간에서 다음 구간 진입까지 거리 + 다음 구간 진입 칸의 거리(가장 작은 값)
    """
    parts = [SEGMENTS[name] for name in seg.compose]
    table = {tuple(r[:4]): r[4] for r in json.loads(Path(parts[-1].dist_path).read_text())}
    for part in reversed(parts[:-1]):
        d = {tuple(r[:4]): r[4] for r in json.loads(Path(part.dist_path).read_text())}
        goals = set(map(tuple, part.goal_maps))
        # 다음 구간에 들어서는 칸들 (앞 구간 거리표에서 거리 0, 목표 맵 칸) 중 다음 구간 거리가 가장 작은 값
        entry = min(table[k] for k, v in d.items() if v == 0 and k[:2] in goals and k in table)
        for k, v in d.items():
            if k[:2] not in goals:
                table.setdefault(k, v + entry)
    Path(seg.dist_path).write_text(json.dumps([[*k, v] for k, v in sorted(table.items())]))
    print(f"[이어 붙임] {seg.compose} → {seg.dist_path} ({len(table)}칸)")
    return table


def build(seg):
    if seg.compose:
        return compose(seg)
    goals = set(map(tuple, seg.goal_maps))
    start, states, edges = explore(seg)
    dist = distances(states, edges, goals)

    # 안전장치: 시작 칸에서 목표까지 경로가 없으면 기존 파일을 덮어쓰지 않음
    if start not in dist:
        print("[중단] 시작 칸에서 목표까지 경로를 못 찾음 → 파일 저장 안 함")
        print("  가능한 원인: 스토리 이벤트로 길이 막힘, allowed_maps에 필요한 맵이 빠짐")
        return None

    Path(seg.dist_path).parent.mkdir(parents=True, exist_ok=True)
    Path(seg.dist_path).write_text(json.dumps([[*k, d] for k, d in sorted(dist.items())]))
    Path(seg.edges_path).write_text(json.dumps([[*u, *v] for u, vs in edges.items() for v in vs]))
    print(f"[저장] {seg.dist_path} ({len(dist)}칸), 시작→목표 최단 {dist[start]}걸음")
    if len(states) > len(dist):
        print(f"  목표로 못 가는 칸 {len(states) - len(dist)}개")
    return dist


def walk_to_goal(seg, avoid_battles=False, on_tile=None):
    """거리표를 따라 목표 맵까지 실제로 걸어감 (--harvest, --demo 공용)
    도착하면 (Explorer, 걸음 수), 실패하면 (None, 걸음 수). 도착했으면 Explorer는 부른 쪽이 stop()

    avoid_battles: 걸을 때마다 전투가 안 나는 타이밍을 찾아서 걸음 (try_move와 같은 방식)
      - --harvest(False): 실제 플레이처럼 전투가 나면 싸우고 계속 감
      - --demo(True): 시연 상태들의 HP가 시작 상태와 같게 유지됨
        (안 피하면 목표 근처 상태일수록 HP가 깎여 있어서, "목표 근처 연습 = HP 낮은 상태 연습"이 섞임)
    on_tile(ex, pos): 목표 밖의 새 칸에 설 때마다 불림 (시연 기록용)
    """
    dist = {tuple(r[:4]): r[4] for r in json.loads(Path(seg.dist_path).read_text())}
    adj = {}
    for e in json.loads(Path(seg.edges_path).read_text()):
        adj.setdefault(tuple(e[:4]), []).append(tuple(e[4:]))
    goals = set(map(tuple, seg.goal_maps))

    ex = Explorer(seg.start_state)
    blocked = 0
    last = None
    for step in range(2000):
        cur = ex.pos()
        if cur[:2] in goals:
            # 맵 전환 화면이 끝나도록 조금 기다림
            ex.pb.tick(60, False)
            return ex, step
        if cur not in dist:
            print(f"[실패] {cur}가 거리표에 없음")
            break
        if on_tile is not None and cur != last:
            on_tile(ex, cur)
            last = cur
        # 목표에 가까운 칸부터 시도. 가장 좋은 칸이 막혀 있으면(돌아다니는 NPC 등) 다음 후보로
        options = sorted((v for v in adj.get(cur, []) if v in dist), key=lambda v: dist[v])
        moved = False
        for nxt in options:
            if nxt[:2] == cur[:2]:
                # 같은 맵: 좌표 차이로 방향 결정 (턱은 2칸 차이라 부호만 봄)
                dx, dy = (nxt[2] > cur[2]) - (nxt[2] < cur[2]), (nxt[3] > cur[3]) - (nxt[3] < cur[3])
                d = {(0, -1): "up", (0, 1): "down", (-1, 0): "left", (1, 0): "right"}.get((dx, dy))
            else:
                # 맵이 바뀌는 이동: 네 방향을 시험해서 그 맵으로 가는 방향을 찾음
                state = ex.snap()
                d = next((dd for dd in DIRS if ex.try_move(state, dd) == nxt), None)
                ex.load(state)
            if d is None:
                continue
            if avoid_battles:
                state = ex.snap()
                # 타이밍 후보 16가지 (기본 6가지로 못 피하는 칸이 있었음: route29_rev (41,10), 10-07)
                #   settle 120: route29_rev (41,10)은 걸음 뒤 60프레임이 지나서야 인카운터가 시작돼서
                #   "피했다"고 보고 저장한 상태가 실제로는 전투 중이었음 (그 뒤 모든 시도가 전투)
                if ex.try_move(state, d, settle=SETTLE, tries=16) is None:
                    # 어떤 타이밍이든 전투가 나면 그냥 싸우고 감 (드묾)
                    print(f"  [전투 못 피함] {cur} → {d}")
                    ex.load(state)
                    ex.move(d)
                    ex.finish_battle()
            else:
                # 실제 이동. 여기선 전투가 나도 피하지 않고 자동으로 끝내고 계속 감 (실제 플레이와 같게)
                ex.move(d)
                ex.finish_battle()
            if ex.pos() != cur:
                moved = True
                break
        if moved:
            blocked = 0
            continue
        blocked += 1
        if blocked > 20:
            print(f"[실패] {cur}에서 계속 막힘")
            break
        ex.pb.tick(30, False)  # NPC가 비켜주길 잠깐 기다림
    ex.stop()
    return None, step


def harvest(seg, avoid_battles=False):
    """거리표를 따라 걸어가서 목표 맵에 들어간 순간을 저장 → 다음 구간 시작 상태
    avoid_battles: 전투를 피하는 타이밍으로 걸음 → HP가 그대로인 도착 상태
      (안 피하면 이전 구간 전투 피해를 물려받음: cherrygrove 시작 HP 10/20이었음)"""
    ex, step = walk_to_goal(seg, avoid_battles=avoid_battles)
    if ex is None:
        return None
    out = Path(seg.goal_state_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(ex.snap())
    print(f"[도착] {step}걸음 만에 {ex.pos()}, HP 합 {mm.read_party_hp_total(ex.pb)} → {out}")
    ex.stop()
    return out


def record_demo(seg):
    """--demo: 시작 → 목표 경로 위의 상태를 전부 저장 (역방향 시연 학습용, envs/curriculum.py DemoReverseEnv)

    왜 필요한가 (AC3에서 배운 것)
    - AC3는 "익힌 자리에서 좌표로 가까운 곳 = 다음 연습 장소"라고 가정했는데,
      U턴(턱 줄) 너머는 좌표로는 가깝지만 실제 경로로는 멀어서 전선이 거기서 멈춤
    - 경로 위의 상태를 순서대로 갖고 있으면 "다음 연습 장소 = 경로상 한두 걸음 앞"이 되어 지형과 상관없이 정확함

    비계 명시: 이 경로는 BFS 탐색(거리표)이 찾은 것 → 선생님이 "정답 경로를 한 번 보여준 것"
      - 학습엔 상태(시작 위치)만 쓰고, 버튼(행동)은 안 줌 → 어떻게 걸을지는 RL이 도착 보상만으로 배움
      - 거리 보상(B2)처럼 매 걸음 방향을 알려주는 것보다 훨씬 약한 도움
    """
    positions, states = [], []

    def on_tile(ex, pos):
        positions.append(pos)
        states.append(zlib.compress(ex.snap()))

    ex, step = walk_to_goal(seg, avoid_battles=True, on_tile=on_tile)
    if ex is None:
        print("[중단] 목표까지 못 감 → 시연 저장 안 함")
        return None
    hp = mm.read_party_hp_total(ex.pb)
    ex.stop()
    dist = {tuple(r[:4]): r[4] for r in json.loads(Path(seg.dist_path).read_text())}
    out = Path(seg.demo_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pickle.dumps({
        "segment": seg.name,
        "positions": positions,                       # 경로 순서대로 (0 = 시작 칸)
        "states": states,                             # zlib 압축한 세이브 상태
        "dist": [dist.get(p) for p in positions],     # 분석용 (학습엔 안 씀)
    }))
    print(f"[시연] {len(states)}칸, {step}걸음 만에 도착, 도착 시 HP 합 {hp} → {out}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("segment", choices=SEGMENTS.keys())
    parser.add_argument("--harvest", action="store_true", help="목표 도착 상태를 저장 (다음 구간 시작용)")
    parser.add_argument("--check", action="store_true", help="탐색을 두 번 해서 거리표 차이 확인")
    parser.add_argument("--demo", action="store_true", help="시작→목표 경로 위 상태들을 저장 (역방향 시연 학습용)")
    parser.add_argument("--rebuild", action="store_true", help="거리표가 있어도 다시 만듦")
    parser.add_argument("--avoid-battles", action="store_true", help="--harvest 때 전투를 피해서 걸음 (HP 유지)")
    args = parser.parse_args()
    seg = SEGMENTS[args.segment]

    if not Path(seg.start_state).exists():
        print(f"시작 상태 없음: {seg.start_state}")
        print("  이전 구간에서 --harvest로 만들고, 그 파일을 이 경로로 복사하면 됨")
        return

    if args.check:
        goals = set(map(tuple, seg.goal_maps))
        d1 = distances(*explore(seg)[1:], goals)
        d2 = distances(*explore(seg)[1:], goals)
        common = set(d1) & set(d2)
        diff = sum(d1[k] != d2[k] for k in common)
        print(f"[결정성] 1차 {len(d1)}칸 / 2차 {len(d2)}칸 / 공통 {len(common)}칸 중 거리 다른 칸 {diff}개")
        return

    if args.rebuild or not Path(seg.dist_path).exists():
        if build(seg) is None:
            return
    else:
        print(f"[생략] 거리표 있음: {seg.dist_path} (다시 만들려면 --rebuild)")

    if args.demo:
        record_demo(seg)

    if args.harvest:
        out = harvest(seg, avoid_battles=args.avoid_battles)
        # 다음 구간(SEGMENTS 순서상 바로 뒤)의 시작 상태가 아직 없으면 자동으로 연결
        names = list(SEGMENTS)
        i = names.index(args.segment)
        if out and i + 1 < len(names):
            nxt = SEGMENTS[names[i + 1]]
            if not Path(nxt.start_state).exists():
                Path(nxt.start_state).parent.mkdir(parents=True, exist_ok=True)
                Path(nxt.start_state).write_bytes(Path(out).read_bytes())
                print(f"[연결] 다음 구간 '{nxt.name}' 시작 상태로 복사: {nxt.start_state}")
            else:
                print(f"[연결 생략] '{nxt.name}' 시작 상태가 이미 있음: {nxt.start_state}")


if __name__ == "__main__":
    main()
