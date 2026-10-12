"""
상위 계획 (1단계): 맵 연결 그래프에서 "지금 맵 → 최종 목표 맵"으로 가는 맵 순서를 계산

왜 필요한가
- 전멸하면 마지막 회복 장소(집 2층, 포켓몬센터)에서 다시 시작함 → 거기서 원래 목표까지 어떤 맵들을 지나야 하는지 알아야 함
- 목표 조건 정책은 "다음 목표 맵"만 주면 그 맵까지 걸어감 → 계획은 맵 순서만 정하면 됨 (지금은 규칙, 나중에 상위 RL 후보)

하드코딩 안 함
- 맵 연결은 구간마다 BFS 탐색으로 만든 edges 파일(envs/data/*_edges.json)에서 "맵이 바뀌는 이동"만 모아서 자동으로 만듦
- 그래서 구간을 추가하면 그래프도 저절로 넓어짐. 탐색한 적 없는 연결은 모름 (그 구간을 만들면 생김)

실행 (확인용): uv run python -m envs.planner 24 7 26 3   (집 2층 → 무궁시티)
"""

import json
import sys
from collections import deque
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


@lru_cache(maxsize=1)
def map_graph():
    """{맵: {이웃 맵, ...}} (방향 있음: 턱처럼 한쪽으로만 갈 수 있는 연결도 그대로)"""
    g = {}
    for f in DATA.glob("*_edges.json"):
        for e in json.loads(f.read_text()):
            a, b = (e[0], e[1]), (e[4], e[5])
            if a != b:
                g.setdefault(a, set()).add(b)
    return g


def plan(cur_map, target_map):
    """cur → target 맵 순서 (cur 제외, target 포함). 못 가면 None"""
    cur_map, target_map = tuple(cur_map), tuple(target_map)
    if cur_map == target_map:
        return []
    g = map_graph()
    prev = {cur_map: None}
    q = deque([cur_map])
    while q:
        u = q.popleft()
        for v in g.get(u, ()):
            if v not in prev:
                prev[v] = u
                if v == target_map:
                    path = [v]
                    while prev[path[-1]] != cur_map:
                        path.append(prev[path[-1]])
                    return path[::-1]
                q.append(v)
    return None


def next_goal(cur_map, target_map):
    """목표 조건 정책에 줄 "다음 목표 맵" (계획의 첫 맵)"""
    p = plan(cur_map, target_map)
    return p[0] if p else None


# ---------------------------------------------------------------
# 회복 판단 (규칙, 상위 계획의 비계 — 나중에 상위 RL로 넘길 후보)
# ---------------------------------------------------------------
HEAL_HP_FRAC = 0.5   # 다시 지나가는 마을에선 HP가 이 비율 아래일 때만 회복


def pc_maps():
    """포켓몬센터 맵 목록 (맵 이름표에서 '포켓몬센터'인 맵. 손으로 좌표를 적지 않음)"""
    from envs.map_names import MAP_INFO
    return {m for m, info in MAP_INFO.items() if "포켓몬센터" in info[0]}


def nearest_pc(cur_map):
    """맵 그래프상 가장 가까운 센터와 그까지의 맵 순서. 없으면 (None, None)"""
    best = (None, None)
    for pc in pc_maps():
        p = plan(cur_map, pc)
        if p is not None and (best[1] is None or len(p) < len(best[1])):
            best = (pc, p)
    return best


def should_heal(cur_map, hp, max_hp, visited_towns, attack_pp=True):
    """구간 경계마다 상위 계획이 부름. 반환: (회복하러 갈까, 이유)
    - 센터가 있는 마을에 처음 왔으면 회복 (부활 위치 갱신)
    - 그 밖엔 HP가 HEAL_HP_FRAC 아래면 가장 가까운 센터로
    - 공격 기술 PP가 전부 0이면 센터로 (attack_pp=False, 10-12: PP 0이면 자동 전투가 야생은 도망만 하고 트레이너는 발버둥뿐)
    visited_towns: 이미 회복한 적 있는 마을(센터와 이어진 맵) 집합 — 부른 쪽이 관리"""
    pcs = pc_maps()
    g = map_graph()
    town_has_pc = any(pc in g.get(tuple(cur_map), ()) for pc in pcs)
    if town_has_pc and tuple(cur_map) not in visited_towns:
        return True, "새 마을 첫 도착"
    if max_hp and hp / max_hp < HEAL_HP_FRAC and nearest_pc(cur_map)[0] is not None:
        return True, f"HP {hp}/{max_hp} < {HEAL_HP_FRAC:.0%}"
    if not attack_pp and nearest_pc(cur_map)[0] is not None:
        return True, "공격 기술 PP 0"
    return False, ""


if __name__ == "__main__":
    from envs.map_names import map_name
    a = (int(sys.argv[1]), int(sys.argv[2]))
    b = (int(sys.argv[3]), int(sys.argv[4]))
    p = plan(a, b)
    print(map_name(a), "→", " → ".join(map_name(m) for m in p) if p is not None else "경로 없음")
