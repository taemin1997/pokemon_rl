"""
경로 밖 칸 데이터 (train/offpath.py build가 만든 states/segments/<구간>/offpath.pkl) 공용 함수

학습(envs/curriculum.py LateralDemoEnv)과 진단(train/offpath.py)이 같은 정의를 써야 해서 여기 모음
- 진단에 쓰는 칸 고르기 (sample_starts): 학습에서 빼야 할 칸을 학습 쪽도 똑같이 알아야 함
- 학습에서 통째로 빼는 지역 (HOLDOUT_REGIONS): "안 가본 지역으로 번지나" 시험용

envs → train 방향 import를 안 만들려고 train/이 아니라 envs/에 둠 (train.offpath → train.diagnose → train.train_ppo가 서로 물림)
"""

import pickle
from pathlib import Path

import numpy as np

# 경로에서 떨어진 정도 구간 (걸음). 0 = 경로 위
BANDS = [(0, 0), (1, 2), (3, 5), (6, 9), (10, 99)]

# 학습 시작점에서 통째로 빼는 지역: (맵 그룹, 맵 번호, x 최소, y 최소, x 최대, y 최대)
#   LAT1 (10-07): 29번도로 중간 남쪽 주머니. OFFPATH1에서 DEMO_TILE, B2가 가장 많이 실패한 곳
#   (학습 중 갇혔던 (22,11) 포함). 경로 위 칸(시연)은 그대로 쓰고, 경로 밖 칸만 뺌
#   → 이 지역의 off-path 성공률이 오르면 "안 가본 지역으로 실력이 번짐"
#   손으로 정한 값이지만 게임 데이터가 아니라 실험 설계(평가 분할)라 하드코딩 원칙과 무관
HOLDOUT_REGIONS = {
    "route29": [(24, 3, 15, 10, 27, 15)],
}


def band_name(lo, hi):
    return "0" if hi == 0 else (f"{lo}+" if hi >= 99 else f"{lo}-{hi}")


def offpath_file(seg):
    return Path(seg.demo_path).parent / "offpath.pkl"


def load_offpath(seg):
    """{칸: {path_dist, goal_dist, nearest_demo, state(zlib)}}"""
    return pickle.loads(offpath_file(seg).read_bytes())["tiles"]


def in_holdout(segment, pos):
    return any(pos[:2] == (g, m) and x0 <= pos[2] <= x1 and y0 <= pos[3] <= y1
               for g, m, x0, y0, x1, y1 in HOLDOUT_REGIONS.get(segment, []))


def sample_starts(tiles, per_band=8, seed=0):
    """구간마다 off-path 칸을 고르고, 칸마다 목표 거리가 가장 비슷한 경로 위 칸을 짝으로 붙임

    고르는 방식: 구간 안 칸을 목표 거리 순으로 정렬해서 고르게 per_band개 (가까운 곳/먼 곳이 골고루 섞이게)
    결정적(같은 offpath.pkl이면 항상 같은 칸) → 학습이 같은 칸을 미리 빼둘 수 있음
    반환: [(band, 칸, 짝 경로 칸), ...]
    """
    rng = np.random.default_rng(seed)
    # 순서는 offpath.pkl 저장 순서(결정적) 그대로. OFFPATH1(10-07)과 같은 칸이 뽑히게 바꾸지 말 것
    on_path = [p for p, t in tiles.items() if t["path_dist"] == 0]
    picks = []
    for lo, hi in BANDS[1:]:
        cand = sorted((p for p, t in tiles.items() if lo <= t["path_dist"] <= hi),
                      key=lambda p: tiles[p]["goal_dist"])
        if not cand:
            continue
        idx = np.unique(np.linspace(0, len(cand) - 1, min(per_band, len(cand))).round().astype(int))
        for i in idx:
            p = cand[i]
            g = tiles[p]["goal_dist"]
            best = min(abs(tiles[q]["goal_dist"] - g) for q in on_path)
            twins = [q for q in on_path if abs(tiles[q]["goal_dist"] - g) == best]
            picks.append((band_name(lo, hi), p, twins[rng.integers(len(twins))]))
    return picks


def eval_holdout_tiles(tiles, per_band=8):
    """진단에 쓰는 off-path 칸 + 바로 옆 칸 (칸 단위로 학습에서 뺄 것)"""
    out = set()
    for _, p, _ in sample_starts(tiles, per_band):
        g, m, x, y = p
        out |= {p, (g, m, x + 1, y), (g, m, x - 1, y), (g, m, x, y + 1), (g, m, x, y - 1)}
    return out
