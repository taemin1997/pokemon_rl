"""
역방향 커리큘럼 환경: 목표 근처에서 시작해서, 잘하게 되면 시작 위치를 점점 넓혀감

왜 필요한가
- A3 (도착 보상만, 입구에서만 시작): 25만 스텝 동안 한 번도 도착 못 함 → 도착 보상이 학습 신호가 될 기회 자체가 없었음
- B (거리 보상): 도착은 했지만, 거리표가 "어느 쪽이 정답인지"를 매 걸음 알려준 덕분 (정답 지도 = 선생님)
- 이 환경은 **거리표를 전혀 안 씀**. 바꾸는 건 "어디서 연습을 시작하나"뿐이고, 보상은 도착 보상만
  → 성공하면 "정답 지도 없이 시작 분포만 바꿔서 드문 보상 문제를 풀었다"가 됨

동작 (v2, 2026-10-06)
1. 씨앗: 목표에 도착한 상태(구간의 reached_goal.state)에서 무작위로 걸어서 목표 바로 앞 상태들을 만듦
   (목표가 어디인지는 문제의 정의라 알아도 됨. 경로나 거리는 모름)
2. 판마다 시작 위치 고르기
   - real_start_prob: 실제 시작 상태 (평가 기준과 같은 곳)
   - far_prob: "먼 후보"에서 (아래 4번)
   - 나머지: 본 후보 풀에서. 성공률 20~80% 또는 아직 덜 해본 후보를 우선
3. 후보의 성공률이 80%를 넘으면 거기서 무작위로 더 걸어서 새 후보를 만듦
4. 실제 시작에서 시작한 판이 지나간 자리를 몇 걸음마다 "먼 후보"로 저장
   - 목표에서 먼 후보가 공짜로 생김. 성공률이 20%를 넘으면 본 풀로 올라감
- 같은 좌표에 이미 후보가 있으면 새 후보는 버림 → 후보가 같은 동네에 쌓이지 않고 바깥으로 밀려남

v1에서 배운 것 (RC_route29, 67k에서 중단)
- 무작위 4~12걸음 확장은 확산이라 거의 안 멀어짐 (6만 스텝 후에도 목표 약 9걸음 안쪽)
- 같은 자리에 후보가 계속 쌓임 → 위 2~4번으로 고침

- 무작위 걸음, 지나간 자리 모두 거리표 없이 만들어서 정답 방향 정보가 안 들어감
- 환경(프로세스)마다 자기 풀을 따로 가짐 (6개가 독립적으로 넓혀감)
- 평가는 항상 실제 시작 상태에서 함 (train.diagnose) → 연습 위치가 어디였든 시험은 같은 조건
"""

import io
import pickle
import uuid
import zlib
from pathlib import Path

import numpy as np

from envs import memory_map as mm
from envs.gold_env import GoldEnv

DIRS = ["up", "down", "left", "right"]


class _WalkMixin:
    """두 커리큘럼 환경이 같이 쓰는 도우미: 세이브 상태 찍기, 한 칸 걷기, 무작위 걷기"""

    def _snap(self):
        buf = io.BytesIO()
        self.pyboy.save_state(buf)
        return buf.getvalue()

    def _usable_here(self):
        here = mm.read_position(self.pyboy)
        ok_area = self.allowed_maps is None or here[:2] in self.allowed_maps
        alive = mm.read_party_hp_total(self.pyboy) > 0
        return ok_area and here[:2] not in self.goal_maps and alive and not mm.in_battle(self.pyboy)

    def _walk_step(self, direction):
        """한 칸 이동 (버튼을 길게 눌러 방향 전환만 되는 걸 막음). 전투가 나면 자동 처리"""
        self.pyboy.button_press(direction)
        self.pyboy.tick(16, False)
        self.pyboy.button_release(direction)
        self.pyboy.tick(24, False)
        frames = 0
        while mm.in_battle(self.pyboy) and frames < 6000:
            self.pyboy.button_press("a")
            self.pyboy.tick(4, False)
            self.pyboy.button_release("a")
            self.pyboy.tick(20, False)
            frames += 24

    def _random_walk(self, state, walk_len=None):
        """state에서 무작위로 몇 걸음 → 쓸 수 있는 자리면 (상태, 좌표)를 돌려줌
        거리표 없이 그냥 무작위라 '정답 방향' 정보가 안 들어감"""
        lo, hi = walk_len or self.walk_len
        self.pyboy.load_state(io.BytesIO(state))
        for _ in range(int(self.rng.integers(lo, hi + 1))):
            self._walk_step(DIRS[int(self.rng.integers(0, 4))])
        if not self._usable_here():
            return None
        self.pyboy.tick(30, False)  # 화면 전환 등이 끝나게 잠깐
        return self._snap(), mm.read_position(self.pyboy)


class ReverseCurriculumEnv(_WalkMixin, GoldEnv):
    def __init__(
        self,
        goal_state_path,
        real_start_prob=0.1,
        far_prob=0.15,
        seeds=16,
        walk_len=(10, 30),
        pool_max=64,
        far_max=16,
        far_every=10,
        success_hi=0.8,
        success_lo=0.2,
        min_tries=4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.real_start_prob = real_start_prob
        self.far_prob = far_prob
        self.walk_len = walk_len
        self.pool_max = pool_max
        self.far_max = far_max
        self.far_every = far_every
        self.success_hi = success_hi
        self.success_lo = success_lo
        self.min_tries = min_tries
        self.rng = np.random.default_rng()

        # 후보: [상태 바이트, 시도 수, 성공 수, 좌표]
        self.pool = []   # 본 풀 (목표 쪽에서 넓혀가는 후보)
        self.far = []    # 먼 후보 (실제 시작에서 지나간 자리)
        with open(goal_state_path, "rb") as f:
            goal_state = f.read()
        for _ in range(seeds * 4):
            if len(self.pool) >= seeds:
                break
            cand = self._random_walk(goal_state, (2, 8))  # 씨앗은 목표 바로 앞이 되게 짧게
            if cand is not None and not self._known(cand[1]):
                self.pool.append([cand[0], 0, 0, cand[1]])
        self.current = None       # (목록 이름, 번호) 또는 None(실제 시작)

    def _known(self, pos):
        """이미 후보가 있는 좌표인가 (본 풀, 먼 후보 모두)"""
        return any(e[3] == pos for e in self.pool) or any(e[3] == pos for e in self.far)

    # -----------------------------------------------------------
    def _rate(self, e):
        return e[2] / e[1] if e[1] else 0.0

    def _pick(self, entries):
        """성공률이 적당한(20~80%) 후보 또는 아직 덜 해본 후보를 우선"""
        frontier = [i for i, e in enumerate(entries)
                    if e[1] < self.min_tries or self.success_lo <= self._rate(e) <= self.success_hi]
        if frontier and self.rng.random() < 0.8:
            return int(self.rng.choice(frontier))
        return int(self.rng.integers(0, len(entries)))

    def _load_start(self):
        r = self.rng.random()
        if not self.pool or r < self.real_start_prob:
            self.current = None
            super()._load_start()
        elif self.far and r < self.real_start_prob + self.far_prob:
            self.current = ("far", self._pick(self.far))
            self.pyboy.load_state(io.BytesIO(self.far[self.current[1]][0]))
        else:
            self.current = ("pool", self._pick(self.pool))
            self.pyboy.load_state(io.BytesIO(self.pool[self.current[1]][0]))
        self._steps_since_far = 0
        self._far_added = 0

    def _add_to_pool(self, state, pos):
        if len(self.pool) >= self.pool_max:
            # 가장 쉬운(충분히 해봤고 성공률 가장 높은) 후보를 빼고 자리를 만듦
            tried = [i for i, e in enumerate(self.pool) if e[1] >= self.min_tries]
            if not tried:
                return
            self.pool.pop(max(tried, key=lambda i: self._rate(self.pool[i])))
        self.pool.append([state, 0, 0, pos])

    def _expand(self, idx):
        """잘하게 된 후보에서 무작위로 더 걸어가서 새 후보를 만듦 (이미 있는 좌표면 다시 시도)"""
        for _ in range(6):
            cand = self._random_walk(self.pool[idx][0])
            if cand is not None and not self._known(cand[1]):
                self._add_to_pool(*cand)
                return

    # -----------------------------------------------------------
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # 실제 시작에서 시작한 판: 지나간 자리를 몇 걸음마다 먼 후보로 저장
        if self.current is None and not (terminated or truncated):
            self._steps_since_far += 1
            if self._steps_since_far >= self.far_every and self._far_added < 5 and self._usable_here():
                pos = mm.read_position(self.pyboy)
                if not self._known(pos):
                    if len(self.far) >= self.far_max:
                        self.far.pop(0)  # 오래된 것부터 버림
                    self.far.append([self._snap(), 0, 0, pos])
                    self._far_added += 1
                self._steps_since_far = 0

        if (terminated or truncated) and self.current is not None:
            kind, idx = self.current
            entries = self.pool if kind == "pool" else self.far
            if idx < len(entries):
                e = entries[idx]
                e[1] += 1
                e[2] += int(self.reached_goal)
                if kind == "pool" and e[1] >= self.min_tries and self._rate(e) > self.success_hi:
                    self._expand(idx)
                    e[1], e[2] = 0, 0  # 확장 후엔 통계를 새로 셈
                elif kind == "far" and e[1] >= self.min_tries and self._rate(e) >= self.success_lo:
                    # 먼 후보가 해볼 만해지면 본 풀로 올림
                    self.far.pop(idx)
                    self._add_to_pool(e[0], e[3])
        return obs, reward, terminated, truncated, info

    def _get_info(self):
        info = super()._get_info()
        info["rc_pool"] = len(self.pool)
        info["rc_far"] = len(self.far)
        info["rc_real_start"] = self.current is None
        if self.current is None:
            # 실제 시작 상태에서 시작한 판만 따로: 이게 진짜 실력 (평가와 같은 조건)
            info["real_goal"] = self.reached_goal
        elif self.dist is not None and self.current[0] == "pool":
            # 본 풀 후보의 시작 거리. **분석 기록용으로만** 씀 (후보 선택/보상엔 안 씀)
            # 실제 시작, 먼 후보는 빼고 기록 → 커리큘럼 전선이 얼마나 멀어지는지 정확히 보임
            info["start_dist"] = self.dist_start
        return info


class ArchiveCurriculumEnv(_WalkMixin, GoldEnv):
    """역방향 커리큘럼 v3: Go-Explore식 아카이브 + 성공률 밴드

    왜 (RC2에서 배운 것)
    - RC, RC2는 "후보를 무작위 걸음으로 만드는 것"이 병목이었음 (확산이라 전선이 아주 느리게 넓어짐)
    - 그런데 에이전트는 학습 중에 이미 여기저기 돌아다님 → 그 자리를 그대로 후보로 쓰면 됨

    동작
    - 아카이브: 좌표 → [세이브 상태, 시도 수, 성공 수, 뽑힌 수]
      모든 판에서 처음 밟은 좌표를 save_every걸음마다 저장 (같은 좌표는 하나만)
    - 시작 고르기: real_start_prob로 실제 시작, 나머지는 아카이브에서 가중치로 뽑음
      가중치 = (성공률 20~80% 또는 아직 덜 해봄 → 1.0, 너무 쉽거나 너무 어려움 → 0.1) / √(뽑힌 수 + 1)
      → "적당히 어려운 곳"과 "덜 연습한 곳"을 우선
    - 씨앗: 목표 도착 상태에서 몇 걸음 무작위로 걸어 목표 바로 앞 몇 개 (처음엔 성공 경험이 있어야 하니까)
    - 거리표는 안 씀 (start_dist 기록은 분석용)

    Go-Explore(가본 곳을 저장해두고 거기로 순간이동해서 더 탐험)와 같은 원리.
    에뮬레이터는 세이브 상태가 있어서 잘 맞음

    v4 옵션 (AC2, AC에서 배운 것)
    - AC: 모든 판의 자리를 다 넣었더니 아카이브가 먼 자리로 금방 가득 참 (77k에 170개)
      → 거의 매번 너무 어려운 곳에서 시작 → 도착 67% → 14%, 행동이 안 굳고 빙글빙글 돎
    - admit="competent": 이미 잘하는 자리(씨앗 또는 성공률 ≥ success_lo)에서 시작한 판이
      처음 local_steps걸음 안에 밟은 자리만 넣음 → 실력 있는 곳 주변으로만 넓어짐
    - untried_weight: 아직 덜 해본 자리의 우선순위 (AC 1.0 → AC2 0.5)
    """

    def __init__(self, goal_state_path, real_start_prob=0.1, seeds=8, archive_max=300,
                 save_every=5, success_lo=0.2, success_hi=0.8, min_tries=3,
                 admit="all", local_steps=None, untried_weight=1.0, archive_dir=None, save_every_eps=100,
                 **kwargs):
        super().__init__(**kwargs)
        self.admit = admit
        self.local_steps = local_steps
        self.untried_weight = untried_weight
        self.real_start_prob = real_start_prob
        self.archive_max = archive_max
        self.save_every = save_every
        self.success_lo, self.success_hi, self.min_tries = success_lo, success_hi, min_tries
        self.rng = np.random.default_rng()
        self.archive = {}   # 좌표 → [상태, 시도, 성공, 뽑힌 수, 씨앗 여부]
        self.current = None

        # 아카이브 저장/불러오기: 이어서 학습할 때 익힌 자리를 처음부터 다시 쌓지 않게
        # 환경(프로세스)마다 자기 파일에 저장하고, 시작할 땐 폴더의 파일을 전부 합쳐서 불러옴
        self._loaded_anchors = set()
        self.save_every_eps = save_every_eps
        self._episodes = 0
        self._archive_file = None
        if archive_dir:
            d = Path(archive_dir)
            d.mkdir(parents=True, exist_ok=True)
            self._archive_file = d / f"env_{uuid.uuid4().hex[:8]}.pkl"
            self._load_archives(d)

        # 씨앗: 목표 도착 상태에서 짧게 무작위로 걸어 목표 바로 앞 몇 개
        with open(goal_state_path, "rb") as f:
            goal_state = f.read()
        for _ in range(seeds * 4):
            if len(self.archive) >= seeds:
                break
            cand = self._random_walk(goal_state, (2, 8))
            if cand is not None and cand[1] not in self.archive:
                self.archive[cand[1]] = [cand[0], 0, 0, 0, True]

    def _load_archives(self, d):
        """폴더의 아카이브 파일을 전부 합침. 같은 좌표면 더 많이 해본 쪽을 남김"""
        for f in sorted(d.glob("env_*.pkl")):
            try:
                data = pickle.loads(f.read_bytes())
            except Exception:
                continue  # 저장 도중 끊긴 파일 등은 건너뜀
            for pos, (z, t, sc, pk, seed) in data["archive"].items():
                if pos not in self.archive or t > self.archive[pos][1]:
                    self.archive[pos] = [zlib.decompress(z), t, sc, pk, seed]
            self._loaded_anchors |= set(data.get("anchors", []))
        # 너무 많으면 많이 해본 자리 위주로 남김
        if len(self.archive) > self.archive_max:
            keep = sorted(self.archive, key=lambda k: (self.archive[k][4], self.archive[k][1]), reverse=True)
            self.archive = {k: self.archive[k] for k in keep[: self.archive_max]}

    def _save_archive(self):
        if self._archive_file is None:
            return
        data = {
            "archive": {pos: (zlib.compress(e[0]), e[1], e[2], e[3], e[4]) for pos, e in self.archive.items()},
            "anchors": list(getattr(self, "anchors", set())),
        }
        tmp = self._archive_file.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(data))
        tmp.replace(self._archive_file)  # 저장 도중 끊겨도 이전 파일이 깨지지 않게

    def close(self):
        self._save_archive()
        super().close()

    def _weight(self, e):
        tries, succ, picked = e[1], e[2], e[3]
        if tries < self.min_tries:
            band = self.untried_weight
        else:
            rate = succ / tries
            band = 1.0 if self.success_lo <= rate <= self.success_hi else 0.1
        return band / np.sqrt(picked + 1)

    def _load_start(self):
        if not self.archive or self.rng.random() < self.real_start_prob:
            self.current = None
            super()._load_start()
        else:
            keys = list(self.archive)
            w = np.array([self._weight(self.archive[k]) for k in keys])
            key = keys[int(self.rng.choice(len(keys), p=w / w.sum()))]
            e = self.archive[key]
            e[3] += 1
            self.current = key
            self.pyboy.load_state(io.BytesIO(e[0]))
        self._since_save = 0
        self._ep_steps = 0
        # 이번 판이 아카이브에 자리를 넣을 수 있나
        if self.admit == "all":
            self._may_admit = True
        else:  # "competent": 잘하는 자리에서 시작한 판만 (실제 시작 판은 제외)
            e = self.archive.get(self.current) if self.current is not None else None
            self._may_admit = e is not None and (
                e[4] or (e[1] >= self.min_tries and e[2] / e[1] >= self.success_lo))

    def _evict(self):
        """가득 차면: 충분히 해봤고 너무 쉬운(성공률 최고) 자리부터 뺌. 없으면 가장 많이 뽑힌 자리"""
        tried = [k for k, e in self.archive.items() if e[1] >= self.min_tries]
        if tried:
            k = max(tried, key=lambda k: self.archive[k][2] / self.archive[k][1])
        else:
            k = max(self.archive, key=lambda k: self.archive[k][3])
        del self.archive[k]

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # 처음 밟은 좌표를 아카이브에 저장 (admit 규칙을 통과한 판만, local_steps 안에서만)
        self._ep_steps += 1
        in_window = self.local_steps is None or self._ep_steps <= self.local_steps
        if not (terminated or truncated) and self._may_admit and in_window:
            self._since_save += 1
            if self._since_save >= self.save_every:
                self._since_save = 0
                pos = mm.read_position(self.pyboy)
                if pos not in self.archive and self._usable_here():
                    if len(self.archive) >= self.archive_max:
                        self._evict()
                    self.archive[pos] = [self._snap(), 0, 0, 0, False]

        if (terminated or truncated) and self.current is not None and self.current in self.archive:
            e = self.archive[self.current]
            e[1] += 1
            e[2] += int(self.reached_goal)
        if terminated or truncated:
            # 강제 종료(창 닫기, 정전)에도 너무 많이 잃지 않게 주기적으로 저장
            self._episodes += 1
            if self._episodes % self.save_every_eps == 0:
                self._save_archive()
        return obs, reward, terminated, truncated, info

    def _get_info(self):
        info = super()._get_info()
        info["rc_pool"] = len(self.archive)
        info["rc_real_start"] = self.current is None
        if self.current is None:
            info["real_goal"] = self.reached_goal
        elif self.dist is not None and self.dist_start < 999:
            # 분석용 (선택/보상엔 안 씀). 거리표에 없는 자리(점프 중간 등, 999)는 평균을 망쳐서 뺌
            info["start_dist"] = self.dist_start
        return info


class FrontierArchiveEnv(ArchiveCurriculumEnv):
    """역방향 커리큘럼 v5 (AC3): 넓게 모으고, 실력 경계 바로 바깥에서 고르기

    왜 (RC → AC2에서 배운 것)
    - RC, RC2: 후보를 무작위 걸음으로 만듦 → 너무 느림
    - AC: 모든 판의 자리를 후보로 → 너무 어려운 곳만 연습해서 학습 신호 부족
    - AC2: 잘하는 판에서만 모음 → 잘하는 에이전트는 곧장 목표로 가서 바깥 자리가 안 생김
    → "모으기"와 "고르기"를 분리해야 함

    동작
    - 모으기: AC와 같음 (모든 판에서 처음 밟은 좌표를 아카이브에)
    - 익힌 자리(anchor): 성공률이 success_hi(80%)를 넘은 좌표. 아카이브에서 빠져도 좌표는 기억함
    - 고르기 가중치 (÷ √(뽑힌 수 + 1))
        전선   = 익힌 자리에서 같은 맵 좌표로 radius칸 이내인데 아직 못 익힌 자리 → 1.0
                  (해봤는데 성공률이 20% 미만이면 0.3: 벽/턱 너머라 좌표만 가까운 곳일 수 있음)
        씨앗   = 익힌 자리가 아직 없을 때의 출발점 → 1.0
        익힌 자리 → 0.1 (가끔 복습)
        나머지(먼 곳) → 0.02
    - 좌표는 에이전트 위치일 뿐이라 정답 경로 정보가 아님. 거리표는 여전히 안 씀
    - 원래 역방향 커리큘럼(Florensa 등)의 "잘하는 시작점 근처에서 새 시작점 고르기"를,
      무작위 걸음 대신 이미 모아둔 아카이브에서 좌표로 고르는 방식으로 바꾼 것
    """

    def __init__(self, radius=6, w_frontier=1.0, w_frontier_hard=0.3, w_mastered=0.1, w_far=0.02,
                 archive_max=400, **kwargs):
        super().__init__(archive_max=archive_max, **kwargs)
        self.radius = radius
        self.w = dict(frontier=w_frontier, frontier_hard=w_frontier_hard, mastered=w_mastered, far=w_far)
        self.anchors = set(self._loaded_anchors)   # 익힌 좌표 (맵그룹, 맵번호, x, y). 이어서 학습이면 불러온 것부터

    def _mastered(self, e):
        return e[1] >= self.min_tries and e[2] / e[1] >= self.success_hi

    def _near_anchor(self, pos):
        g, n, x, y = pos
        r = self.radius
        return any(a[0] == g and a[1] == n and abs(a[2] - x) + abs(a[3] - y) <= r for a in self.anchors)

    def _frontier_weight(self, pos, e):
        tries, succ, picked = e[1], e[2], e[3]
        if self._mastered(e):
            w = self.w["mastered"]
        elif self._near_anchor(pos) or (not self.anchors and e[4]):
            hard = tries >= self.min_tries and succ / tries < self.success_lo
            w = self.w["frontier_hard"] if hard else self.w["frontier"]
        else:
            w = self.w["far"]
        return w / np.sqrt(picked + 1)

    def _load_start(self):
        if not self.archive or self.rng.random() < self.real_start_prob:
            self.current = None
            GoldEnv._load_start(self)
        else:
            keys = list(self.archive)
            w = np.array([self._frontier_weight(k, self.archive[k]) for k in keys])
            key = keys[int(self.rng.choice(len(keys), p=w / w.sum()))]
            e = self.archive[key]
            e[3] += 1
            self.current = key
            self.pyboy.load_state(io.BytesIO(e[0]))
        self._since_save = 0
        self._ep_steps = 0
        self._may_admit = True   # 모으기는 모든 판에서 (AC와 같음)

    def _evict(self):
        """가득 차면: 익힌 자리 중 가장 많이 뽑힌 것, 없으면 전선/익힌 자리에서 가장 먼 것부터 뺌
        (익힌 좌표는 anchors에 남아 있어서 빼도 전선 계산엔 영향 없음)"""
        mastered = [k for k, e in self.archive.items() if self._mastered(e)]
        if mastered:
            k = max(mastered, key=lambda k: self.archive[k][3])
        else:
            far = [k for k in self.archive if not self._near_anchor(k) and not self.archive[k][4]]
            k = far[int(self.rng.integers(0, len(far)))] if far else max(self.archive, key=lambda k: self.archive[k][3])
        del self.archive[k]

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # 판이 끝나서 통계가 갱신된 뒤, 익힌 자리면 anchors에 추가
        if (terminated or truncated) and self.current is not None and self.current in self.archive:
            if self._mastered(self.archive[self.current]):
                self.anchors.add(self.current)
        return obs, reward, terminated, truncated, info

    def _get_info(self):
        info = super()._get_info()
        info["rc_anchors"] = len(self.anchors)
        return info


class DemoReverseEnv(GoldEnv):
    """역방향 시연 학습 (Salimans & Chen 2018 "한 번의 시연으로 몬테주마 배우기", Go-Explore 2단계와 같은 방식)

    왜 (AC3에서 배운 것)
    - AC3는 "익힌 자리에서 좌표로 6칸 이내 = 다음 연습 장소"로 전선을 넓혔고, 선생님 없이 처음으로 U턴(약 50)까지 옴
    - 그런데 U턴 너머(턱 줄 아래 길)는 좌표로는 가깝지만 실제로는 x=31 틈으로 돌아가야 해서 멂
      → "좌표 근처 = 다음 단계" 가정이 깨지는 지형에서 전선이 멈춤
    - 여기선 전선이 좌표가 아니라 **경로를 따라** 움직임: 시연 경로 위의 상태를 순서대로 갖고 있다가
      "목표 쪽 끝"에서 시작 → 잘하게 되면 시작점을 경로상 몇 걸음 앞으로(입구 쪽으로) 당김

    동작
    - 시연: build_segment --demo가 저장한 경로 위 상태들 (0 = 실제 시작, 마지막 = 목표 바로 앞)
    - 전선 ptr: 시연 번호. 판 시작은 [ptr, ptr + window] 중 하나에서 무작위 (ptr이 가장 어려운 곳)
    - ptr을 당기는 건 이 환경이 아니라 학습 쪽 콜백(train_ppo의 DemoCurriculumCallback)이 함
      → 환경 6개의 결과를 모아서 **전선 하나를 같이** 씀 (환경마다 따로면 6배 느림)
    - real_start_prob: 실제 시작에서 시작 (진짜 실력 기록용, 평가와 같은 조건)

    RL이 하는 것 vs 비계
    - 비계: 시연 경로 (BFS가 찾은 것 = 선생님이 정답 경로를 한 번 보여준 것), 세이브 상태로 순간이동
    - RL: 각 시작점에서 어떤 버튼을 누를지는 도착 보상(goal_only)만으로 배움. 버튼은 안 따라 하게 함
      (경로 상태는 "어디서 연습할까"에만 쓰고, 보상/관찰엔 안 들어감)
    - 거리 보상(B2)은 매 걸음 방향을 알려줬다면, 이건 "이 길 위 어디서 연습할지"만 알려줌 → 더 약한 도움
    """

    def __init__(self, demo_path, real_start_prob=0.1, window=4, **kwargs):
        super().__init__(**kwargs)
        with open(demo_path, "rb") as f:
            data = pickle.load(f)
        self.demo_states = [zlib.decompress(z) for z in data["states"]]
        self.demo_pos = data["positions"]
        self.demo_len = len(self.demo_states)
        self.window = window
        self.real_start_prob = real_start_prob
        self.ptr = self.demo_len - 1   # 처음 전선: 목표 1걸음 앞. 콜백이 set_ptr로 바꿈
        self.rng = np.random.default_rng()
        self.current = None            # 이번 판 시연 번호 (None = 실제 시작)
        self._results = []             # (시작 번호, 도착 여부). 콜백이 pop_results로 가져감

    # 콜백이 env_method로 부르는 것들 -------------------------------
    def set_ptr(self, ptr):
        self.ptr = int(np.clip(ptr, 0, self.demo_len - 1))

    def pop_results(self):
        out, self._results = self._results, []
        return out

    # -----------------------------------------------------------
    def _load_start(self):
        if self.rng.random() < self.real_start_prob:
            self.current = None
            super()._load_start()
        else:
            hi = min(self.ptr + self.window, self.demo_len - 1)
            self.current = int(self.rng.integers(self.ptr, hi + 1))
            self.pyboy.load_state(io.BytesIO(self.demo_states[self.current]))

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if (terminated or truncated) and self.current is not None:
            self._results.append((self.current, bool(self.reached_goal)))
        return obs, reward, terminated, truncated, info

    def _get_info(self):
        info = super()._get_info()
        info["demo_front"] = self.demo_len - self.ptr   # 전선: 목표에서 경로상 몇 걸음 앞에서 연습 중
        if self.current is None:
            info["real_goal"] = self.reached_goal
        else:
            info["demo_start"] = self.demo_len - self.current
            if self.dist is not None and self.dist_start < 999:
                info["start_dist"] = self.dist_start   # 분석용 (거리표 기준. 선택/보상엔 안 씀)
        return info


class LateralDemoEnv(DemoReverseEnv):
    """경로 옆 시작 (LAT1): DemoReverseEnv + 일부 판은 시연 경로 "밖" 칸에서 시작

    왜 (OFFPATH1에서 배운 것)
    - DEMO_TILE은 모든 판이 시연 경로 위에서 시작 → 경로에서 3칸 넘게 떨어진 곳에서 시작하면 도착률 −40~−88%p
      (같은 목표 거리의 경로 위 칸은 ~100%). 실패는 안 가본 주머니에서 헤매다 시간 초과
    - 거리 보상(B2)도 똑같이 무너져서 "학습 분포가 좁아서"인지 "화면만으론 어디인지 몰라서"인지 못 가름
    - 시작 자리를 경로 밖으로 넓혀 학습했는데 안 가본 지역(빼둔 지역)에서도 오르면 → 분포 문제였음

    동작
    - 판 시작: real_start_prob → 실제 입구 / lateral_prob → 경로 밖 칸 / 나머지 → 지금처럼 시연 전선
    - 경로 밖 후보: 가장 가까운 시연 칸이 전선(ptr) 뒤쪽(= 이미 익힌 쪽)이고, 경로에서 1~radius걸음 떨어진 칸
      → 경로 위 실력이 있는 곳에서만 옆으로 넓힘 (진행도 × 폭 2차원 커리큘럼)
    - 경로 밖 판의 절반은 가장 바깥(path_dist == radius) 칸, 절반은 안쪽 칸
    - radius는 콜백이 넓힘 (가장 바깥 칸에서 시작한 판이 잘 되면 +1)
    - 학습에서 빼는 칸: 진단 칸 + 바로 옆 (칸 단위), HOLDOUT_REGIONS 지역의 경로 밖 칸 전부 (지역 단위)
    - 상태는 offpath.pkl (train.offpath build, 전투 피하는 타이밍으로 걸어서 만든 상태 → HP는 시연과 같음)

    RL이 하는 것 vs 비계
    - 비계 추가: 경로 밖 시작 상태 (BFS로 걸어간 세이브). 여전히 버튼은 안 줌. 보상도 도착뿐
    - 전선 당기기 규칙은 DEMO와 같음 (시연 판만 셈). 단 시연 판 비율이 줄어서 전선은 느려질 수 있음
    """

    def __init__(self, offpath_path, segment, lateral_prob=0.25, holdout_eval=True, **kwargs):
        from envs.offpath_tiles import eval_holdout_tiles, in_holdout

        super().__init__(**kwargs)
        with open(offpath_path, "rb") as f:
            tiles = pickle.load(f)["tiles"]
        banned = eval_holdout_tiles(tiles) if holdout_eval else set()
        # 경로 밖 칸만 (경로 위 칸은 시연 상태로 이미 씀)
        self.lat_tiles = [(p, t["path_dist"], t["nearest_demo"], t["state"]) for p, t in tiles.items()
                          if t["path_dist"] > 0 and p not in banned and not in_holdout(segment, p)]
        self.lateral_prob = lateral_prob
        self.segment_name = segment    # 여러 구간 섞어 학습(GOAL1)에서 콜백이 환경을 구간별로 묶을 때 씀
        self.radius = 1
        self.lateral = None            # 이번 판이 경로 밖 시작이면 그 칸의 path_dist
        self._lat_results = []         # (path_dist, 도착 여부)
        self._real_results = []        # 실제 시작 판의 도착 여부 (구간별 real_goal 기록용)

    def set_radius(self, r):
        self.radius = int(r)

    def pop_lat_results(self):
        out, self._lat_results = self._lat_results, []
        return out

    def pop_real_results(self):
        out, self._real_results = self._real_results, []
        return out

    def _load_start(self):
        self.lateral = None
        u = self.rng.random()
        if self.real_start_prob <= u < self.real_start_prob + self.lateral_prob:
            cand = [c for c in self.lat_tiles if c[2] >= self.ptr and c[1] <= self.radius]
            # 절반은 가장 바깥(path_dist == radius), 나머지는 안쪽에서 고르게
            #   왜: 칸 수는 경로 바로 옆(1칸)이 압도적이라 그냥 고르면 바깥 칸 판이 너무 적음
            #       → 넓히기 기준(가장 바깥 칸 도착률)을 채우는 데 반경 1칸에 약 3만 스텝 → 250k 안에 다 못 넓힘
            outer = [c for c in cand if c[1] == self.radius]
            inner = [c for c in cand if c[1] < self.radius]
            if outer and (not inner or self.rng.random() < 0.5):
                cand = outer
            elif inner:
                cand = inner
            if cand:
                pos, pd, _, state = cand[self.rng.integers(len(cand))]
                self.current = None
                self.lateral = pd
                self.pyboy.load_state(io.BytesIO(zlib.decompress(state)))
                return
        # 실제 시작 / 시연 전선은 부모와 같음. 부모가 다시 난수를 뽑으니 실제 시작 확률을 그만큼 보정
        if u < self.real_start_prob:
            self.current = None
            GoldEnv._load_start(self)
        else:
            hi = min(self.ptr + self.window, self.demo_len - 1)
            self.current = int(self.rng.integers(self.ptr, hi + 1))
            self.pyboy.load_state(io.BytesIO(self.demo_states[self.current]))

    def step(self, action):
        if self.lateral is None:
            out = super().step(action)
            if (out[2] or out[3]) and self.current is None:
                self._real_results.append(bool(self.reached_goal))
            return out
        obs, reward, terminated, truncated, info = GoldEnv.step(self, action)
        if terminated or truncated:
            self._lat_results.append((self.lateral, bool(self.reached_goal)))
        return obs, reward, terminated, truncated, info

    def _get_info(self):
        if self.lateral is None:
            info = super()._get_info()
        else:
            # 경로 밖 시작 판: real_goal/demo_start에 섞이지 않게 따로 기록
            info = GoldEnv._get_info(self)
            info["demo_front"] = self.demo_len - self.ptr
            info["lat_start"] = self.lateral
        info["lat_radius"] = self.radius
        return info
