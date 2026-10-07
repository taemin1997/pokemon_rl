"""
4단계: GoldEnv에 PPO 학습

1단계 02_ppo_train.py와 뼈대는 같음 (환경 만들기 → PPO → learn → 저장)
달라진 점은 전부 "포켓몬이 CartPole보다 훨씬 느리고 길다"는 문제를 해결하려는 것:
  - 환경 여러 개를 동시에 돌림 (SubprocVecEnv): 환경 1개는 170스텝/초라 너무 느림
  - 중간 저장 (CheckpointCallback): 몇 시간짜리 학습이 중간에 죽어도 날리지 않게
  - 게임 지표 기록 (GameStatsCallback): 보상 숫자만으로는 "뭘 배웠는지"가 안 보여서
    탐험 칸 수, 맵 수, 배지 수를 TensorBoard에 같이 남김 → 보상 설계 비교의 핵심 자료
  - 보상 프리셋: 실험마다 보상만 바꿔서 같은 조건으로 비교

실행 (프로젝트 루트에서)
- 짧게 동작 확인:  uv run python -m train.train_ppo --reward v1_explore --steps 20000 --name smoke
- 본 학습:         uv run python -m train.train_ppo --reward v1_explore --steps 1000000
- 이어서 학습:     uv run python -m train.train_ppo --env route29 --reward task --name B2_route29 --resume latest --total-steps 250000
                  (latest = 그 실험 폴더에서 가장 최근 저장 파일. --total-steps = 누적 목표, 남은 만큼만 자동으로 학습)
- 하이퍼파라미터 실험: uv run python -m train.train_ppo --reward v1_explore --ent-coef 0.03 --name v2_ent003
- 29번도로 커리큘럼: uv run python -m train.train_ppo --env route29 --steps 250000 --name A_route29
- 거리 shaping (실험 B): uv run python -m train.train_ppo --env route29 --reward B_progress --steps 250000 --name B_route29
- 역방향 시연:     uv run python -m train.train_ppo --env route29_demo --reward goal_only --steps 250000 --name DEMO_route29
- 조작 방식 바꾸기: ... --move tile   (방향 버튼 1번 = 1칸. 기본 tap)
                  (먼저 uv run python -m train.build_segment route29 --demo 로 시연 경로 저장)
- 학습 곡선:       uv run tensorboard --logdir runs
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from envs.curriculum import ArchiveCurriculumEnv, DemoReverseEnv, FrontierArchiveEnv, LateralDemoEnv, ReverseCurriculumEnv
from envs.gold_env import GoldEnv
from envs.segments import SEGMENTS

# ---------------------------------------------------------------
# 보상 프리셋: 실험 이름 → GoldEnv의 reward_cfg
# ---------------------------------------------------------------
# 비교 실험을 할 때 여기에 한 줄씩 추가하면 됨 (적지 않은 항목은 DEFAULT_REWARD 값 사용)
# 한 번에 하나만 바꿔야 "무엇 때문에 행동이 달라졌는지"를 말할 수 있음
REWARD_PRESETS = {
    "v1_explore": {},  # 기본값 그대로: 새 칸 0.1, 새 맵 1.0, 배지 100
    # 실험 B: v1 보상 + 거리 shaping (걸음당 0.1). 거리표가 있는 환경(--env route29)에서만 작동
    #   29번도로 전체(103걸음)를 가면 +10.3 → 새 칸 보상(칸당 0.1)과 비슷한 크기라 한쪽이 압도하지 않음
    "B_progress": {"progress": 0.1},
    # 구간 학습 기본 보상 (B2~): 역할별로 분리
    #   과제: 목표 도착 +10 / 안내: 거리 감소 0.1 / 탐험: 끔 (거리표가 있는 구간에선 필요 없음)
    #   B에서 새 칸 보상이 "끝내지 말고 근처를 더 훑어라"라고 해서 목표 앞에서 뜸을 들였음
    "task": {"new_tile": 0.0, "new_map": 0.0, "progress": 0.1, "goal": 10.0},
    # 실험 A3: BFS 없이 "탐험 + 도착 보너스" (거리 보상 끔)
    #   B의 도착이 BFS 거리 신호 덕분이었는지, "도착하면 칭찬"만으로도 U턴을 넘는지 가르는 실험
    "A3_goal": {"goal": 10.0},
    # 역방향 커리큘럼용: 도착 보상만 (새 칸, 새 맵, 거리 보상 전부 끔) → 가장 드문 보상
    "goal_only": {"new_tile": 0.0, "new_map": 0.0, "goal": 10.0},
}

# ---------------------------------------------------------------
# 환경 프리셋: 어디서 시작하고, 언제 끝나고, 전투를 어떻게 처리할지
# ---------------------------------------------------------------
# 보상(REWARD_PRESETS)과 따로 둔 이유: "보상을 바꾼 실험"과 "환경을 바꾼 실험"을 섞지 않으려고
ENV_PRESETS = {
    # v1, v2와 같은 환경: 집 1층에서 시작, 2048스텝
    "home": {},
    # 구간 학습 환경: envs/segments.py에 구간을 추가하면 자동으로 여기에 생김 (예: route29, cherrygrove)
    #   route29 = 실험 A, A2, B가 쓴 것과 같은 설정 (근거: notes/05 진단)
    **{name: seg.env_kwargs() for name, seg in SEGMENTS.items()},
}

# 커리큘럼 환경: 학습할 때만 ReverseCurriculumEnv를 씀. 평가/라이브는 base 구간의 실제 시작에서
#   base: 어느 구간을 바탕으로 하나 / 나머지: ReverseCurriculumEnv 설정 (envs/curriculum.py)
CURRICULUM_PRESETS = {
    "route29_rc": {
        "base": "route29",
        "goal_state_path": SEGMENTS["route29"].goal_state_path,  # 무궁시티에 막 들어온 상태 (씨앗용)
        "real_start_prob": 0.1,  # 10%는 실제 시작(입구)에서 → 진짜 실력을 학습 중에도 기록
    },
    # v3: Go-Explore식 아카이브 (에이전트가 다녀본 좌표를 전부 후보로)
    "route29_ac": {
        "base": "route29",
        "goal_state_path": SEGMENTS["route29"].goal_state_path,
        "real_start_prob": 0.1,
        "_class": "archive",
    },
    # v4: 잘하는 자리 주변으로만 아카이브를 넓힘 (AC는 너무 빨리 넓어져서 어려운 곳만 연습함)
    "route29_ac2": {
        "base": "route29",
        "goal_state_path": SEGMENTS["route29"].goal_state_path,
        "real_start_prob": 0.1,
        "admit": "competent",
        "local_steps": 60,
        "untried_weight": 0.5,
        "_class": "archive",
    },
    # v5 (AC3): 넓게 모으고(AC처럼), 익힌 자리에서 좌표로 6칸 이내인 "전선"에서 고르기
    "route29_ac3": {
        "base": "route29",
        "goal_state_path": SEGMENTS["route29"].goal_state_path,
        "real_start_prob": 0.1,
        "radius": 6,
        "_class": "frontier",
    },
    # 역방향 시연 학습: 시연 경로(build_segment --demo) 위에서 목표 쪽 끝부터 시작, 잘하면 입구 쪽으로 당김
    #   전선을 당기는 규칙은 DemoCurriculumCallback (아래)
    "route29_demo": {
        "base": "route29",
        "demo_path": SEGMENTS["route29"].demo_path,
        "real_start_prob": 0.1,
        "window": 4,
        "_class": "demo",
    },
    # 같은 커리큘럼, 시연만 Go-Explore 1단계가 찾은 경로로 (train/go_explore.py, BFS 거리표 없이 찾음)
    "route29_demo_ge": {
        "base": "route29",
        "demo_path": str(Path(SEGMENTS["route29"].demo_path).with_name("demo_goexplore.pkl")),
        "real_start_prob": 0.1,
        "window": 4,
        "_class": "demo",
    },
    # LAT1: route29_demo + 일부 판(25%)은 시연 경로 밖 칸에서 시작. 경로에서 떨어진 폭(radius)은 잘하면 넓힘
    #   경로 밖 상태: train.offpath build route29 (states/segments/route29/offpath.pkl)
    #   진단 칸 + 옆 칸, HOLDOUT_REGIONS(남쪽 주머니)의 경로 밖 칸은 학습에서 뺌 → 진단이 외운 걸 재지 않게
    "route29_demo_lat": {
        "base": "route29",
        "demo_path": SEGMENTS["route29"].demo_path,
        "offpath_path": str(Path(SEGMENTS["route29"].demo_path).with_name("offpath.pkl")),
        "segment": "route29",
        "real_start_prob": 0.1,
        "lateral_prob": 0.25,
        "window": 4,
        "_class": "demo_lat",
    },
    # GOAL1: 29번도로 양방향을 정책 하나로. 환경을 구간별로 나눠 맡김 (환경 i → parts[i % 구간 수])
    #   구간마다 자기 시연 전선/폭 (route29_demo_lat과 같은 규칙). 목표 정보는 --goal로 켬 (끄면 대조군)
    #   base: 라이브/진단 기본값용 (첫 구간)
    "route29_bidir_lat": {
        "base": "route29",
        "parts": ["route29", "route29_rev"],
        "_class": "multi_lat",
    },
}


def lat_kwargs(segment):
    """구간 하나의 경로 옆 시작 커리큘럼 설정 (route29_demo_lat과 같은 규칙). 섞어 학습(multi_lat)에서 구간마다 씀"""
    seg = SEGMENTS[segment]
    return {
        **ENV_PRESETS[segment],
        "demo_path": seg.demo_path,
        "offpath_path": str(Path(seg.demo_path).with_name("offpath.pkl")),
        "segment": segment,
        "real_start_prob": 0.1,
        "lateral_prob": 0.25,
        "window": 4,
    }


# 구간마다 역방향 시연 커리큘럼을 자동으로 만듦: --env <구간>_demo
#   (build_segment <구간> --demo 로 시연을 먼저 저장해야 함). route29_demo는 위에 직접 적은 것과 같은 설정
for _name, _seg in SEGMENTS.items():
    CURRICULUM_PRESETS.setdefault(f"{_name}_demo", {
        "base": _name,
        "demo_path": _seg.demo_path,
        "real_start_prob": 0.1,
        "window": 4,
        "_class": "demo",
    })

# info에 있으면 에피소드 끝날 때 TensorBoard에 기록할 진행 지표들
GAME_METRICS = [
    "visited_tiles", "visited_maps", "badges",
    "battles", "exp_gain", "reached_goal", "left_area", "progress", "dist_best", "shaping_sum", "goal_step", "whiteouts",
    "rc_pool", "rc_far", "rc_anchors", "real_goal", "start_dist",  # 커리큘럼: 후보 풀 크기, 실제 시작에서 시작한 판의 도착률
    "demo_start",  # 역방향 시연: 이번 판 시작이 경로상 목표에서 몇 걸음 앞이었나
    "lat_start",   # 경로 옆 시작(LAT1): 경로 밖 시작 판의 경로에서 거리
]


def load_env_config(model_dir):
    """모델 폴더의 env_config.json (학습 때 쓴 조작 방식 등). 없으면 {} = 예전 실험 = tap
    모델 .zip엔 환경 설정이 안 들어가서, 평가/라이브가 학습과 다른 조건으로 돌지 않게 따로 저장함"""
    f = Path(model_dir) / "env_config.json"
    return json.loads(f.read_text()) if f.exists() else {}


class GameStatsCallback(BaseCallback):
    """에피소드가 끝날 때마다 info의 게임 지표를 TensorBoard에 기록

    ep_rew_mean(보상 합계)만 보면 "보상이 올랐다"까지만 알 수 있음
    보상 설계를 바꾸면 보상 숫자의 크기 자체가 달라져서 실험끼리 직접 비교도 안 됨
    → 보상과 상관없이 같은 기준으로 잴 수 있는 '행동 지표'를 따로 남김
    """

    def _on_step(self):
        # 환경 여러 개가 동시에 돌아서 dones/infos가 환경 수만큼의 리스트로 옴
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                # record_mean: 로그를 찍는 시점까지 모인 값의 평균으로 기록됨
                # reached_goal 같은 True/False는 평균을 내면 곧 "비율"(성공률)이 됨
                for key in GAME_METRICS:
                    if key in info:
                        self.logger.record_mean(f"game/{key}", float(info[key]))
        return True  # False를 돌려주면 학습이 중단됨


class LiveSnapshotCallback(BaseCallback):
    """라이브 창용: 약 every스텝마다 models/<실험>/live.zip을 덮어씀

    왜: 체크포인트는 5만 스텝마다라 라이브 창이 첫 모습을 보여주기까지 수 분~수십 분 걸림
       (시연 학습은 초반 변화가 빨라서 그 사이가 제일 볼 만함)
    live.zip은 항상 가장 최근 정책이라 --resume latest가 이걸 골라도 문제없음
    """

    def __init__(self, path, every=10_000):
        super().__init__()
        self.path, self.every, self.last = Path(path), every, 0

    def _on_rollout_end(self):
        if self.num_timesteps - self.last >= self.every:
            self.last = self.num_timesteps
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.path)

    def _on_step(self):
        return True


class DemoCurriculumCallback(BaseCallback):
    """역방향 시연 학습의 전선(ptr)을 당기는 규칙. 환경 6개의 결과를 모아 전선 하나를 같이 씀

    왜 환경 안이 아니라 여기서 하나
    - AC 계열은 환경(프로세스)마다 자기 아카이브를 따로 가졌음 → 각자 처음부터 넓혀야 해서 느림
    - 시연 경로는 모든 환경이 같으니 전선도 하나면 됨 → 결과를 6배 빨리 모아서 6배 빨리 당겨짐

    규칙 (Salimans & Chen 2018의 역방향 알고리즘을 단순화)
    - 롤아웃(3072스텝)이 끝날 때마다 각 환경의 판 결과를 가져옴
    - 지금 전선 구간 [ptr, ptr + window]에서 시작한 판만 셈 (전선을 옮기기 전 결과는 버림)
    - 그 판이 min_eps개 이상이고 도착률 ≥ success면 → ptr을 step_back만큼 입구 쪽으로
    - 전선 위치는 runs/curriculum/<실험>/demo_ptr.json에 저장 → --resume 때 이어서 씀

    경로 옆 시작(LateralDemoEnv, LAT1)이면 폭도 같이 조절 (진행도 × 폭 2차원)
    - 경로 밖 판 중 "지금 가장 바깥(path_dist == radius)"에서 시작한 판만 셈 (전선과 같은 원리: 가장 어려운 곳)
    - min_eps개 이상 + 도착률 ≥ success → radius +1 (최대 max_radius). 전선(ptr) 규칙은 그대로
    """

    def __init__(self, save_path, success=0.5, min_eps=24, step_back=3, max_radius=8):
        super().__init__()
        self.save_path = Path(save_path)
        self.success, self.min_eps, self.step_back = success, min_eps, step_back
        self.max_radius = max_radius   # 학습에 쓸 수 있는 경로 밖 칸이 최대 8걸음 (나머지는 빼둔 지역)

    def _on_training_start(self):
        venv = self.training_env
        self.n = venv.get_attr("demo_len", [0])[0]
        self.window = venv.get_attr("window", [0])[0]
        self.ptr = self.n - 1   # 처음: 목표 1걸음 앞
        self.lateral = venv.get_attr("lateral_prob", [0])[0] if venv.has_attr("lateral_prob") else None
        self.radius = 1
        if self.save_path.exists():
            saved = json.loads(self.save_path.read_text())
            self.ptr = saved["ptr"]
            self.radius = saved.get("radius", 1)
            print(f"[시연 전선] 이어서: 목표에서 {self.n - self.ptr}걸음 앞부터 (경로 밖 폭 {self.radius})")
        self.hits = []          # 지금 전선 구간에서 시작한 판들의 도착 여부
        self.lat_hits = []      # 경로 밖 가장 바깥(path_dist == radius)에서 시작한 판들의 도착 여부
        venv.env_method("set_ptr", self.ptr)
        if self.lateral:
            venv.env_method("set_radius", self.radius)

    def _on_rollout_end(self):
        for results in self.training_env.env_method("pop_results"):
            for idx, ok in results:
                if self.ptr <= idx <= self.ptr + self.window:
                    self.hits.append(ok)
        rate = sum(self.hits) / len(self.hits) if self.hits else 0.0
        self.logger.record("game/demo_front", self.n - self.ptr)   # 전선: 목표에서 경로상 몇 걸음 앞
        self.logger.record("game/demo_front_rate", rate)           # 전선 구간 도착률 (당기기 기준)
        changed = False
        if len(self.hits) >= self.min_eps and rate >= self.success and self.ptr > 0:
            self.ptr = max(0, self.ptr - self.step_back)
            self.hits = []
            self.training_env.env_method("set_ptr", self.ptr)
            changed = True
        if self.lateral:
            all_lat = [r for results in self.training_env.env_method("pop_lat_results") for r in results]
            self.lat_hits += [ok for pd, ok in all_lat if pd == self.radius]
            lat_rate = sum(self.lat_hits) / len(self.lat_hits) if self.lat_hits else 0.0
            self.logger.record("game/lat_radius", self.radius)          # 경로 밖 폭
            self.logger.record("game/lat_rate", lat_rate)               # 가장 바깥 칸 도착률 (넓히기 기준)
            if all_lat:
                self.logger.record("game/lat_goal", sum(ok for _, ok in all_lat) / len(all_lat))  # 경로 밖 판 전체 도착률
            if len(self.lat_hits) >= self.min_eps and lat_rate >= self.success and self.radius < self.max_radius:
                self.radius += 1
                self.lat_hits = []
                self.training_env.env_method("set_radius", self.radius)
                changed = True
        if changed:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_path.write_text(json.dumps({"ptr": self.ptr, "radius": self.radius, "steps": self.num_timesteps}))

    def _on_step(self):
        return True


class MultiDemoCurriculumCallback(BaseCallback):
    """여러 구간 섞어 학습(GOAL1)용: DemoCurriculumCallback 규칙을 구간마다 따로 적용

    왜 따로 만드나
    - 기존 콜백은 환경 전부가 전선 하나를 공유 (구간 하나라서)
    - 섞어 학습에선 구간마다 시연 길이도, 익힌 정도도 다름 → 구간별 전선(ptr)과 폭(radius)
    - 환경은 시작할 때 구간이 정해짐 (env.segment_name). 같은 구간 환경끼리 결과를 모아 그 구간 전선만 당김
    - 기록: game/demo_front/<구간>, game/real_goal/<구간> … (구간별 곡선)
    - 저장: runs/curriculum/<실험>/demo_ptr.json = {구간: {ptr, radius, steps}}
    규칙 자체(롤아웃마다, 24판 이상 + 도착 ≥ 50%면 3칸 당김 / 폭 +1)는 DemoCurriculumCallback과 같음
    """

    def __init__(self, save_path, success=0.5, min_eps=24, step_back=3, max_radius=8):
        super().__init__()
        self.save_path = Path(save_path)
        self.success, self.min_eps, self.step_back, self.max_radius = success, min_eps, step_back, max_radius

    def _on_training_start(self):
        venv = self.training_env
        names = venv.get_attr("segment_name")
        self.groups = {}
        for i, n in enumerate(names):
            self.groups.setdefault(n, []).append(i)
        saved = json.loads(self.save_path.read_text()) if self.save_path.exists() else {}
        self.state = {}
        for n, idx in self.groups.items():
            length = venv.get_attr("demo_len", [idx[0]])[0]
            st = saved.get(n, {})
            self.state[n] = {"n": length, "window": venv.get_attr("window", [idx[0]])[0],
                             "ptr": st.get("ptr", length - 1), "radius": st.get("radius", 1),
                             "hits": [], "lat_hits": []}
            venv.env_method("set_ptr", self.state[n]["ptr"], indices=idx)
            venv.env_method("set_radius", self.state[n]["radius"], indices=idx)
            if st:
                print(f"[시연 전선 {n}] 이어서: 목표에서 {length - st['ptr']}걸음 앞부터 (폭 {st.get('radius', 1)})")

    def _on_rollout_end(self):
        venv = self.training_env
        changed = False
        for n, idx in self.groups.items():
            s = self.state[n]
            for results in venv.env_method("pop_results", indices=idx):
                s["hits"] += [ok for i, ok in results if s["ptr"] <= i <= s["ptr"] + s["window"]]
            lat = [r for results in venv.env_method("pop_lat_results", indices=idx) for r in results]
            s["lat_hits"] += [ok for pd, ok in lat if pd == s["radius"]]
            real = [ok for results in venv.env_method("pop_real_results", indices=idx) for ok in results]
            rate = sum(s["hits"]) / len(s["hits"]) if s["hits"] else 0.0
            lat_rate = sum(s["lat_hits"]) / len(s["lat_hits"]) if s["lat_hits"] else 0.0
            self.logger.record(f"game/demo_front/{n}", s["n"] - s["ptr"])
            self.logger.record(f"game/demo_front_rate/{n}", rate)
            self.logger.record(f"game/lat_radius/{n}", s["radius"])
            self.logger.record(f"game/lat_rate/{n}", lat_rate)
            if lat:
                self.logger.record(f"game/lat_goal/{n}", sum(ok for _, ok in lat) / len(lat))
            if real:
                self.logger.record(f"game/real_goal/{n}", sum(real) / len(real))
            if len(s["hits"]) >= self.min_eps and rate >= self.success and s["ptr"] > 0:
                s["ptr"] = max(0, s["ptr"] - self.step_back)
                s["hits"] = []
                venv.env_method("set_ptr", s["ptr"], indices=idx)
                changed = True
            if len(s["lat_hits"]) >= self.min_eps and lat_rate >= self.success and s["radius"] < self.max_radius:
                s["radius"] += 1
                s["lat_hits"] = []
                venv.env_method("set_radius", s["radius"], indices=idx)
                changed = True
        if changed:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_path.write_text(json.dumps({n: {"ptr": s["ptr"], "radius": s["radius"], "steps": self.num_timesteps}
                                                  for n, s in self.state.items()}))

    def _on_step(self):
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", default="v1_explore", choices=REWARD_PRESETS.keys())
    parser.add_argument("--env", default="home", choices=[*ENV_PRESETS, *CURRICULUM_PRESETS])
    # 라이브 창: 기본으로 같이 띄움 (첫 체크포인트가 생기면 자동으로 보여주기 시작)
    parser.add_argument("--no-live", action="store_true", help="라이브 창 안 띄우기")
    parser.add_argument("--name", default=None, help="실험 이름 (기본: 보상 프리셋 이름)")
    parser.add_argument("--steps", type=int, default=1_000_000)
    # 코어 8개 중 6개만 환경에 씀. 나머지는 PPO 신경망 업데이트랑 윈도우가 쓰게 남겨둠
    parser.add_argument("--envs", type=int, default=6)
    parser.add_argument("--resume", default=None,
                        help="이어서 학습할 모델 .zip 경로, 또는 latest (models/<name>/에서 가장 최근 파일)")
    # --steps는 "이번 실행에서 추가로 몇 스텝"이라 이어서 할 때 남은 양을 손으로 계산해야 했음 (실수하기 쉬움)
    # --total-steps를 주면 "누적 목표"로 보고, 이어서 할 때 남은 만큼만 자동으로 학습함
    parser.add_argument("--total-steps", type=int, default=None, help="누적 목표 스텝 (주면 --steps 대신 씀)")
    # 보상 말고 하이퍼파라미터를 바꾸는 실험용 (v2: 탐험 부족이 원인인지 확인)
    # 새로 학습할 때만 적용됨. --resume이면 저장된 모델의 값을 그대로 씀
    parser.add_argument("--ent-coef", type=float, default=0.01)
    # 조작 방식: tap(8프레임, A2~DEMO) / tile(방향 버튼 16프레임 = 1칸, DEMO_TILE~). envs/gold_env.py 참고
    # 모델 폴더의 env_config.json에 저장 → diagnose, live_watch가 자동으로 같은 방식으로 평가
    parser.add_argument("--move", default=None, choices=["tap", "tile"], help="조작 방식 (기본 tap)")
    # 짧은 기억: 관찰에 "직전 버튼 + 움직였나"를 추가 (MEM 실험~). 정책이 CnnPolicy → MultiInputPolicy로 바뀜
    parser.add_argument("--memory", action="store_true", help="관찰에 직전 버튼 + 움직였나 추가")
    parser.add_argument("--goal", action="store_true", help="관찰에 목표 맵 추가 (GOAL1, 여러 구간 섞어 학습용)")
    # 시드: 같은 설정으로 시드만 바꿔 다시 돌려서 결과가 우연이 아닌지 확인할 때 (예: --seed 1 --name ..._seed1)
    #   새로 학습할 때만 적용. 이어서 할 땐 저장된 모델의 값 그대로
    parser.add_argument("--seed", type=int, default=0)
    # 전이 실험용: 다른 실험의 모델 가중치로 시작하는 "새" 학습 (스텝 수, 커리큘럼, 로그는 처음부터)
    #   --resume과 다른 점: resume은 같은 실험을 이어감, init은 다른 구간 실험의 실력을 출발점으로 빌려옴
    #   신경망 모양이 같아야 함 (조작 방식, 기억 설정을 원래 모델과 맞춤)
    parser.add_argument("--init", default=None, help="이 모델의 가중치로 시작 (예: models/DEMO_TILE_route29/final.zip)")
    args = parser.parse_args()

    name = args.name or args.reward
    save_dir = Path("models") / name

    # 구간 환경이면 필요한 파일(시작 상태, 거리표)이 있는지 먼저 확인
    if args.env in CURRICULUM_PRESETS:
        preset = CURRICULUM_PRESETS[args.env]
        seg = SEGMENTS[preset["base"]]
        # 커리큘럼마다 필요한 파일: 씨앗용 목표 도착 상태(RC, AC 계열) 또는 시연 경로(demo)
        need = [p for p in (preset.get("goal_state_path"), preset.get("demo_path")) if p]
        if not seg.ready() or not all(Path(p).exists() for p in need):
            print(f"커리큘럼 준비 안 됨: 거리표/시작 상태 또는 {need} 없음")
            print(f"  uv run python -m train.build_segment {seg.name} --harvest   (시연이면 --demo)")
            return
    if args.env in SEGMENTS and not SEGMENTS[args.env].ready():
        seg = SEGMENTS[args.env]
        print(f"구간 '{args.env}' 준비 안 됨: 시작 상태 {seg.start_state}, 거리표 {seg.dist_path}")
        print(f"  uv run python -m train.build_segment {args.env}   로 거리표부터 만들면 됨")
        return

    # 안전장치: 새로 학습하는데 같은 이름 폴더에 이미 모델이 있으면 멈춤
    # 그대로 진행하면 기존 실험의 체크포인트와 final.zip을 덮어씀 (v1에서 실제로 겪을 뻔한 일)
    if not args.resume and any(save_dir.glob("*.zip")):
        print(f"{save_dir}에 이미 모델이 있음. --name으로 다른 이름을 주거나, 이어서 하려면 --resume을 쓰면 됨")
        return

    # 조작 방식, 기억: 이어서 할 땐 처음 학습한 설정을 그대로 (도중에 바뀌면 같은 실험이 아님)
    saved = load_env_config(save_dir)
    if args.resume and saved:
        if args.move and args.move != saved.get("move_mode", "tap"):
            print(f"이 실험은 move={saved.get('move_mode', 'tap')}로 학습됨. --move {args.move}는 무시함")
        args.move = saved.get("move_mode", "tap")
        args.memory = saved.get("obs_memory", False)
        args.goal = saved.get("obs_goal", False)
    move_mode = args.move or "tap"
    env_config = {"move_mode": move_mode, "obs_memory": bool(args.memory), "obs_goal": bool(args.goal)}
    save_dir.mkdir(parents=True, exist_ok=True)
    # 기록용: 시드까지 같이 적어둠 (환경 설정은 env_config만 환경에 넘김)
    (save_dir / "env_config.json").write_text(json.dumps({**env_config, "seed": saved.get("seed", args.seed),
                                                         "init": saved.get("init", args.init)}))

    # -----------------------------------------------------------
    # 1) 환경 여러 개
    # -----------------------------------------------------------
    # make_vec_env: GoldEnv를 n개 만들고 각각 Monitor로 감쌈 (1단계에서 직접 하던 일)
    # SubprocVecEnv: 각 환경을 별도 프로세스에서 돌림 → CPU 코어를 나눠 써서 거의 n배 빨라짐
    #   (DummyVecEnv는 한 프로세스에서 차례로 돌려서 빨라지지 않음)
    # 커리큘럼이면 학습 환경 클래스와 설정을 바꿈. base 구간 이름은 라이브/평가에 씀
    parts = None
    if args.env in CURRICULUM_PRESETS and CURRICULUM_PRESETS[args.env].get("_class") == "multi_lat":
        # 여러 구간 섞기: 환경 i는 parts[i % 구간 수]를 맡음 (6개, 2구간이면 3개씩)
        parts = CURRICULUM_PRESETS[args.env]["parts"]
        base_env = CURRICULUM_PRESETS[args.env]["base"]
        env_cls, env_kwargs = LateralDemoEnv, None
    elif args.env in CURRICULUM_PRESETS:
        rc = dict(CURRICULUM_PRESETS[args.env])
        base_env = rc.pop("base")
        cls = {"archive": ArchiveCurriculumEnv, "frontier": FrontierArchiveEnv,
               "demo": DemoReverseEnv, "demo_lat": LateralDemoEnv}.get(rc.pop("_class", None), ReverseCurriculumEnv)
        if cls in (ArchiveCurriculumEnv, FrontierArchiveEnv):
            # 아카이브(익힌 자리 포함)를 실험 이름별 폴더에 저장 → --resume 때 이어서 씀
            rc["archive_dir"] = str(Path("runs/curriculum") / name)
        env_cls, env_kwargs = cls, {**ENV_PRESETS[base_env], **rc}
    else:
        base_env = args.env
        env_cls, env_kwargs = GoldEnv, ENV_PRESETS[args.env]

    if parts:
        # make_vec_env는 모든 환경에 같은 설정을 주니까 직접 만듦 (Monitor로 감싸는 것도 같게)
        from stable_baselines3.common.monitor import Monitor

        def make(i):
            kw = {"reward_cfg": REWARD_PRESETS[args.reward], **lat_kwargs(parts[i % len(parts)]), **env_config}
            return lambda: Monitor(LateralDemoEnv(**kw))
        env = SubprocVecEnv([make(i) for i in range(args.envs)])
        env.seed(args.seed)
    else:
        env = make_vec_env(
            env_cls,
            n_envs=args.envs,
            vec_env_cls=SubprocVecEnv,
            env_kwargs={"reward_cfg": REWARD_PRESETS[args.reward], **env_kwargs, **env_config},
            seed=args.seed,
        )

    # -----------------------------------------------------------
    # 2) PPO
    # -----------------------------------------------------------
    if args.resume:
        if args.resume == "latest":
            # 중단 방식과 상관없이 가장 최근 파일을 고름
            # (Ctrl+C로 멈추면 final.zip이 생기고, 강제 종료/정전이면 마지막 체크포인트만 남음)
            zips = list(save_dir.glob("*.zip"))
            if not zips:
                print(f"{save_dir}에 이어서 할 모델이 없음")
                env.close()
                return
            args.resume = str(max(zips, key=lambda p: p.stat().st_mtime))
        # 저장된 신경망 + 하이퍼파라미터를 그대로 불러와서 새 환경에 연결
        model = PPO.load(args.resume, env=env, tensorboard_log="runs")
        print(f"이어서 학습: {args.resume} (지금까지 {model.num_timesteps:,}스텝)")
    else:
        model = PPO(
            # 관찰이 이미지라서 CNN. 화면에서 벽/길/NPC 같은 모양을 뽑아낸 뒤 행동 확률을 냄
            # 짧은 기억을 쓰면 관찰이 Dict(화면 + 숫자)라 MultiInputPolicy (화면은 같은 CNN, 숫자는 그대로 이어 붙임)
            "MultiInputPolicy" if (args.memory or args.goal) else "CnnPolicy",
            env,
            # 신경망 모양을 CnnPolicy와 똑같이: 화면 CNN 출력 512 (MultiInputPolicy 기본 256),
            #   CNN 뒤 은닉층 없음 (MultiInputPolicy 기본은 64-64를 붙임, CnnPolicy는 안 붙임)
            #   → 기억 실험에서 바뀌는 건 "숫자 7개가 더 들어감"뿐이 되게 (한 번에 하나만 바꾸기)
            #   목표 정보(--goal)도 같은 방식: 바뀌는 건 "목표 숫자 96개가 더 들어감"뿐
            policy_kwargs={"features_extractor_kwargs": {"cnn_output_dim": 512}, "net_arch": []} if (args.memory or args.goal) else None,
            # n_steps: 환경 하나당 몇 스텝 모으고 업데이트할지
            #   → 한 번 업데이트에 512 x 6 = 3072스텝 경험을 씀
            n_steps=512,
            batch_size=256,  # 3072를 256씩 12조각으로 나눠 학습
            n_epochs=4,      # 같은 경험으로 몇 번 반복 학습할지
            # gamma: 미래 보상을 얼마나 쳐줄지 (CartPole 기본값은 0.99)
            #   포켓몬은 "지금 이 길로 가면 수백 스텝 뒤에 새 맵"처럼 보상이 멀리 있어서 조금 더 크게
            #   0.99면 약 100스텝 뒤, 0.995면 약 200스텝 뒤 보상까지 의미 있게 반영됨
            gamma=0.995,
            # ent_coef: 행동을 너무 빨리 하나로 굳히지 않게 하는 힘 (탐험 유지)
            #   0이면 초반에 "벽에 대고 위 연타" 같은 걸로 굳어버리기 쉬움
            ent_coef=args.ent_coef,
            learning_rate=2.5e-4,
            verbose=1,
            tensorboard_log="runs",
            seed=args.seed,
        )
        if args.init:
            src_cfg = load_env_config(Path(args.init).parent)
            if (src_cfg.get("obs_memory", False) != bool(args.memory)
                    or src_cfg.get("obs_goal", False) != bool(args.goal)):
                print("--init 모델과 --memory/--goal 설정이 달라서 신경망 모양이 안 맞음")
                env.close()
                return
            if src_cfg.get("move_mode", "tap") != move_mode:
                print(f"주의: --init 모델은 move={src_cfg.get('move_mode', 'tap')}로 학습됨 (지금 {move_mode})")
            # 정책/가치 신경망 가중치만 복사 (옵티마이저 상태, 스텝 수는 새로)
            model.policy.load_state_dict(PPO.load(args.init, device="cpu").policy.state_dict())
            print(f"가중치 시작점: {args.init}")

    # -----------------------------------------------------------
    # 3) 학습
    # -----------------------------------------------------------
    # 이번 실행에서 학습할 양
    if args.total_steps is not None:
        done_steps = model.num_timesteps if args.resume else 0
        steps_to_run = args.total_steps - done_steps
        if steps_to_run <= 0:
            print(f"이미 목표({args.total_steps:,}스텝)에 도달함 ({done_steps:,}스텝)")
            env.close()
            return
        print(f"누적 목표 {args.total_steps:,} → 이번에 {steps_to_run:,}스텝 학습")
    else:
        steps_to_run = args.steps

    callbacks = [
        # save_freq는 "환경 1개 기준 스텝"이라 환경 수로 나눔 → 전체 약 5만 스텝마다 저장
        CheckpointCallback(save_freq=max(50_000 // args.envs, 1), save_path=str(save_dir), name_prefix="ppo"),
        GameStatsCallback(),
    ]
    if parts:
        callbacks.append(MultiDemoCurriculumCallback(Path("runs/curriculum") / name / "demo_ptr.json"))
    elif issubclass(env_cls, DemoReverseEnv):
        callbacks.append(DemoCurriculumCallback(Path("runs/curriculum") / name / "demo_ptr.json"))
    # 라이브 창 자동 실행 (학습과 별도 프로세스라, 창을 닫아도 학습은 계속됨)
    # 커리큘럼 학습이어도 라이브는 보통 실제 시작 상태(base 구간)에서 보여줌 = 평가와 같은 조건
    # 역방향 시연이면 라이브도 전선 위치에서 시작 (실제 입구는 몇 판에 한 번)
    if not args.no_live:
        live_env = args.env if issubclass(env_cls, DemoReverseEnv) else base_env
        callbacks.append(LiveSnapshotCallback(save_dir / "live.zip"))
        subprocess.Popen([sys.executable, "-m", "train.live_watch", name, "--env", live_env, "--wait"])
        print("라이브 창: 약 1만 스텝 뒤 첫 모습이 뜸 (live.zip). 창을 닫으면 라이브만 꺼짐")

    try:
        model.learn(
            total_timesteps=steps_to_run,
            callback=callbacks,
            tb_log_name=name,                      # runs/<name>_1/ 에 곡선 저장
            reset_num_timesteps=not args.resume,   # 이어서 할 땐 스텝 수를 이어 붙임
        )
    except KeyboardInterrupt:
        # Ctrl+C로 중간에 멈춰도 그때까지 학습한 걸 저장하고 끝냄
        print("\n중단됨 → 지금까지 학습한 모델을 저장함")

    # -----------------------------------------------------------
    # 4) 저장
    # -----------------------------------------------------------
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(save_dir / "final")
    print(f"저장: {save_dir / 'final.zip'}")
    env.close()


# 윈도우에서 SubprocVecEnv를 쓰려면 이 가드가 꼭 필요함
# 자식 프로세스가 이 파일을 다시 import하는데, 가드가 없으면 자식도 main()을 실행해서
# 프로세스가 끝없이 생기다 에러가 남
if __name__ == "__main__":
    main()
