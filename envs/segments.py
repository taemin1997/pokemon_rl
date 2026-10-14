"""
구간(segment) 정의: 게임 전체를 "시작 상태 → 목표 맵" 단위로 잘라서 하나씩 학습

왜 이렇게 나누나 (확장성)
- 전당등록까지 한 번에 학습하는 건 CPU로 불가능 → 구간별로 학습하고 이어 붙임
- 구간마다 필요한 것은 전부 같음: 시작 상태, 벗어나면 실패인 맵, 도착하면 성공인 맵, 거리표
  → 이걸 데이터 한 줄로 적어두면 학습/진단/라이브 보기/거리표 생성 스크립트가 전부 그대로 동작함
- 새 구간 추가 절차 (자세한 건 progress/segments.md)
  1. 이전 구간의 도착 상태를 다음 구간 시작으로 (train/build_segment.py --harvest 가 자동 저장)
  2. 여기에 Segment 한 줄 추가
  3. uv run python -m train.build_segment <이름>   → 거리표 자동 생성
  4. uv run python -m train.train_ppo --env <이름> --reward task ...

조건(스토리 이벤트, 비전 기술)을 따로 모델링하지 않는 이유
- 거리표를 "그 조건이 갖춰진 세이브 상태"에서 PyBoy로 직접 탐색해서 만들기 때문
- 예: NPC가 길을 막는 구간은 길이 열린 뒤의 세이브에서 거리표를 만들면 됨
"""

from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path("envs/data")
STATE_DIR = Path("states/segments")


@dataclass
class Segment:
    name: str
    allowed_maps: list            # 이 맵들 밖으로 나가면 실패 종료 (BFS 탐색 범위이기도 함)
    goal_maps: list               # 이 맵에 들어가면 성공 종료
    start_state: str = None       # 기본: states/segments/<이름>/start.state
    max_steps: int = 512
    auto_battle: bool = True      # 전투는 자동 처리 (전투 모듈이 생기기 전까지)
    eval_states: dict = field(default_factory=dict)  # 진단용 추가 시작 상태 {이름: 경로}
    compose: list = None          # 평가용 구간: 이 구간들의 거리표를 이어 붙여서 씀 (BFS를 새로 안 돌림)
    note: str = ""                # 사람이 읽는 메모 (스토리 조건 등)
    eval_only: bool = False       # 평가 전용 (E26 10-14): 학습·통합 실행 구간 선택에서 뺌
    dist_of: str = None           # 거리표를 다른 구간 것으로 (같은 목표·같은 맵. 진단 지표 전용, 보상 아님)
    demo_file: str = None         # 시연 파일 경로 덮어쓰기 (E26: 실험별 폴더에 둔 Go-Explore 경로)
    # 트레이너가 길을 막는 구간 (M2b부터): 탐색 중 시야에 걸려 잠기면 "막힌 칸"이 아니라 대화+전투를 A로 넘기고 지나감
    trainers: bool = False

    def __post_init__(self):
        if self.start_state is None:
            self.start_state = str(STATE_DIR / self.name / "start.state")

    @property
    def dist_path(self):
        return str(DATA_DIR / f"{self.dist_of or self.name}_dist.json")

    @property
    def edges_path(self):
        return str(DATA_DIR / f"{self.name}_edges.json")

    @property
    def goal_state_path(self):
        """이 구간 목표에 도착한 순간의 상태. 다음 구간의 시작 상태 후보"""
        return str(STATE_DIR / self.name / "reached_goal.state")

    @property
    def demo_path(self):
        """시작 → 목표 경로 위의 상태들 (build_segment --demo). 역방향 시연 학습용"""
        return self.demo_file or str(STATE_DIR / self.name / "demo.pkl")

    def ready(self):
        """학습에 필요한 파일이 다 있나"""
        return Path(self.start_state).exists() and Path(self.dist_path).exists()

    def env_kwargs(self):
        """GoldEnv에 넘길 설정"""
        return {
            "state_path": self.start_state,
            "max_steps": self.max_steps,
            "allowed_maps": self.allowed_maps,
            "goal_maps": self.goal_maps,
            "auto_battle": self.auto_battle,
            "dist_path": self.dist_path,
        }

    def all_eval_states(self):
        return {"start": self.start_state, **self.eval_states}


# ---------------------------------------------------------------
# 구간 목록 (진행 순서대로)
# ---------------------------------------------------------------
SEGMENTS = {
    # M1 앞부분: 집 1층 → 연두마을 → 29번도로 입구
    # 시작 상태는 v1부터 쓰던 집 1층 (포켓몬 받은 뒤). 마을엔 야생 포켓몬이 없어서 전투 없음
    # route29 구간(입구 → 무궁시티) 앞에 붙여서 "연두마을 → 무궁시티" 전체를 이어서 검증하는 용도
    "newbark": Segment(
        name="newbark",
        allowed_maps=[(24, 6), (24, 4)],       # 집 1층, 연두마을 (연구소/옆집/2층에 들어가면 실패)
        goal_maps=[(24, 3)],                    # 29번도로에 들어서면 성공
        start_state="states/start.state",
        note="집 1층 (9,2)에서 시작. 연두마을 서쪽 끝으로 나가면 29번도로",
    ),
    # M1: 29번도로 → 무궁시티
    # 기존 실험(A, A2, B)과 같은 파일을 쓰려고 경로를 직접 지정
    "route29": Segment(
        name="route29",
        allowed_maps=[(24, 3)],
        goal_maps=[(26, 3)],
        start_state="states/eval/S1_entry.state",
        eval_states={
            "S2_deadend": "states/eval/S2_deadend.state",
            "S3_fork": "states/eval/S3_fork.state",
        },
        note="시작: 29번도로 입구 (59,9). 최단 103걸음",
    ),
    # M2 첫 구간: 무궁시티 → 30번도로
    # 시작 상태는 route29를 --harvest로 자동 생성 (무궁시티에 막 들어온 순간)
    "cherrygrove": Segment(
        name="cherrygrove",
        allowed_maps=[(26, 3)],
        goal_maps=[(26, 1)],
        note="무궁시티 안에서 북쪽 30번도로로. 건물은 허용 안 함 (들어가면 실패 종료)",
    ),
    # M2 둘째 구간: 30번도로 남쪽 입구 → 포켓몬할아버지 집 (알 받기 이벤트)
    #   시작 상태: cherrygrove를 --harvest --avoid-battles로 자동 생성 (HP 16/20)
    #   30번도로엔 트레이너가 있음 (눈 마주치면 강제 전투) → 전투 모듈 B2 검증용 트레이너 전투 상태도 여기서
    #   열매집(26,9)은 허용 안 함 (들어가면 실패)
    "route30": Segment(
        name="route30",
        allowed_maps=[(26, 1)],
        goal_maps=[(26, 10)],
        note="30번도로 (7,53) → 포켓몬할아버지 집. 트레이너 시야, 스토리 이벤트 확인 필요",
    ),
    # ---- 전멸 대응: 부활 위치에서 출발하는 구간 (원칙: 전멸은 지표로만, 최종 플레이는 부활 위치에서 이어감) ----
    # 집 2층 (전멸 → 마지막 회복 장소 = 집 2층, HP 가득, 돈 절반) → 집 1층. 그 뒤는 newbark 구간
    #   시작 상태: 29번도로 전투에서 HP를 1로 써넣고 진 상태 (데이터 생성용 비계, 10-08)
    "house2f": Segment(
        name="house2f",
        allowed_maps=[(24, 7)],
        goal_maps=[(24, 6)],
        start_state="states/segments/respawn_home.state",
        note="전멸 부활 위치 (24,7,3,3) → 계단 → 집 1층",
    ),
    # ---- 회복 (상위 계획 규칙: 새 마을 첫 도착 또는 HP 절반 미만이면 센터, 10-08 사용자 제안) ----
    # 무궁시티 (29번도로 쪽 입구) → 포켓몬센터 안. 그 뒤 간호사 대화는 매크로 (envs/heal.py)
    "cherrygrove_pc": Segment(
        name="cherrygrove_pc",
        allowed_maps=[(26, 3)],
        goal_maps=[(26, 5)],
        start_state="states/segments/route29/reached_goal.state",
        note="무궁시티 도착 상태 → 포켓몬센터 문",
    ),
    # 포켓몬센터 안 (회복 직후 = 전멸 부활 위치이기도 함) → 무궁시티
    "cherrygrove_pc_exit": Segment(
        name="cherrygrove_pc_exit",
        allowed_maps=[(26, 5)],
        goal_maps=[(26, 3)],
        note="센터 안 간호사 앞 → 문 → 무궁시티",
    ),
    # ---- M2a 돌아오는 길 (포켓몬할아버지 집 → 공박사 연구소). 스토리상 알/도감 이벤트 뒤에 지나감 ----
    #   지금 시작 상태는 이벤트 전 상태 (이동 학습용). 이벤트 후 상태가 생기면 바꿔 끼움 (길은 같음)
    # 할아버지 집 → 30번도로 → 무궁시티. 시작: route30 도착 상태 (집 안)
    "route30_rev": Segment(
        name="route30_rev",
        allowed_maps=[(26, 10), (26, 1)],
        goal_maps=[(26, 3)],
        note="포켓몬할아버지 집 안 → 30번도로 남쪽 → 무궁시티. 열매집(26,9) 들어가면 실패",
    ),
    # 무궁시티 북쪽 입구 → 29번도로 (동쪽 출구). 시작: route30_rev 도착 상태. 돌아올 때 라이벌전이 여기서 (이벤트 뒤)
    "cherrygrove_rev": Segment(
        name="cherrygrove_rev",
        allowed_maps=[(26, 3)],
        goal_maps=[(24, 3)],
        note="무궁시티 (30번도로 쪽) → 29번도로. 건물 들어가면 실패",
    ),
    # 연두마을 서쪽 입구 → 공박사 연구소. 시작: route29_rev 도착 상태
    "newbark_lab": Segment(
        name="newbark_lab",
        allowed_maps=[(24, 4)],
        goal_maps=[(24, 5)],
        note="연두마을 (29번도로 쪽) → 공박사 연구소 문",
    ),
    # GOAL1용: 29번도로를 반대로 (서쪽 끝 → 연두마을). 스토리상 "할아버지 집 → 공박사 연구소" 귀환 때 같은 길을 동쪽으로 감
    #   같은 화면에서 정답 방향이 route29와 반대 → 화면만 보는 정책은 둘 다 못 함 (목표 정보 시험)
    #   시작 상태: route29 시연의 마지막 칸 (0,7) 상태 (무궁시티 바로 앞, 시연과 같은 HP)
    "route29_rev": Segment(
        name="route29_rev",
        allowed_maps=[(24, 3)],
        goal_maps=[(24, 4)],
        note="29번도로 서쪽 끝 (0,7) → 연두마을. 턱은 동쪽으로 못 넘는 방향이 있어 route29와 길이 다를 수 있음",
    ),
    # ---- M2b (10-10): 연구소 이벤트 뒤 → 31번도로 → 도라지시티 → 체육관 ----
    # 30번도로 남쪽 입구 → 31번도로. 시작: M2a 끝(연구소 이벤트 뒤) 상태에서 run_story로 30번도로 입구까지 (무궁 센터 회복, HP 가득)
    #   route30(할아버지 집 목표)과 같은 길을 북쪽 끝까지 더 감 → 할아버지 집 갈림길이 새 분기점
    "route30_north": Segment(
        name="route30_north",
        trainers=True,
        allowed_maps=[(26, 1)],
        goal_maps=[(26, 2)],
        note="30번도로 (무궁시티 쪽) → 31번도로. 할아버지 집/열매집 들어가면 실패. 트레이너 있음",
    ),
    # 31번도로 (30번도로 쪽) → 게이트 → 도라지시티. 어두운동굴 입구 들어가면 실패
    "route31": Segment(
        name="route31",
        trainers=True,
        allowed_maps=[(26, 2), (26, 11)],
        goal_maps=[(10, 5)],
        start_state="states/segments/route30_north/reached_goal.state",
        note="31번도로 → 31번-도라지 게이트 → 도라지시티. 턱 있음",
    ),
    # 도라지시티 (서쪽 게이트 쪽) → 체육관 문
    "violet_gym": Segment(
        name="violet_gym",
        trainers=True,
        allowed_maps=[(10, 5)],
        goal_maps=[(10, 7)],
        start_state="states/segments/route31/reached_goal.state",
        note="도라지시티 → 도라지체육관 문. 다른 건물 들어가면 실패",
    ),
    # 도라지시티 → 포켓몬센터 (회복 규칙: 새 마을 첫 도착)
    "violet_pc": Segment(
        name="violet_pc",
        trainers=True,
        allowed_maps=[(10, 5)],
        goal_maps=[(10, 10)],
        start_state="states/segments/route31/reached_goal.state",
        note="도라지시티 → 도라지 포켓몬센터",
    ),
    # 센터 안 → 도라지시티
    "violet_pc_exit": Segment(
        name="violet_pc_exit",
        trainers=True,
        allowed_maps=[(10, 10)],
        goal_maps=[(10, 5)],
        start_state="states/segments/violet_pc/reached_goal.state",
        note="도라지 포켓몬센터 안 → 도라지시티",
    ),
    # (진행 순서와 무관한 평가용 구간은 맨 뒤에 둠: --harvest의 '다음 구간' 자동 연결이 순서를 봄)
    # 평가용: 집 1층 → 무궁시티 한 판 (newbark + route29를 이어 붙였을 때 끝까지 가나)
    #   학습엔 안 씀. train.diagnose에 chain: 정책을 주면 구간 경계에서 모델을 바꿔가며 평가
    "newbark_cherrygrove": Segment(
        name="newbark_cherrygrove",
        allowed_maps=[(24, 6), (24, 4), (24, 3)],   # 집 1층, 연두마을, 29번도로
        goal_maps=[(26, 3)],                         # 무궁시티
        start_state="states/start.state",
        max_steps=768,                               # 두 구간 합 (24 + 103걸음) 대비 넉넉히
        compose=["newbark", "route29"],              # 거리표 = 두 구간 거리표 이어 붙이기 (아래 build_segment 참고)
        note="평가 전용. newbark → route29 이어 붙이기 검증",
    ),
    # E26 평가 전용 (10-14): 구간을 이을 때 생기는 미학습 경계 출발 상태. **학습 데이터에 넣지 않음** (states/e26/eval/)
    #   이탈(다른 맵으로 들어감) = 실패 → 규칙·매크로 없이 정책만 잼
    "newbark_labdoor": Segment(
        name="newbark_labdoor",
        allowed_maps=[(24, 4)],                      # 연두마을 (연구소로 다시 들어가면 이탈 = 실패)
        goal_maps=[(24, 3)],                         # 29번도로 (newbark 구간과 같은 목표)
        start_state="states/e26/eval/e_lab.state",
        dist_of="newbark",
        eval_only=True,
        note="E-lab: 연구소 이벤트·공박사 대화·나가는 길 이벤트 뒤 연두마을로 막 나온 상태 (10-12 판정 3 실패 지점)",
    ),
    "violet_pc_counter": Segment(
        name="violet_pc_counter",
        allowed_maps=[(10, 10)],                     # 도라지 포켓몬센터
        goal_maps=[(10, 5)],                         # 도라지시티 (violet_pc_exit와 같은 목표)
        start_state="states/e26/eval/e_violet.state",
        dist_of="violet_pc_exit",
        eval_only=True,
        note="E-violet (전이 평가 전용, 어떤 학습에도 금지): 도라지 센터 회복 매크로 직후 간호사 앞",
    ),
}


# ---------------------------------------------------------------
# 실험 중에 자동으로 만든 구간 (E26 트랙 A, 10-14)
#   환경변수 POKE_EXTRA_SEGMENTS = JSON 파일 경로 (구간 정의 목록). 학습 병렬 프로세스까지 같은 구간을 보게 환경변수로
#   사람이 이 파일을 쓰지 않음: train/selfrepair.py가 실패 감지 → Go-Explore 뒤에 씀
# ---------------------------------------------------------------
def _load_extra_segments():
    import json
    import os
    path = os.environ.get("POKE_EXTRA_SEGMENTS")
    if not path or not Path(path).exists():
        return
    for d in json.loads(Path(path).read_text(encoding="utf-8")):
        d = dict(d)
        d["allowed_maps"] = [tuple(m) for m in d["allowed_maps"]]
        d["goal_maps"] = [tuple(m) for m in d["goal_maps"]]
        SEGMENTS[d["name"]] = Segment(**d)


_load_extra_segments()
