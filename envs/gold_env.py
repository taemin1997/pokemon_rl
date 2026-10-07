"""
3단계: 포켓몬 골드 Gym 환경

CartPole에서는 gym.make("CartPole-v1") 한 줄로 환경이 생겼음
포켓몬은 그런 게 없으니까, CartPole이 해주던 일을 직접 채워야 함:
  - observation_space / action_space : 에이전트가 뭘 보고, 뭘 할 수 있는지
  - reset()  : 에피소드 시작 상태로 되돌리기 → 세이브 스테이트 불러오기
  - step()   : 버튼 누르고 → 몇 프레임 진행하고 → 화면(관찰) + 보상 돌려주기
이 규격만 맞추면 1단계의 PPO 코드를 거의 그대로 붙일 수 있음 (그게 Gym 규격을 쓰는 이유)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pyboy import PyBoy

from envs import memory_map as mm

# ---------------------------------------------------------------
# 행동: 에이전트가 고를 수 있는 버튼
# ---------------------------------------------------------------
# START/SELECT를 뺀 이유:
#   - START는 메뉴를 염. 랜덤 탐험 초반에 메뉴 열고 닫기만 반복하면서 시간을 날리기 쉬움
#   - 메뉴 안의 "레포트"로 게임 안 세이브를 덮어쓸 수도 있음
#   - 첫 배지까지는 메뉴 없이도 갈 수 있음 (회복은 포켓몬센터에서 A로 대화)
ACTIONS = ["up", "down", "left", "right", "a", "b"]

# ---------------------------------------------------------------
# 한 번의 step() = 몇 프레임?
# ---------------------------------------------------------------
# 1프레임마다 행동을 고르면 에이전트 입장에선 "아무것도 안 변하는" 스텝이 대부분이라 학습이 안 됨
# 한 칸 걷는 데 약 16프레임 → 버튼을 8프레임 누르고 떼고, 나머지는 걸음이 끝날 때까지 기다림
# 24프레임 = 약 0.4초 = 사람이 버튼 한 번 누르는 템포와 비슷함
PRESS_FRAMES = 8
STEP_FRAMES = 24

# move_mode="tile" (DEMO_TILE부터): 방향 버튼만 16프레임 누르고, 한 걸음에 40프레임
#   왜: 8프레임은 "짧게 누르기"라서, 보고 있는 방향과 다른 방향이면 몸만 돌고 안 걸음
#       → 방향을 자주 바꾸는 확률적 정책은 걸음의 절반 이상을 제자리 회전에 씀 (DEMO 전선 판 56%)
#   16프레임 누르기는 거리표 BFS(build_segment)가 쓰는 방식과 같음 → "방향 버튼 1번 = 1칸 이동"
#   40프레임 = 방향 전환 + 한 칸 걷기가 끝날 때까지. 대신 걸음당 게임 시간이 길어져서 학습이 조금 느려짐
TILE_PRESS_FRAMES = 16
TILE_STEP_FRAMES = 40
MOVE_BUTTONS = ("up", "down", "left", "right")
# 목표 관찰 크기 (obs_goal): 맵 그룹/번호 원핫. 금 버전 맵 그룹은 26개, 그룹 안 번호는 수십 개라 넉넉히 고정
GOAL_GROUPS = 32
GOAL_NUMBERS = 64

# ---------------------------------------------------------------
# 보상 설계 기본값 (v1: 탐험 보상)
# ---------------------------------------------------------------
# 배지만 보상으로 주면 랜덤 행동으로는 배지를 절대 못 땀 → 보상이 영원히 0 → 학습할 게 없음
# 그래서 "처음 밟는 칸"마다 작은 보상을 줘서 일단 멀리 돌아다니게 만드는 게 출발점
# 4단계에서는 GoldEnv(reward_cfg={...})로 이 값들을 실험마다 바꿔 넣음
# (train/train_ppo.py의 REWARD_PRESETS 참고)
DEFAULT_REWARD = {
    "new_tile": 0.1,   # 처음 밟은 (맵, x, y)
    "new_map": 1.0,    # 처음 들어간 맵 (건물, 길 등). 칸 보상에 추가로 줌
    "badge": 100.0,    # 배지 1개당. 최종 목표라 다른 보상보다 압도적으로 크게
    # 거리 shaping (실험 B): 목표까지 남은 걸음이 줄면 +, 늘면 −. 거리표(dist_path)가 있을 때만 작동
    #   보상 = progress × (이전 거리 − 지금 거리)
    #   - 제자리면 0, 왕복하면 합이 0 → 같은 곳에서 보상을 계속 뽑아먹을 수 없음
    #   - x좌표가 아니라 거리표 기준이라, 목표 반대 방향으로 가야 하는 U턴 구간도 올바른 진행이면 +
    #   - γΦ'−Φ 형태(Φ=−거리)는 제자리에서 (1−γ)×거리만큼 보상이 생기는 함정이 있어서 안 씀
    #   0이면 꺼짐 (v1, v2, A2와 똑같이 동작)
    "progress": 0.0,
    # 과제 보상: 목표 맵(goal_maps)에 도착하면 한 번. "무엇을 원하나"를 직접 말하는 항목
    #   B에서 목표 앞에서 뜸을 들인 이유: 도착 = 종료인데 도착 자체를 칭찬하는 항목이 없었음
    "goal": 0.0,
}


class GoldEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        rom_path="roms/gold_rom.gb",
        state_path="states/start.state",
        max_steps=2048,
        render_mode=None,
        reward_cfg=None,
        auto_battle=False,
        allowed_maps=None,
        goal_maps=None,
        dist_path=None,
        move_mode="tap",
        obs_memory=False,
        obs_goal=False,
    ):
        super().__init__()
        self.state_path = state_path
        # 조작 방식: "tap"(기본, A2~DEMO와 같음) 또는 "tile"(방향 버튼 1번 = 1칸). 위 TILE_* 설명 참고
        assert move_mode in ("tap", "tile")
        self.move_mode = move_mode
        # 짧은 기억 (MEM 실험부터): 관찰에 "직전에 누른 버튼 + 실제로 움직였나"를 붙임
        #   왜: 벽에 부딪혀도 화면이 거의 안 바뀜 → 화면만 보는 정책은 방금 막혔다는 걸 모르고 같은 벽을 계속 밈
        #       (DEMO_TILE 전선 판 걸음의 45%가 막힘, 같은 벽 47번 연속)
        #   규칙("막힌 방향 금지")이 아니라 정보만 줌 → "막혔으면 다른 걸 해본다"를 RL이 배우게 함
        #   (규칙으로 막으면 비전머신 나무/바위, 움직이는 NPC, 벽 앞에서 A 같은 경우를 영구히 막아버림)
        self.obs_memory = obs_memory
        # 목표 정보 (GOAL1부터): 관찰에 "이번 판의 목표 맵"을 붙임
        #   왜: 같은 화면(예: 29번도로)에서 목표에 따라 정답 방향이 반대 (무궁시티로 vs 연두마을로)
        #       화면만 보는 정책은 둘 중 하나밖에 못 함 → 여러 구간을 정책 하나로 하려면 필요
        #   주는 것: 목표 맵 번호만 (맵 그룹 원핫 32칸 + 맵 번호 원핫 64칸)
        #   안 주는 것: 좌표, 방향, 거리 (그건 거리 보상처럼 "어느 쪽으로 가라"는 선생님 신호가 됨)
        #   크기를 맵 그룹/번호 최대치로 고정 → 구간이 늘어도 신경망 입력 크기가 안 바뀜
        self.obs_goal = obs_goal
        first_goal = list(goal_maps)[0] if goal_maps else (0, 0)
        self.goal_vec = np.zeros(GOAL_GROUPS + GOAL_NUMBERS, dtype=np.float32)
        self.goal_vec[min(first_goal[0], GOAL_GROUPS - 1)] = 1.0
        self.goal_vec[GOAL_GROUPS + min(first_goal[1], GOAL_NUMBERS - 1)] = 1.0

        # ---- 커리큘럼용 옵션 (기본값은 전부 꺼짐 → v1, v2와 똑같이 동작) ----
        # auto_battle: 전투가 시작되면 A 연타로 자동 처리하고, 에이전트는 전투 끝난 화면부터 다시 고름
        #   이유: 길 찾기를 배우는 동안 전투 비용(무작위 조작 시 판의 절반)을 빼서 두 문제를 분리
        #   주의: "전투가 없는 세계"에서 배운 정책이라, 나중에 전투 정책과 합치면 성능이 달라질 수 있음
        self.auto_battle = auto_battle
        # allowed_maps: 이 맵들 밖으로 나가면 실패로 종료 (예: 29번도로 커리큘럼에서 마을로 돌아가기)
        #   이유: 안 막으면 마을/건물 칸 보상 농장으로 돌아감 (notes/05 진단에서 100% 마을 복귀)
        self.allowed_maps = set(allowed_maps) if allowed_maps else None
        # goal_maps: 이 맵에 들어가면 성공으로 종료
        self.goal_maps = set(goal_maps) if goal_maps else set()
        # dist_path: "목표까지 남은 걸음" 거리표. 보상에는 아직 안 쓰고 진행도 측정(info)용
        self.dist = None
        if dist_path:
            import json
            with open(dist_path) as f:
                self.dist = {tuple(r[:4]): r[4] for r in json.load(f)}
        # 기본값 위에 실험별 값을 덮어씀 → 바꾸고 싶은 항목만 적으면 됨
        self.reward_cfg = {**DEFAULT_REWARD, **(reward_cfg or {})}
        # 에피소드 최대 길이. CartPole의 500스텝 제한(truncated)과 같은 역할
        # 너무 짧으면 멀리 못 가고, 너무 길면 같은 곳을 맴도는 시간이 길어짐 → 4단계 실험 대상
        self.max_steps = max_steps
        self.render_mode = render_mode

        # window="null": 창 없이 실행 → 학습 속도가 수십 배 빨라짐
        # 사람이 보고 싶을 때만 render_mode="human"으로 창을 띄움
        window = "SDL2" if render_mode == "human" else "null"
        self.pyboy = PyBoy(rom_path, window=window)
        # 0 = 속도 제한 없음. 창을 띄울 땐 사람이 볼 수 있게 3배속 정도로
        self.pyboy.set_emulation_speed(3 if render_mode == "human" else 0)

        self.action_space = spaces.Discrete(len(ACTIONS))

        # 관찰: 게임 화면 (흑백, 가로세로 절반으로 축소)
        # 메모리 값(좌표 등)만 주면 벽, NPC, 길을 볼 수 없어서 어디로 가야 할지 판단이 안 됨
        # 원본 144x160 컬러 → 72x80 흑백: 픽셀 수가 1/12이라 CPU 학습에 부담이 적고, 길/벽은 충분히 구분됨
        # 모양이 (높이, 너비, 채널)인 이미지라서 학습 때 "CnnPolicy"를 씀 (CartPole은 숫자 4개라 MLP였음)
        screen_space = spaces.Box(low=0, high=255, shape=(72, 80, 1), dtype=np.uint8)
        if obs_memory or obs_goal:
            # 관찰이 화면 + 숫자 두 종류라 Dict로 → 학습은 MultiInputPolicy (화면은 CNN, 숫자는 작은 MLP로 처리해서 합침)
            #   memory: 직전 버튼 원핫 (6칸, 판 첫 걸음은 전부 0) + 직전 걸음에 좌표가 바뀌었으면 1
            #   goal: 목표 맵 (그룹 원핫 + 번호 원핫)
            parts = {"screen": screen_space}
            if obs_memory:
                parts["memory"] = spaces.Box(low=0.0, high=1.0, shape=(len(ACTIONS) + 1,), dtype=np.float32)
            if obs_goal:
                parts["goal"] = spaces.Box(low=0.0, high=1.0, shape=(GOAL_GROUPS + GOAL_NUMBERS,), dtype=np.float32)
            self.observation_space = spaces.Dict(parts)
        else:
            self.observation_space = screen_space
        self._last_action = None
        self._moved = 0.0

    # -----------------------------------------------------------
    def _get_obs(self):
        screen = self._screen()
        if not (self.obs_memory or self.obs_goal):
            return screen
        obs = {"screen": screen}
        if self.obs_memory:
            mem = np.zeros(len(ACTIONS) + 1, dtype=np.float32)
            if self._last_action is not None:
                mem[self._last_action] = 1.0
            mem[-1] = self._moved
            obs["memory"] = mem
        if self.obs_goal:
            obs["goal"] = self.goal_vec.copy()
        return obs

    def _screen(self):
        # screen.ndarray: (144, 160, 4) RGBA
        rgb = self.pyboy.screen.ndarray[:, :, :3].astype(np.float32)
        gray = rgb.mean(axis=2)          # 3채널 평균 → 흑백
        small = gray[::2, ::2]           # 2칸마다 1칸 → 72x80
        return small.astype(np.uint8)[:, :, None]  # 채널 축 추가 → (72, 80, 1)

    def _get_info(self):
        # info는 학습에 안 쓰이고 사람이 보는 용도 (로그, 디버깅)
        # 4단계에서 "몇 칸 탐험했나, 맵 몇 개 갔나"를 비교할 때 여기 값을 씀
        info = {
            "position": mm.read_position(self.pyboy),
            "visited_tiles": len(self.visited_tiles),
            "visited_maps": len(self.visited_maps),
            "party_count": mm.read_party_count(self.pyboy),
            "money": mm.read_money(self.pyboy),
            "badges": bin(mm.read_badges(self.pyboy)).count("1"),
            # 아래는 실제 게임 진행 지표 (보상과 상관없이 실험끼리 비교하는 용도)
            "battles": self.battles,
            "exp_gain": mm.read_mon1_exp(self.pyboy) - self.exp_start,
            "reached_goal": self.reached_goal,
            "left_area": self.left_area,
            "whiteouts": self.whiteouts,
            "shaping_sum": self.shaping_sum,
        }
        if self.reached_goal:
            # 도착까지 걸린 걸음 (도착한 판에서만 기록 → 평균 내면 "도착한 판의 평균 걸음")
            info["goal_step"] = self.step_count
        if self.dist is not None:
            # 판 중 목표에 가장 가까웠던 거리, 그리고 시작보다 얼마나 줄였나
            info["dist_best"] = self.dist_best
            info["progress"] = self.dist_start - self.dist_best
        return info

    # -----------------------------------------------------------
    def _run_auto_battle(self):
        """전투가 끝날 때까지 A 연타. 커서 기본 위치가 '싸우다' → 첫 번째 기술이라 A만 누르면 공격함
        도중에 지면 화이트아웃(집으로 이동)되고, 그건 allowed_maps 밖이라 종료로 처리됨"""
        self.battles += 1
        frames = 0
        lowest_hp = mm.read_party_hp_total(self.pyboy)
        # 안전장치: 이상한 메뉴에 갇혀도 무한히 돌지 않게 약 100초(6000프레임)에서 끊고 에이전트에게 넘김
        while mm.in_battle(self.pyboy) and frames < 6000:
            self.pyboy.button_press("a")
            self.pyboy.tick(4, False)
            self.pyboy.button_release("a")
            self.pyboy.tick(20, self.render_mode == "human")  # 창으로 볼 땐 전투 화면도 보이게
            frames += 24
            lowest_hp = min(lowest_hp, mm.read_party_hp_total(self.pyboy))
        self.battle_frames += frames
        # 전투 중 파티 HP 합이 0까지 내려갔으면 전멸 → 게임이 집(마지막 회복 장소)으로 보냄
        # 맵 번호로 추측하지 않고 게임 상태(HP)로 판단 → 어느 구간에서든 그대로 작동
        if lowest_hp == 0:
            self.whiteouts += 1

    # -----------------------------------------------------------
    def _load_start(self):
        """판 시작 상태 불러오기. 커리큘럼 환경(envs/curriculum.py)은 이것만 바꿔서 시작 위치를 고름"""
        with open(self.state_path, "rb") as f:
            self.pyboy.load_state(f)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 세이브 스테이트로 되돌림 = CartPole에서 막대를 다시 세우는 것과 같은 역할
        # 매번 오프닝부터 하면 에피소드 대부분이 인트로 대사 넘기기가 돼버림
        self._load_start()

        self.step_count = 0
        self._last_action = None   # 짧은 기억도 판마다 초기화
        self._moved = 0.0
        # 탐험 기록은 에피소드마다 초기화
        # (안 지우면 두 번째 에피소드부터는 이미 가본 칸이라 보상이 안 나와서 학습 신호가 사라짐)
        pos = mm.read_position(self.pyboy)
        self.visited_tiles = {pos}
        self.visited_maps = {pos[:2]}
        self.badges = bin(mm.read_badges(self.pyboy)).count("1")

        self.battles = 0
        self.battle_frames = 0
        self.whiteouts = 0
        self.exp_start = mm.read_mon1_exp(self.pyboy)
        self.reached_goal = False
        self.left_area = False
        if self.dist is not None:
            self.dist_start = self.dist.get(pos, 999)
            self.dist_best = self.dist_start
        # shaping용: 직전에 "거리표에 있는 칸"에 있었을 때의 거리
        self.dist_prev = self.dist.get(pos) if self.dist is not None else None
        self.shaping_sum = 0.0   # 한 판 동안 shaping으로 받은 보상 합 (보상 구성 확인용)

        return self._get_obs(), self._get_info()

    # -----------------------------------------------------------
    def step(self, action):
        button = ACTIONS[action]
        pos_before = mm.read_position(self.pyboy)   # 짧은 기억용: 이번 걸음에 실제로 움직였나

        # 버튼 누르고 PRESS_FRAMES 동안 유지 → 떼고 나머지 프레임 진행
        # render=False: 중간 프레임은 화면을 그릴 필요가 없어서 생략 → 빨라짐
        # 마지막 1프레임만 그려서 관찰로 씀
        if self.move_mode == "tile" and button in MOVE_BUTTONS:
            press, total = TILE_PRESS_FRAMES, TILE_STEP_FRAMES
        else:
            press, total = PRESS_FRAMES, STEP_FRAMES
        self.pyboy.button_press(button)
        self.pyboy.tick(press, False)
        self.pyboy.button_release(button)
        self.pyboy.tick(total - press - 1, False)
        # tick()은 창이 닫히면 False를 돌려줌 → 보는 스크립트(live_watch)가 이걸 보고 종료함
        self.window_open = self.pyboy.tick(1, True)

        # 이번 걸음으로 전투가 시작됐으면 자동 처리 (에이전트 스텝으로는 1번으로 침)
        if self.auto_battle and mm.in_battle(self.pyboy):
            self._run_auto_battle()
            self.pyboy.tick(1, True)  # 관찰용 화면 갱신

        self.step_count += 1

        # 이탈 판정을 보상 계산보다 먼저 함
        here = mm.read_position(self.pyboy)
        self._last_action = int(action)
        self._moved = float(here != pos_before)
        if self.dist is not None and here in self.dist:
            self.dist_best = min(self.dist_best, self.dist[here])
        if here[:2] in self.goal_maps:
            self.reached_goal = True
        elif self.allowed_maps is not None and here[:2] not in self.allowed_maps:
            self.left_area = True

        # 허용 구역 밖으로 나간 걸음에는 보상을 주지 않음
        # (버그였던 것: 마을에 들어가는 순간 "처음 들어간 맵 +1"이 붙어서
        #  "입구에서 몇 걸음 뒤로 나가기 = +1"이 가장 쉬운 보상이 됐음 → A_route29 첫 4만 스텝이 이것만 배움)
        reward = 0.0 if self.left_area else self._compute_reward()
        if self.reached_goal:
            reward += self.reward_cfg["goal"]

        # terminated: 목표 달성으로 끝남 (CartPole의 "막대 쓰러짐" 자리)
        # truncated : 시간 제한으로 끊김 (CartPole의 "500스텝" 자리)
        # 둘을 구분하는 이유: 시간 제한으로 끊긴 건 "여기서 끝나는 상황"이 아니라서
        #   PPO가 그 뒤에도 보상이 이어질 수 있다고 계산해야 함
        # 커리큘럼: 목표 맵 도착(성공)과 허용 구역 이탈(실패)도 "진짜 끝"이라 terminated
        #   이탈을 끝으로 치면, 그 뒤로 받을 보상이 0이 되니까 "나가면 손해"를 배움
        terminated = self.badges >= 1 or self.reached_goal or self.left_area
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # -----------------------------------------------------------
    def _compute_reward(self):
        # 보상 계산을 메서드 하나로 모아둔 이유:
        # 4단계에서 보상 설계만 바꿔가며 비교할 때 여기만 고치면 되게
        cfg = self.reward_cfg
        reward = 0.0

        pos = mm.read_position(self.pyboy)
        if pos not in self.visited_tiles:
            self.visited_tiles.add(pos)
            reward += cfg["new_tile"]
        if pos[:2] not in self.visited_maps:
            self.visited_maps.add(pos[:2])
            reward += cfg["new_map"]

        # 배지는 "늘어난 만큼"만 보상 (이미 가진 배지로 매 스텝 보상 받는 걸 막음)
        badges = bin(mm.read_badges(self.pyboy)).count("1")
        if badges > self.badges:
            reward += cfg["badge"] * (badges - self.badges)
            self.badges = badges

        # 거리 shaping
        if cfg["progress"] and self.dist is not None:
            d_now = self.dist.get(pos)
            # 거리표에 없는 칸(턱 점프 도중의 중간 칸 등)은 건너뜀
            # → 착지한 다음 걸음에 "점프 전 칸 → 착지 칸" 차이로 한 번에 계산됨
            if d_now is not None:
                if self.dist_prev is not None:
                    shaping = cfg["progress"] * (self.dist_prev - d_now)
                    reward += shaping
                    self.shaping_sum += shaping
                self.dist_prev = d_now

        return reward

    # -----------------------------------------------------------
    def close(self):
        # save=False가 중요함: 기본값이면 종료할 때 roms/gold_rom.gb.ram(게임 안 세이브)을 덮어씀
        # 학습 중 에이전트가 만든 상태로 내 세이브 파일이 바뀌면 안 되니까 저장 안 함
        self.pyboy.stop(save=False)
