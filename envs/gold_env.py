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

# ---------------------------------------------------------------
# 보상 설계 v1: 탐험 보상
# ---------------------------------------------------------------
# 배지만 보상으로 주면 랜덤 행동으로는 배지를 절대 못 땀 → 보상이 영원히 0 → 학습할 게 없음
# 그래서 "처음 밟는 칸"마다 작은 보상을 줘서 일단 멀리 돌아다니게 만드는 게 출발점
# 4단계에서 이 값들을 바꾸거나 항목(레벨, 전투 승리 등)을 추가하면서 행동 변화를 비교할 예정
REWARD_NEW_TILE = 0.1   # 처음 밟은 (맵, x, y)
REWARD_NEW_MAP = 1.0    # 처음 들어간 맵 (건물, 길 등). 칸 보상에 추가로 줌
REWARD_BADGE = 100.0    # 배지 1개당. 최종 목표라 다른 보상보다 압도적으로 크게


class GoldEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        rom_path="roms/gold_rom.gb",
        state_path="states/start.state",
        max_steps=2048,
        render_mode=None,
    ):
        super().__init__()
        self.state_path = state_path
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
        self.observation_space = spaces.Box(low=0, high=255, shape=(72, 80, 1), dtype=np.uint8)

    # -----------------------------------------------------------
    def _get_obs(self):
        # screen.ndarray: (144, 160, 4) RGBA
        rgb = self.pyboy.screen.ndarray[:, :, :3].astype(np.float32)
        gray = rgb.mean(axis=2)          # 3채널 평균 → 흑백
        small = gray[::2, ::2]           # 2칸마다 1칸 → 72x80
        return small.astype(np.uint8)[:, :, None]  # 채널 축 추가 → (72, 80, 1)

    def _get_info(self):
        # info는 학습에 안 쓰이고 사람이 보는 용도 (로그, 디버깅)
        # 4단계에서 "몇 칸 탐험했나, 맵 몇 개 갔나"를 비교할 때 여기 값을 씀
        return {
            "position": mm.read_position(self.pyboy),
            "visited_tiles": len(self.visited_tiles),
            "visited_maps": len(self.visited_maps),
            "party_count": mm.read_party_count(self.pyboy),
            "money": mm.read_money(self.pyboy),
            "badges": bin(mm.read_badges(self.pyboy)).count("1"),
        }

    # -----------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 세이브 스테이트로 되돌림 = CartPole에서 막대를 다시 세우는 것과 같은 역할
        # 매번 오프닝부터 하면 에피소드 대부분이 인트로 대사 넘기기가 돼버림
        with open(self.state_path, "rb") as f:
            self.pyboy.load_state(f)

        self.step_count = 0
        # 탐험 기록은 에피소드마다 초기화
        # (안 지우면 두 번째 에피소드부터는 이미 가본 칸이라 보상이 안 나와서 학습 신호가 사라짐)
        pos = mm.read_position(self.pyboy)
        self.visited_tiles = {pos}
        self.visited_maps = {pos[:2]}
        self.badges = bin(mm.read_badges(self.pyboy)).count("1")

        return self._get_obs(), self._get_info()

    # -----------------------------------------------------------
    def step(self, action):
        button = ACTIONS[action]

        # 버튼 누르고 PRESS_FRAMES 동안 유지 → 떼고 나머지 프레임 진행
        # render=False: 중간 프레임은 화면을 그릴 필요가 없어서 생략 → 빨라짐
        # 마지막 1프레임만 그려서 관찰로 씀
        self.pyboy.button_press(button)
        self.pyboy.tick(PRESS_FRAMES, False)
        self.pyboy.button_release(button)
        self.pyboy.tick(STEP_FRAMES - PRESS_FRAMES - 1, False)
        self.pyboy.tick(1, True)

        self.step_count += 1
        reward = self._compute_reward()

        # terminated: 목표 달성으로 끝남 (CartPole의 "막대 쓰러짐" 자리)
        # truncated : 시간 제한으로 끊김 (CartPole의 "500스텝" 자리)
        # 둘을 구분하는 이유: 시간 제한으로 끊긴 건 "여기서 끝나는 상황"이 아니라서
        #   PPO가 그 뒤에도 보상이 이어질 수 있다고 계산해야 함
        terminated = self.badges >= 1
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # -----------------------------------------------------------
    def _compute_reward(self):
        # 보상 계산을 메서드 하나로 모아둔 이유:
        # 4단계에서 보상 설계만 바꿔가며 비교할 때 여기만 고치면 되게
        reward = 0.0

        pos = mm.read_position(self.pyboy)
        if pos not in self.visited_tiles:
            self.visited_tiles.add(pos)
            reward += REWARD_NEW_TILE
        if pos[:2] not in self.visited_maps:
            self.visited_maps.add(pos[:2])
            reward += REWARD_NEW_MAP

        # 배지는 "늘어난 만큼"만 보상 (이미 가진 배지로 매 스텝 보상 받는 걸 막음)
        badges = bin(mm.read_badges(self.pyboy)).count("1")
        if badges > self.badges:
            reward += REWARD_BADGE * (badges - self.badges)
            self.badges = badges

        return reward

    # -----------------------------------------------------------
    def close(self):
        # save=False가 중요함: 기본값이면 종료할 때 roms/gold_rom.gb.ram(게임 안 세이브)을 덮어씀
        # 학습 중 에이전트가 만든 상태로 내 세이브 파일이 바뀌면 안 되니까 저장 안 함
        self.pyboy.stop(save=False)
