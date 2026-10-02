import gymnasium as gym

# render_mode="human" → 창을 띄워서 눈으로 확인
# (학습할 때는 느려지니까 빼고 돌림. 지금은 보는 게 목적이라 켜둠)
env = gym.make("CartPole-v1", render_mode="human")

# 관찰 공간 / 행동 공간 = "에이전트가 뭘 보고, 뭘 할 수 있는지"의 규격서
# - Box(..., (4,), float32): 실수 4개짜리 배열 (카트 위치, 카트 속도, 막대 각도, 막대 각속도)
# - Discrete(2): 0 또는 1 중 하나 (0=왼쪽, 1=오른쪽)
print("관찰 공간:", env.observation_space)
print("행동 공간:", env.action_space)

for episode in range(5):
    # 에피소드 시작마다 환경을 초기 상태로 되돌림
    # reset()은 (첫 관찰, 부가정보) 2개를 돌려줌
    # for 안에 있어야 하는 이유: 끝난 환경에 step()을 계속 하면 안 되기 때문
    obs, info = env.reset()

    total_reward = 0
    done = False

    while not done:
        # 행동을 랜덤으로 뽑음. 아직 "학습"이 없어서 obs를 전혀 안 보고 고름
        # → 나중에 PPO를 붙이면 이 한 줄이 "obs를 보고 고르는" 걸로 바뀜
        action = env.action_space.sample()

        # 행동을 환경에 넣고 결과를 받음. 값 5개:
        # obs        : 행동 후의 새 관찰
        # reward     : 이번 스텝 보상 (CartPole은 버티면 +1)
        # terminated : 규칙상 끝남 (막대가 쓰러짐)
        # truncated  : 시간 제한으로 끊김 (500스텝 도달)
        # info       : 디버깅용 부가정보
        obs, reward, terminated, truncated, info = env.step(action)

        # 에피소드 총점. 강화학습이 키우려는 숫자가 바로 이거
        total_reward += reward

        # 둘 중 하나라도 True면 에피소드 종료
        # terminated만 보면 500스텝을 채웠을 때 루프가 안 끝남
        done = terminated or truncated

    print(f"episode {episode}: total_reward = {total_reward}")

# 창 닫고 자원 정리
env.close()