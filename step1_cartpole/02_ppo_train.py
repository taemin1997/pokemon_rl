import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# ---------------------------------------------------------------
# 1) 학습용 환경
# ---------------------------------------------------------------
# render_mode를 안 줌 → 창 없이 돌아감. 이유: 화면을 그리면 수십 배 느려짐
# Monitor로 감싸는 이유: 에피소드별 점수/길이를 기록해줘서
#                       학습 로그에 ep_rew_mean(평균 점수)이 찍힘
env = Monitor(gym.make("CartPole-v1"))

# ---------------------------------------------------------------
# 2) PPO 모델 만들기
# ---------------------------------------------------------------
# "MlpPolicy" : 관찰이 숫자 4개짜리 벡터라서 MLP(완전연결 신경망)를 씀
#               입력 = obs(4개) → 출력 = 각 행동(왼쪽/오른쪽)을 고를 확률
#               나중에 포켓몬에서 화면 픽셀을 쓰면 "CnnPolicy"로 바뀜
# verbose=1   : 학습 중 로그 표를 터미널에 출력
# tensorboard_log : 학습 곡선 저장 위치 (runs/는 .gitignore에 들어 있음)
# seed        : 랜덤 시드 고정. 이유: 설정을 바꿔가며 비교할 때
#               "설정 때문인지 운 때문인지" 구분하려고
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="runs/cartpole",
    seed=0,
)

# ---------------------------------------------------------------
# 3) 학습
# ---------------------------------------------------------------
# 이 한 줄 안에서 01_random_loop.py의 루프가 그대로 돌아감
# 차이는 딱 두 가지:
#   - 행동을 sample()이 아니라 신경망이 obs를 보고 고름
#   - 일정 스텝(기본 2048)을 모을 때마다 "점수가 높았던 행동이
#     더 자주 나오도록" 신경망을 업데이트함
# total_timesteps = 에피소드 수가 아니라 step() 호출 횟수 총합
model.learn(total_timesteps=50_000)

# ---------------------------------------------------------------
# 4) 평가
# ---------------------------------------------------------------
# 학습 중 점수 말고 따로 평가하는 이유:
#   학습 중에는 일부러 확률적으로 이것저것 시도(탐험)해서 점수가 낮게 나옴
#   deterministic=True → 가장 확률 높은 행동만 골라서 "실력"만 측정
mean_reward, std_reward = evaluate_policy(
    model, env, n_eval_episodes=20, deterministic=True
)
print(f"\n[평가] 20판 평균 점수: {mean_reward:.1f} ± {std_reward:.1f}  (랜덤은 20점 안팎, 만점 500)")

# ---------------------------------------------------------------
# 5) 저장
# ---------------------------------------------------------------
# models/cartpole_ppo.zip 으로 저장됨 (models/도 .gitignore에 있음)
# 저장해두면 나중에 PPO.load()로 불러와서 다시 학습 없이 쓸 수 있음
model.save("models/cartpole_ppo")
env.close()

# ---------------------------------------------------------------
# 6) 학습된 에이전트 눈으로 보기
# ---------------------------------------------------------------
# 이번엔 창을 띄움. 루프 모양은 01_random_loop.py와 완전히 같고
# 행동 고르는 한 줄만 다름
env = gym.make("CartPole-v1", render_mode="human")

for episode in range(3):
    obs, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        # 랜덤 뽑기 대신 모델에게 물어봄
        # predict()는 (행동, 내부상태) 2개를 돌려주는데 내부상태는 안 써서 _로 버림
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"episode {episode}: total_reward = {total_reward}")

env.close()
