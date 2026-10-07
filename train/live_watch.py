"""
라이브 뷰어: 켜두면 "가장 최신 체크포인트"로 계속 한 판씩 플레이를 보여줌

왜 필요한가
- watch.py는 모델 하나로 한 판 보고 끝남 → 학습이 진행될 때마다 파일명을 바꿔서 다시 실행해야 함
- 이 스크립트는 판이 끝날 때마다 models/<실험>/ 폴더에서 가장 최근 .zip을 찾아서
  새 파일이 생겼으면 그걸로 바꿔 끼움 → 학습 옆에 틀어두면 실력이 바뀌는 게 그대로 보임
- 판이 끝날 때마다 터미널에 한 줄 요약(어느 맵까지, 전투 몇 번)을 남김 → 나중에 훑어보기 좋음

실행 (학습과 다른 터미널에서)
- uv run python -m train.live_watch v1_explore
- 빠르게 보기:     ... --speed 6
- 한 판 짧게:      ... --steps 1024
- 가장 확률 높은 행동만: ... --deterministic
- 29번도로 커리큘럼 모델: uv run python -m train.live_watch A_route29 --env route29
- 역방향 시연 학습:   uv run python -m train.live_watch DEMO_route29 --env route29_demo
  (지금 연습 중인 전선 위치에서 시작. --real-every판마다 한 번은 실제 입구에서)
- 끄기: 터미널에서 Ctrl+C 한 번, 또는 게임 창 닫기
창 하나가 CPU 코어 하나 정도를 써서, 학습이 약간(10% 안팎) 느려질 수 있음
"""

import argparse
import json
import signal
from pathlib import Path

from stable_baselines3 import PPO

from envs import memory_map as mm
from envs.curriculum import DemoReverseEnv
from envs.gold_env import GoldEnv
from envs.map_names import map_name
from envs.segments import SEGMENTS
from train.train_ppo import CURRICULUM_PRESETS, ENV_PRESETS, load_env_config


# ---------------------------------------------------------------
# 종료 처리
# ---------------------------------------------------------------
# Ctrl+C를 그냥 두면 KeyboardInterrupt 예외가 PyBoy 내부(Cython 코드) 안에서 터지는 경우가 많음
# → PyBoy가 "Exception ignored"로 삼켜버려서 에러만 쏟아지고 안 꺼짐
# 그래서 Ctrl+C를 예외 대신 "그만하라는 깃발"로 바꾸고, 스텝 사이(파이썬 코드)에서 깃발을 확인해서 빠져나감
stop_requested = False


def _on_ctrl_c(signum, frame):
    global stop_requested
    stop_requested = True


def should_stop(env):
    # Ctrl+C를 눌렀거나, 게임 창이 닫혔으면 그만
    return stop_requested or not getattr(env, "window_open", True)


def newest_model(model_dir):
    # 수정 시각 기준 가장 최근 파일 = 학습이 가장 최근에 저장한 체크포인트
    zips = list(model_dir.glob("*.zip"))
    return max(zips, key=lambda p: p.stat().st_mtime) if zips else None


def play_episode(env, model, deterministic=False):
    """한 판 플레이하고 요약을 돌려줌"""
    obs, info = env.reset()
    maps_in_order = []    # 처음 들어간 순서대로
    battle_lengths = []   # 전투마다 걸린 스텝 수
    in_battle = False
    exp_start = mm.read_mon1_exp(env.pyboy)
    done = False

    while not done:
        if should_stop(env):
            return None
        # 학습 때와 같은 확률적 행동
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        here = info["position"][:2]
        if here not in maps_in_order:
            maps_in_order.append(here)

        # 전투 플래그가 꺼짐 → 켜짐으로 바뀌는 순간 = 전투 시작
        # 전투 길이를 같이 기록하는 이유: 무작위로 누르면 전투 하나에 수백 스텝을 씀
        # → 전투가 짧아지면 "싸우다"를 고르는 법을 배우고 있다는 신호
        if mm.in_battle(env.pyboy):
            if not in_battle:
                battle_lengths.append(0)
            battle_lengths[-1] += 1
            in_battle = True
        else:
            in_battle = False

    return {
        "tiles": info["visited_tiles"],
        "maps": [map_name(m) for m in maps_in_order],
        "battle_lengths": battle_lengths,
        # 자동 전투(auto_battle) 환경에선 전투가 한 걸음 안에서 끝나서 위에서 못 셈 → 환경이 센 값을 씀
        "battles": max(len(battle_lengths), info.get("battles", 0)),
        "auto_battle": env.auto_battle,
        # 경험치가 늘었으면 전투에서 이긴 것 (도망치거나 지면 안 늘어남)
        # 레벨은 한 번 이겨도 안 오를 수 있어서 경험치로 봄 (꼬렛 Lv4 한 마리 = +32, Lv6까지 44 필요)
        "exp_gain": mm.read_mon1_exp(env.pyboy) - exp_start,
        "badges": info["badges"],
        # 전투 중 파티 HP가 0이 됐으면 전멸 → 게임이 마지막 회복 장소(지금은 집 2층)로 보냄
        "whiteouts": info.get("whiteouts", 0),
        "reached": info.get("reached_goal", False),
        "steps": env.step_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="실험 이름 (models/<name>/)")
    parser.add_argument("--speed", type=int, default=3, help="배속 (0이면 제한 없음)")
    parser.add_argument("--steps", type=int, default=None, help="한 판 길이 (기본: 환경 프리셋 값)")
    # 학습한 환경과 같은 조건으로 봐야 함 (route29 모델을 집에서 시작시키면 엉뚱한 행동이 나옴)
    parser.add_argument("--env", default="home", choices=[*ENV_PRESETS, *CURRICULUM_PRESETS])
    # 역방향 시연(demo) 커리큘럼일 때: 몇 판마다 한 번 실제 입구에서 시작할지 (나머지는 지금 전선에서)
    parser.add_argument("--real-every", type=int, default=4)
    # --wait: 체크포인트가 아직 없으면 꺼지지 않고 생길 때까지 기다림 (학습 스크립트가 자동 실행할 때 씀)
    parser.add_argument("--wait", action="store_true")
    # 확률이 가장 높은 행동만 고름. 헤매는 게 확률 뽑기 노이즈인지, 정책이 그 길을 좋다고 믿는 건지 구분할 때
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _on_ctrl_c)

    model_dir = Path("models") / args.name
    if args.wait:
        import time
        print(f"라이브: {model_dir}에 첫 체크포인트가 생길 때까지 기다리는 중...")
        while newest_model(model_dir) is None and not stop_requested:
            time.sleep(15)
        if stop_requested:
            return
    # 커리큘럼 이름이면 바탕 구간 설정을 씀. 역방향 시연이면 전선 위치에서 시작하는 환경으로
    #   왜: 시연 학습은 목표 근처부터 배워서, 초반엔 실제 입구에서 보면 헤매는 모습만 보임
    #       → "지금 연습 중인 곳"에서 시작해야 실력이 늘어나는 게 보임
    demo = None
    parts = None
    if args.env in CURRICULUM_PRESETS and CURRICULUM_PRESETS[args.env].get("_class") == "multi_lat":
        # 여러 구간 섞어 학습(GOAL1): 판마다 구간을 번갈아 보여줌. 판 시작 때 그 구간 환경을 새로 만듦
        parts = CURRICULUM_PRESETS[args.env]["parts"]
        env_kwargs = dict(ENV_PRESETS[parts[0]])
        demo = {"demo_path": SEGMENTS[parts[0]].demo_path, "real_start_prob": 0.0, "window": 0}
    elif args.env in CURRICULUM_PRESETS:
        preset = CURRICULUM_PRESETS[args.env]
        env_kwargs = dict(ENV_PRESETS[preset["base"]])
        if preset.get("_class") in ("demo", "demo_lat"):   # 경로 옆 시작도 라이브는 시연 전선에서
            demo = {"demo_path": preset["demo_path"], "real_start_prob": 0.0, "window": 0}
    else:
        env_kwargs = dict(ENV_PRESETS[args.env])
    if args.steps:
        env_kwargs["max_steps"] = args.steps
    # 조작 방식은 학습 때와 같게 (models/<실험>/env_config.json, 없으면 tap)
    cfg = load_env_config(model_dir)
    env_kwargs["move_mode"] = cfg.get("move_mode", "tap")
    env_kwargs["obs_memory"] = cfg.get("obs_memory", False)   # 짧은 기억 모델이면 관찰 모양도 맞춰야 함
    env_kwargs["obs_goal"] = cfg.get("obs_goal", False)       # 목표 정보 모델이면 관찰에 목표도
    if demo:
        env = DemoReverseEnv(render_mode="human", **demo, **env_kwargs)
        ptr_file = Path("runs/curriculum") / args.name / "demo_ptr.json"   # 학습 콜백이 전선을 당길 때마다 씀
    else:
        env = GoldEnv(render_mode="human", **env_kwargs)
    env.pyboy.set_emulation_speed(args.speed)

    loaded, model = None, None
    episode = 0
    try:
        while not should_stop(env):
            # 판 사이에만 모델을 바꿈 (판 도중에 바꾸면 어느 모델의 행동인지 섞여서 해석이 안 됨)
            path = newest_model(model_dir)
            if path is None:
                print(f"{model_dir}에 아직 체크포인트가 없음. 첫 저장(라이브용 약 1만 스텝)까지 기다려야 함")
                return
            # live.zip은 같은 이름으로 덮어써지니까 이름이 아니라 (이름, 수정 시각)으로 새 파일인지 봄
            key = (path, path.stat().st_mtime)
            if key != loaded:
                model = PPO.load(path, device="cpu")
                loaded = key
                print(f"--- 모델 교체: {path.name} ({model.num_timesteps:,}스텝)")
            loaded_path = path

            start_note = ""
            if parts:
                part = parts[episode % len(parts)]
                if getattr(env, "_part", None) != part:
                    env.close()
                    kw = {**env_kwargs, **ENV_PRESETS[part], "move_mode": env_kwargs["move_mode"],
                          "obs_memory": env_kwargs["obs_memory"], "obs_goal": env_kwargs["obs_goal"]}
                    env = DemoReverseEnv(render_mode="human", demo_path=SEGMENTS[part].demo_path,
                                         real_start_prob=0.0, window=0, **kw)
                    env.pyboy.set_emulation_speed(args.speed)
                    env._part = part
            if demo:
                real = episode % args.real_every == args.real_every - 1
                env.real_start_prob = 1.0 if real else 0.0
                ptr = env.demo_len - 1
                if ptr_file.exists():
                    try:
                        saved = json.loads(ptr_file.read_text())
                        ptr = saved[env._part]["ptr"] if parts else saved["ptr"]
                    except (ValueError, KeyError):
                        pass  # 학습 쪽이 쓰는 도중이면 이번 판은 이전 값
                env.set_ptr(ptr)
                start_note = "실제 입구에서 시작" if real else f"전선에서 시작 (목표 {env.demo_len - env.ptr}걸음 앞)"
                if parts:
                    start_note = f"[{env._part}] " + start_note

            s = play_episode(env, model, args.deterministic)
            if s is None:  # 판 도중에 종료 요청
                break
            episode += 1
            if start_note:
                print(f"[{episode}] {start_note} → {'도착 O' if s['reached'] else '도착 X'} ({s['steps']}걸음)")
            print(
                f"[{episode}] {loaded_path.stem} | 칸 {s['tiles']} | 맵 {len(s['maps'])}개 "
                f"(가장 먼 곳: {s['maps'][-1]}) | 전투 {s['battles']}번"
                f"{' (자동)' if s['auto_battle'] else ' ' + str(s['battle_lengths']) + '스텝'} | "
                f"경험치 +{s['exp_gain']} | 배지 {s['badges']}"
            )
            route = " > ".join(s["maps"])
            if s["whiteouts"]:
                route += f"   ※ 전멸 {s['whiteouts']}번 (마지막 맵은 전멸 후 이동한 곳)"
            print(f"     경로: {route}")
    except KeyboardInterrupt:
        print("\n종료")
    finally:
        env.close()


if __name__ == "__main__":
    main()
