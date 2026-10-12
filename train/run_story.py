"""
통합 실행 (M2a 한 판): 집 1층 → 포켓몬할아버지 집 [이벤트] → 연구소 [이벤트], 정책 하나 + 상위 계획 + 회복 + 이벤트 매크로

왜 필요한가
- 구간별 평가(diagnose)는 "구간 시작 상태에서 그 구간 목표까지"만 잼. 실제 플레이는 구간이 이어지고,
  HP가 쌓여서 줄고, 이벤트가 끼고, 전멸하면 집 2층에서 다시 시작함 → 한 판으로 이어서 되는지 봐야 함
- M2A_r30(정책 하나, 목표 11개)이 구간별로는 다 됨 → 이제 사람이 구간을 바꿔주지 않고 한 판

역할 나누기 (무엇이 RL이고 무엇이 비계인지)
- RL: 걷기 (목표 맵을 받아서 그 맵까지). M2A_r30 정책
- 비계 (규칙/매크로, 출력에 표시):
  1. 상위 계획: 맵 연결 그래프(envs/planner.py, 구간 edges에서 자동)로 "지금 맵 → 최종 목표" 맵 순서.
     정책에 줄 목표 = 지금 맵을 포함하는 학습 구간 중 그 구간 목표가 계획에서 가장 가까운 것
     (정책은 "바로 옆 맵"이 아니라 학습한 구간의 목표로 배웠음. 예: 집 1층에선 목표 = 29번도로)
  2. 회복: 새 마을(센터 있음) 첫 도착이면 센터, 그 밖엔 HP 절반 미만이거나 공격 기술 PP 0이면 (planner.should_heal). 센터 안에선 heal_at_pc
  3. 이벤트: 방향키를 눌렀는데 안 움직이고 움직일 수도 없으면(잠김) → run_event (A + 움직임 확인, 이름 화면이면 결정)
  4. 할아버지 집 안 → 밖: 학습한 구간이 없음 → 맵이 바뀔 때까지 아래 (규칙)
  5. 연구소 안: 위로 걸어서 이벤트를 일으킴 (규칙) → run_event
- 전투: 학습과 같이 A 연타 자동. 전멸하면 집 2층(또는 마지막 센터)에서 계획을 다시 계산해서 이어감 (판 안 끝냄)

실행
- uv run python -m train.run_story models/M2A_r30/final.zip --headless --episodes 10
- 보기: uv run python -m train.run_story models/M2A_r30/final.zip --speed 4
"""

import argparse
import json
import signal
import os
import time
from pathlib import Path

import numpy as np
import torch

from envs import memory_map as mm
from envs.gold_env import ACTIONS, GOAL_GROUPS, GOAL_NUMBERS, GoldEnv
from envs.heal import _press, can_move, heal_at_pc, party_max_hp
import envs.heal as envs_heal
from envs.map_names import map_name
from envs.menu import run_event
from envs.planner import map_graph, pc_maps, plan, should_heal
from envs.segments import SEGMENTS
from train.diagnose import _load
from train.train_ppo import load_env_config

MR_POKEMON = (26, 10)
LAB = (24, 5)
stop = False


def _on_ctrl_c(signum, frame):
    global stop
    stop = True


def goal_vec(m):
    v = np.zeros(GOAL_GROUPS + GOAL_NUMBERS, dtype=np.float32)
    v[min(m[0], GOAL_GROUPS - 1)] = 1.0
    v[GOAL_GROUPS + min(m[1], GOAL_NUMBERS - 1)] = 1.0
    return v


def policy_goal(cur, target, parts):
    """지금 맵을 포함하는 학습 구간 중, 구간 목표가 계획(cur → target)에서 가장 앞에 있는 것의 목표 맵"""
    p = plan(cur, target)
    if not p:
        return None, None
    best = None
    for name in parts:
        seg = SEGMENTS[name]
        g = tuple(seg.goal_maps[0])
        if tuple(cur) in {tuple(m) for m in seg.allowed_maps} and g in p:
            if best is None or p.index(g) < p.index(best[1]):
                best = (name, g)
    return best if best else (None, p[0])


def _save(pb, args):
    """--save-state: 성공한 판의 마지막 상태를 저장 (다음 구간 시작 상태 만들기용)"""
    if args.save_state:
        import io
        buf = io.BytesIO()
        pb.save_state(buf)
        Path(args.save_state).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_state).write_bytes(buf.getvalue())


def run_episode(env, model, parts, args, ep, log):
    obs, info = env.reset()
    pb = env.pyboy
    targets = args.target_list           # 스토리 순서 (사람이 정함: 진행 순서는 손으로 적는 몇 안 되는 것)
    ti = 0
    evt_mr, evt_lab = False, False       # 이벤트를 이번 판에 처리했나
    visited_towns, heals, events, whiteouts, scripted, late_battles = set(), 0, 0, 0, 0, 0
    lock_battles = 0
    lab_rule = 0                         # 연구소 나가기 규칙 연속 횟수 (무한 반복 방지)
    heal_target = None                   # 회복하러 가는 중이면 그 센터 (회복 끝날 때까지 유지)
    got_pokemon_event = False
    last_goal = None
    t0 = time.time()
    while env.step_count < args.max_steps and not stop:
        here = tuple(mm.read_position(pb)[:2])
        target = targets[ti]
        if here != LAB:
            lab_rule = 0

        # --- 이벤트/장소 규칙 (비계) ---
        if here == MR_POKEMON and target == MR_POKEMON and not evt_mr:
            # 할아버지 집 도착 → 이벤트 ① (들어서는 순간 잠김)
            r = run_event(pb)
            events += 1
            log(f"  할아버지 집 이벤트: A {r['presses']}번, 끝={r['done']}")
            ti, got_pokemon_event, evt_mr = ti + 1, True, True
            continue
        if here == MR_POKEMON:
            # 집 안 → 밖: 학습 구간 없음 → 맵이 바뀔 때까지 아래 (규칙)
            _press(pb, "down")
            scripted += 1
            env.step_count += 1
            continue
        if here == LAB and target != LAB:
            # 연구소 안 → 밖 (학습 구간 없음): 맵이 바뀔 때까지 아래, 막히면 왼쪽 한 칸 (책상) (규칙)
            #   10-12: 다시 들어온 뒤 이 규칙이 6,000걸음까지 반복된 판이 있음 → 연속 60번이면 상태 저장 후 판 실패로
            lab_rule += 1
            if lab_rule > 60:
                from envs.menu import _dump_stuck
                _dump_stuck(pb, "story_lab_exit")
                log(f"  연구소 나가기 규칙 60번 반복 → 중단, 위치 {mm.read_position(pb)}")
                break
            before = mm.read_position(pb)
            _press(pb, "down")
            if mm.read_position(pb) == before:
                _press(pb, "left")
            if os.environ.get("STORY_DEBUG"):
                log(f"    연구소 규칙 {lab_rule}: {before} → {mm.read_position(pb)}")
            scripted += 1
            env.step_count += 1
            if not can_move(pb):
                r = run_event(pb)       # 나가는 길에 조수가 몬스터볼을 줌 (자동 이벤트)
                events += 1
                log(f"  연구소 나가는 길 이벤트: A {r['presses']}번")
            continue
        if here == LAB and target == LAB and not evt_lab:
            # 연구소 안: 위로 → 잠기면 이벤트 ④ (규칙 + 매크로)
            for _ in range(12):
                before = mm.read_position(pb)
                _press(pb, "up")
                scripted += 1
                if not can_move(pb):
                    break
            r = run_event(pb)
            events += 1
            evt_lab = True
            log(f"  연구소 이벤트: A·결정 {r['presses']}번 (이름 화면 {r['names']}번), 끝={r['done']}")
            # 이벤트 뒤 공박사에게 말 걸기 (알 건네기 → "도라지시티로"). 이걸 안 하면 30번도로 서쪽 길이 안 열림 (10-10 확인)
            #   규칙: 오른쪽 한 칸 → 위를 보고 → A (공박사가 오른쪽 위 대각선에 서 있음)
            _press(pb, "right")
            pb.button_press("up"); pb.tick(2, envs_heal.RENDER); pb.button_release("up"); pb.tick(20, envs_heal.RENDER)
            _press(pb, "a", wait=60)
            r2 = run_event(pb)
            scripted += 3
            log(f"  공박사 대화: A {r2['presses'] + 1}번")
            if ti + 1 < len(targets):
                # 연구소 뒤에도 목표가 남았으면 이어감 (10-12 버그: 여기서 항상 끝내서 M2B "도라지체육관까지" 20판이
                #   실제론 연구소까지만 잰 것이었음). 연구소 나가기는 위 "LAB and target != LAB" 규칙이 맡음
                ti += 1
                continue
            _save(pb, args)
            return {"ok": r["done"], "steps": env.step_count, "heals": heals, "events": events,
                    "whiteouts": whiteouts, "scripted": scripted, "battles": env.battles, "late_battles": late_battles, "lock_battles": lock_battles,
                    "hp": mm.read_party_hp_total(pb), "seconds": round(time.time() - t0)}
        if here in pc_maps() and heal_target == here:
            r = heal_at_pc(pb)
            heals += 1
            heal_target = None
            visited_towns |= {m for m in map_graph() if here in map_graph().get(m, ())}   # 이 센터와 이어진 마을
            log(f"  센터 회복: HP {r['hp_before']} → {r['hp_after']}/{r['max_hp']}")
            # 센터 안에서 나가기는 학습 구간(cherrygrove_pc_exit)이 있음 → 아래 정책으로

        # --- 회복 판단 (규칙) ---
        hp, mx = mm.read_party_hp_total(pb), party_max_hp(pb)
        if heal_target is None and here not in pc_maps() and here not in args.no_heal:
            from envs.battle import has_attack_pp
            need, why = should_heal(here, hp, mx, visited_towns, attack_pp=has_attack_pp(pb))
            near = [pc for pc in pc_maps() if pc in map_graph().get(here, ())]
            if need and near:          # 지금은 센터와 바로 이어진 마을에서만 (먼 센터로 돌아가기는 아직 안 함)
                heal_target = near[0]
                log(f"  회복하러 감: {why} → {map_name(heal_target)}")
        if heal_target is not None:
            target = heal_target
        elif here == target:
            # 이벤트 없는 목표 맵 도착 → 다음 목표 (마지막이면 성공)
            ti += 1
            if ti == len(targets):
                log(f"  [{env.step_count}걸음] 최종 목표 {map_name(here)} 도착")
                _save(pb, args)
                return {"ok": True, "steps": env.step_count, "heals": heals, "events": events, "whiteouts": whiteouts,
                        "scripted": scripted, "battles": env.battles, "late_battles": late_battles,
                        "lock_battles": lock_battles, "hp": mm.read_party_hp_total(pb), "seconds": round(time.time() - t0)}
            continue

        # --- 정책: 목표 맵 주고 한 걸음 ---
        seg, g = policy_goal(here, target, parts)
        if g is None:
            log(f"  계획 없음: {map_name(here)} → {map_name(target)}")
            break
        if g != last_goal:
            log(f"  [{env.step_count}걸음] {map_name(here)}: 목표 {map_name(g)} (구간 {seg}, 최종 {map_name(target)})")
            last_goal = g
        env.goal_vec = goal_vec(g)
        action, _ = model.predict(env._get_obs(), deterministic=args.deterministic)
        before = mm.read_position(pb)
        w0 = info.get("whiteouts", 0)
        obs, _, terminated, truncated, info = env.step(action)
        if info.get("whiteouts", 0) > w0 or (tuple(mm.read_position(pb)[:2]) == (24, 7) and here != (24, 7)):
            whiteouts += 1
            log(f"  전멸 → {map_name(mm.read_position(pb)[:2])}에서 이어감")
        # 늦게 시작한 전투 (걸음 처리 뒤에 인카운터): 학습과 같은 A 연타 자동으로
        if mm.in_battle(pb):
            env._run_auto_battle()
            late_battles += 1
            continue
        # 잠김 판별: 방향키인데 안 움직였고 움직일 수도 없음 → 이벤트
        if int(action) < 4 and mm.read_position(pb) == before and not can_move(pb):
            r = run_event(pb)
            events += 1
            kind = {0: "이벤트", 1: "야생 전투 (A 연타)", 2: "트레이너 전투 (A 연타)"}.get(r["battle"], "이벤트")
            if r["battle"]:
                lock_battles += 1
            log(f"  {kind} (잠김, {map_name(here)}): A {r['presses']}번, 끝={r['done']}, HP {mm.read_party_hp_total(pb)}")
    return {"ok": False, "steps": env.step_count, "heals": heals, "events": events, "whiteouts": whiteouts,
            "scripted": scripted, "battles": env.battles, "late_battles": late_battles, "lock_battles": lock_battles, "hp": mm.read_party_hp_total(pb),
            "seconds": round(time.time() - t0), "where": map_name(mm.read_position(pb)[:2]),
            "stage": "할아버지 집 전" if ti == 0 else "연구소 전"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--speed", type=int, default=3)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="판마다 요약만")
    parser.add_argument("--start", default="states/start.state", help="시작 상태 (기본: 집 1층)")
    parser.add_argument("--targets", default="26,10;24,5", help="목표 맵 순서 '그룹,번호;...' (기본: 할아버지 집 → 연구소)")
    parser.add_argument("--save-state", default=None, help="성공한 판의 마지막 상태 저장 경로 (다음 구간 시작용)")
    parser.add_argument("--no-heal-at", default="", help="이 맵들에선 회복하러 가지 않음 '그룹,번호;...' "
                        "(10-12: 도라지시티 — violet_pc_exit 구간이 문 앞 시작이라 회복 뒤 간호사 앞에서 못 나옴. 비계 변경, 결과에 명시)")
    args = parser.parse_args()
    args.target_list = [tuple(int(v) for v in t.split(",")) for t in args.targets.split(";")]
    args.no_heal = {tuple(int(v) for v in t.split(",")) for t in args.no_heal_at.split(";") if t}
    signal.signal(signal.SIGINT, _on_ctrl_c)

    cfg = load_env_config(Path(args.model).parent)
    assert cfg.get("obs_goal"), "목표 정보(--goal)로 학습한 모델이어야 함"
    model = _load(args.model)
    parts = [n for n in SEGMENTS if SEGMENTS[n].goal_maps and n != "newbark_cherrygrove"]
    all_maps = set(map_graph()) | {m for v in map_graph().values() for m in v}
    env = GoldEnv(render_mode=None if args.headless else "human", state_path=args.start,
                  max_steps=args.max_steps + 100, auto_battle=True, allowed_maps=list(all_maps), goal_maps=[MR_POKEMON],
                  move_mode=cfg.get("move_mode", "tile"), obs_memory=cfg.get("obs_memory", False), obs_goal=True,
                  fast_battle=cfg.get("fast_battle", False))
    env.goal_maps = set()   # 도착으로 판이 끝나지 않게 (판 종료는 이 스크립트가 정함)
    env.pyboy.set_emulation_speed(0 if args.headless else args.speed)
    import envs.heal
    envs.heal.RENDER = not args.headless   # 라이브면 이벤트/전투/회복 매크로도 화면에 (안 그리면 그동안 멈춘 듯 보이다 튐)

    results = []
    try:
        for ep in range(args.episodes):
            torch.manual_seed(ep)
            np.random.seed(ep)
            log = (lambda s: None) if args.quiet else print
            print(f"\n[{ep + 1}] 집 1층 출발 → 할아버지 집 → 연구소")
            r = run_episode(env, model, parts, args, ep, log)
            results.append(r)
            print(f"[{ep + 1}] {'성공' if r['ok'] else '실패'}: {r}")
            if stop:
                break
    finally:
        env.close()
    ok = sum(r["ok"] for r in results)
    print(f"\n성공 {ok}/{len(results)}, 걸음 중앙값 {np.median([r['steps'] for r in results if r['ok']]) if ok else '-'}")
    out = Path("runs/story") / f"m2a_{time.strftime('%m%d_%H%M')}_{Path(args.model).parent.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print("저장:", out)


if __name__ == "__main__":
    main()
