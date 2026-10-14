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


# 판 기록 (10-12, E26 1단계): 지나간 구간 순서, 도착한 목표, 멈춘 이유 → main이 결과에 붙임
REC = {}


def _state_key(pb):
    """의미 키 (train/e26_eval_states.py와 같은 정의): 맵·좌표·방향·이벤트 플래그 해시"""
    import hashlib
    from train.e26_eval_states import EVENT_FLAGS, EVENT_LEN, FACING
    ev = bytes(pb.memory[1, EVENT_FLAGS + i] for i in range(EVENT_LEN))
    return {"position": list(mm.read_position(pb)), "facing": pb.memory[1, FACING], "event_sha1": hashlib.sha1(ev).hexdigest()[:12]}


def _harvest(pb, args, seg, g, ep, k):
    """E26 트랙 B: 정책이 새 구간을 맡는 순간의 상태 저장 (경계 출발점 후보)"""
    import io
    d = Path(args.harvest_dir) / seg
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"s{args.seed_offset + ep}_{k:02d}.state"
    buf = io.BytesIO(); pb.save_state(buf); f.write_bytes(buf.getvalue())
    with open(Path(args.harvest_dir) / "manifest.jsonl", "a", encoding="utf-8") as m:
        m.write(json.dumps({"segment": seg, "goal": list(g), "path": str(f), "seed": args.seed_offset + ep, **_state_key(pb)},
                           ensure_ascii=False) + "\n")


def _detected(det, pb, args, ep, log):
    """감지됨: 기록. stop이면 경계 상태 + 감지 순간 상태 저장하고 True (판과 실행을 멈춤)"""
    rep = {**det.report(), "seed": args.seed_offset + ep, "at": _state_key(pb)}
    REC["detections"].append(rep)
    log(f"  [감지] {rep['reason']} (목표 {rep['goal']}, 시작 {rep['start_map']} {rep['start_pos']}, 정책 걸음 {rep['policy_steps']}, 재방문 {rep['revisits']})")
    if args.detect != "stop":
        return False
    import io
    out = Path(args.detect_out or "runs/e26/detect")
    out.mkdir(parents=True, exist_ok=True)
    (out / "boundary.state").write_bytes(det.boundary)
    buf = io.BytesIO(); pb.save_state(buf); (out / "detected.state").write_bytes(buf.getvalue())
    (out / "detection.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    REC["stop_reason"] = f"detect:{rep['reason']}"
    global stop
    stop = True       # 실행 전체 멈춤 (트랙 A가 여기서 탐색·학습으로 넘어감)
    return True


def run_episode(env, model, parts, args, ep, log):
    obs, info = env.reset()
    from envs import clock
    REC.clear()
    REC.update({"segments": [], "targets_reached": [], "stop_reason": None, "detections": [], **clock.info(env.pyboy)})
    from train.detector import FailureDetector
    det = FailureDetector() if args.detect != "off" else None
    n_harvest = 0
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
        if det is not None and det.on_map(here) and _detected(det, pb, args, ep, log):
            break
        if args.snap_lab_exit and evt_lab and here == (24, 4):
            # 연구소 이벤트 뒤 연두마을에 처음 나온 순간 (E-lab). 저장하고 끝냄
            import io
            buf = io.BytesIO(); pb.save_state(buf)
            Path(args.snap_lab_exit).parent.mkdir(parents=True, exist_ok=True)
            Path(args.snap_lab_exit).write_bytes(buf.getvalue())
            log(f"  [E-lab 저장] {mm.read_position(pb)} → {args.snap_lab_exit}")
            REC["stop_reason"] = "snapshot:lab_exit"
            return {"ok": False, "snapshot": args.snap_lab_exit, "steps": env.step_count, "heals": heals, "events": events,
                    "whiteouts": whiteouts, "scripted": scripted, "battles": env.battles, "late_battles": late_battles,
                    "lock_battles": lock_battles, "hp": mm.read_party_hp_total(pb), "seconds": round(time.time() - t0)}

        # --- 이벤트/장소 규칙 (비계) ---
        if here == MR_POKEMON and target == MR_POKEMON and not evt_mr:
            # 할아버지 집 도착 → 이벤트 ① (들어서는 순간 잠김)
            r = run_event(pb)
            events += 1
            log(f"  할아버지 집 이벤트: A {r['presses']}번, 끝={r['done']}")
            ti, got_pokemon_event, evt_mr = ti + 1, True, True
            REC["targets_reached"].append(list(MR_POKEMON))
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
                REC["stop_reason"] = "rule:lab_exit_cap"
                if det is not None and args.detect_rule_cap:
                    # E26 (10-14, 확인 필요 항목): 규칙이 막혀 판이 끊기는 것도 "지금 목표 실패"로 감지 (원인은 정책이 연구소에 다시 들어간 것)
                    #   이게 없으면 재방문 2회 전에 규칙 상한이 먼저 판을 끝내서 감지가 안 됨 (E-lab 시험 3/3)
                    det.fired = "rule_cap"
                    _detected(det, pb, args, ep, log)
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
            REC["targets_reached"].append(list(LAB))
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
            REC["targets_reached"].append(list(here))
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
            REC["stop_reason"] = f"transition:no_plan {here}->{target}"
            break
        if not REC["segments"] or REC["segments"][-1] != seg:
            REC["segments"].append(seg)
            if args.harvest_dir and seg:
                n_harvest += 1
                _harvest(pb, args, seg, g, ep, n_harvest)
        if det is not None and getattr(det, "leg", None) != tuple(target):
            det.new_leg(target, here)
        if det is not None and (det.goal is None or tuple(g) != det.goal):
            import io as _io
            _b = _io.BytesIO(); pb.save_state(_b)
            det.new_goal(g, here, mm.read_position(pb), _b.getvalue())
        if g != last_goal:
            log(f"  [{env.step_count}걸음] {map_name(here)}: 목표 {map_name(g)} (구간 {seg}, 최종 {map_name(target)})")
            last_goal = g
        env.goal_vec = goal_vec(g)
        action, _ = model.predict(env._get_obs(), deterministic=args.deterministic)
        before = mm.read_position(pb)
        w0 = info.get("whiteouts", 0)
        b0, e0 = env.battles, getattr(env, "events", 0)
        obs, _, terminated, truncated, info = env.step(action)
        # 감지기 걸음: 이 반복에서 전투도 잠김 처리(run_event)도 없었을 때만 아래(반복 끝)에서 셈 (설계 보강 1)
        #   10-14 시험: 잠김(대화) 걸음을 세서 집 2층에서 'stall' 오탐 → 위치를 반복 끝으로 옮김
        plain_step = env.battles == b0 and getattr(env, "events", 0) == e0
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
            continue
        if det is not None and plain_step and det.step(mm.read_position(pb)) and _detected(det, pb, args, ep, log):
            break
    return {"ok": False, "steps": env.step_count, "heals": heals, "events": events, "whiteouts": whiteouts,
            "scripted": scripted, "battles": env.battles, "late_battles": late_battles, "lock_battles": lock_battles, "hp": mm.read_party_hp_total(pb),
            "seconds": round(time.time() - t0), "where": map_name(mm.read_position(pb)[:2]),
            "stage": "할아버지 집 전" if ti == 0 else "연구소 전"}


def _finalize(r, args, env):
    """판 결과에 기록 붙이기: 지나간 구간, 도착한 목표, 종료 위치, 실패 종류, 유효성
    유효성 (10-12): 성공이라면 의도한 목표를 순서대로 전부 도착해야 함. 아니면 무효 (예: 연구소에서 끝난 '성공')
    실패 종류: rule (규칙·매크로가 멈춤) / transition (상위 계획·구간 전환 오류) / policy (정책이 걸음 상한까지 목표 못 감)"""
    from envs import clock
    REC.update(clock.info(env.pyboy))    # 판 끝 게임 시각 (불러온 직후 값은 옛 시각이라 끝에서 읽음)
    reached = REC["targets_reached"]
    intended = [list(t) for t in args.target_list]
    if r["ok"]:
        valid = reached == intended
        kind = "ok" if valid else "invalid_scope"
    else:
        sr = REC["stop_reason"]
        kind = "rule" if sr and sr.startswith("rule") else "transition" if sr and sr.startswith("transition") else \
            "detected" if sr and sr.startswith("detect") else "interrupted" if stop else "policy"
        valid = kind != "interrupted" and reached == intended[:len(reached)]
    return {"segments": list(REC["segments"]), "targets_reached": reached, "end_pos": list(mm.read_position(env.pyboy)),
            "stop_reason": REC["stop_reason"], "fail_kind": kind, "valid": valid, "detections": REC.get("detections", []),
            "clock": REC["clock"], "game_time": REC["game_time"], "time_of_day": REC["time_of_day"]}


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
    parser.add_argument("--seed-offset", type=int, default=0, help="판 시드 = 오프셋 + 판 번호 (E26: 평가 상태 만들기와 학습용 실행의 시드를 분리)")
    parser.add_argument("--snap-lab-exit", default=None,
                        help="E26 (10-14): 연구소 이벤트를 마치고 연두마을로 처음 나온 순간 상태를 이 경로로 저장하고 판을 끝냄 (평가 상태 E-lab 만들기)")
    parser.add_argument("--harvest-dir", default=None,
                        help="E26 트랙 B (10-14): 구간이 바뀌는 순간(정책이 새 구간을 맡는 순간)의 상태를 <폴더>/<구간>/에 저장 + manifest.jsonl")
    parser.add_argument("--detect", choices=["off", "observe", "stop"], default="off",
                        help="E26 트랙 A: 실패 감지기 (train/detector.py). observe = 기록만, stop = 감지하면 경계 상태 저장하고 실행 전체를 멈춤")
    parser.add_argument("--detect-rule-cap", action="store_true",
                        help="규칙 상한(연구소 나가기 60번)으로 판이 끊기는 것도 감지로 셈 (E26 10-14 추가 기준, 사용자 확인 전)")
    parser.add_argument("--detect-out", default=None, help="--detect stop일 때 경계 상태·감지 기록을 둘 폴더")
    parser.add_argument("--no-heal-at", default="", help="이 맵들에선 회복하러 가지 않음 '그룹,번호;...' "
                        "(10-12: 도라지시티 — violet_pc_exit 구간이 문 앞 시작이라 회복 뒤 간호사 앞에서 못 나옴. 비계 변경, 결과에 명시)")
    args = parser.parse_args()
    args.target_list = [tuple(int(v) for v in t.split(",")) for t in args.targets.split(";")]
    args.no_heal = {tuple(int(v) for v in t.split(",")) for t in args.no_heal_at.split(";") if t}
    signal.signal(signal.SIGINT, _on_ctrl_c)

    cfg = load_env_config(Path(args.model).parent)
    assert cfg.get("obs_goal"), "목표 정보(--goal)로 학습한 모델이어야 함"
    model = _load(args.model)
    parts = [n for n in SEGMENTS if SEGMENTS[n].goal_maps and n != "newbark_cherrygrove" and not SEGMENTS[n].eval_only]
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
            torch.manual_seed(args.seed_offset + ep)
            np.random.seed(args.seed_offset + ep)
            log = (lambda s: None) if args.quiet else print
            print(f"\n[{ep + 1}] 집 1층 출발 → 할아버지 집 → 연구소")
            r = run_episode(env, model, parts, args, ep, log)
            r.update(_finalize(r, args, env))
            results.append(r)
            print(f"[{ep + 1}] {'성공' if r['ok'] else '실패'}: {r}")
            if stop:
                break
    finally:
        env.close()
    ok = sum(r["ok"] for r in results)
    # 10-12 실행 기록: 코드 버전, 시계, 시작 상태/모델 해시, 의도한 범위 → <stem>_meta.json (결과 파일과 같은 이름)
    print(f"\n성공 {ok}/{len(results)}, 걸음 중앙값 {np.median([r['steps'] for r in results if r['ok']]) if ok else '-'}")
    run_id = time.strftime("%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
    out = Path("runs/story") / f"m2a_{run_id}_{Path(args.model).parent.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    from train.evalrec import sha1_file, write_meta
    write_meta(out.with_suffix(""), {
        "kind": "run_story", "model": args.model, "model_sha1": sha1_file(args.model),
        "start": args.start, "start_sha1": sha1_file(args.start),
        "targets": [list(t) for t in args.target_list], "no_heal_at": sorted(list(t) for t in args.no_heal),
        "max_steps": args.max_steps, "episodes": len(results), "deterministic": args.deterministic,
        "clock": results[0]["clock"] if results else None,
        "game_times": sorted({r["game_time"] for r in results}),
        "ok": ok, "valid": sum(r["valid"] for r in results),
        "fail_kinds": {k: sum(r["fail_kind"] == k for r in results) for k in {r["fail_kind"] for r in results}},
    })
    inv = [i + 1 for i, r in enumerate(results) if not r["valid"]]
    if inv:
        print(f"⚠️ 무효 판 {len(inv)}개 (의도한 범위와 다름): {inv}")
    print("저장:", out)


if __name__ == "__main__":
    main()
