"""
일관성 시험 (10-23, 아키텍처 점검 1단계, 10-22 사용자 합의): 학습 / 단독 평가(diagnose) / 통합 실행(run_story)이
같은 상태·같은 행동에서 같은 관찰·같은 진행·같은 판정을 내는지 확인. **동작은 안 바꿈** — 세 경로의 실제 코드를 그대로 돌리고 기록만 함

왜 필요한가
- 10-19 보고서 판정 버그, 10-20 '지난 화면' 버그 (run_story), 10-21 E-lab 첫 화면 팔레트 — 셋 다 경로끼리 다르게 동작한 것
- 이게 남아 있으면 '정책이 잘못 배웠나 vs 실행기가 잘못 돌았나'를 못 가름 (E31 전에 확인)

어떻게
- PyBoy를 감싸는 대리 객체로 바꿔 끼움 (envs.gold_env.PyBoy): tick/save/load를 그대로 넘기면서 '실제 게임 진행 프레임'과
  '마지막으로 그린 프레임'을 셈 (can_move처럼 눌러보고 되돌리는 프레임은 세이브 상태 단위로 되돌림)
- 정책 자리에 기록용 Probe를 넣음 (diagnose·run_story의 _load를 바꿔 끼움). 예측할 때마다 기록:
  위치, 지난 예측 뒤 진행 프레임, 지난 화면 프레임 수(stale), 관찰 화면 해시, 목표 칸, 시간대 RAM
- 1번 경로(학습 환경 MultiSegmentEnv, 실제 시작)에서 M2B가 고른 행동을 저장 → 2·3번 경로에 같은 행동열을 그대로 넣음
  → 처음 갈라지는 예측 번호와 갈라진 항목을 보고
- diagnose의 시작 뒤 0~29프레임 흘리기는 '의도된 차이'라, 흘림이 0인 시드를 써서 맞춤 (흘림 자체는 따로 기록)

시험 항목 (10-22 표)
1. 관찰 생성: 같은 상태에서 관찰 해시·목표 칸이 같은가, 예측 때 지난 화면(stale > 0)이 있나
2. 행동 실행: 같은 행동열에서 위치 궤적·예측 사이 프레임 수가 같은가
3. 게임 시간: 시간대 RAM이 같은가 + 첫 관찰이 시간대 팔레트를 반영하나 (첫 화면 vs 몇 프레임 뒤 밝기)
4. 종료 판정: 도착·이탈·전멸·걸음 상한이 같은 걸음에서 나는가
5. 결과 기록: diagnose summary를 raw에서 다시 계산해 대조, run_story 기록의 성공 수 대조

실행: uv run python -m train.consistency_check [--segments route29,route30] [--clocks lock,day,night] [--seeds 2]
결과: runs/consistency/<날짜시각>.json + report.md
"""

import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import envs.gold_env as ge
from envs import memory_map as mm

MODEL = "models/M2B_multi/final.zip"
OUT = Path("runs/consistency")
TIME_OF_DAY = 0xD214

# ---------------------------------------------------------------
# PyBoy 대리 객체: 실제 게임 진행 프레임과 마지막으로 그린 프레임을 셈
# ---------------------------------------------------------------
_RealPyBoy = ge.PyBoy
CURRENT = {}


class PbProxy:
    def __init__(self, *a, **k):
        object.__setattr__(self, "_pb", _RealPyBoy(*a, **k))
        object.__setattr__(self, "clock", 0)          # 실제 진행 프레임 (되돌린 시험 프레임은 뺌)
        object.__setattr__(self, "last_render", None)  # 마지막으로 그린 프레임의 clock (None = 지금 상태의 화면을 아직 안 그림)
        object.__setattr__(self, "_saved", {})
        CURRENT["pb"] = self

    def tick(self, count=1, render=True, *a, **k):
        r = self._pb.tick(count, render, *a, **k)
        object.__setattr__(self, "clock", self.clock + count)
        if render:
            object.__setattr__(self, "last_render", self.clock)
        return r

    def save_state(self, f):
        self._pb.save_state(f)
        if isinstance(f, io.BytesIO):
            if len(self._saved) > 2000:
                self._saved.clear()
            self._saved[hashlib.sha1(f.getvalue()).digest()] = (self.clock, self.last_render)

    def load_state(self, f):
        data = f.read()
        self._pb.load_state(io.BytesIO(data))
        key = hashlib.sha1(data).digest()
        if key in self._saved:   # 눌러보고 되돌리기: 진행 프레임을 되돌림 (화면 버퍼는 안 바뀌므로 마지막 그림도 그대로)
            c, lr = self._saved[key]
            object.__setattr__(self, "clock", c)
            object.__setattr__(self, "last_render", lr)
        else:                    # 파일에서 새 상태: 화면 버퍼는 이전 상태 것
            object.__setattr__(self, "clock", 0)
            object.__setattr__(self, "last_render", None)

    def stale(self):
        return None if self.last_render is None else self.clock - self.last_render

    def __getattr__(self, name):
        return getattr(self._pb, name)

    def __setattr__(self, name, value):
        setattr(self._pb, name, value)


ge.PyBoy = PbProxy


# ---------------------------------------------------------------
# 기록용 정책
# ---------------------------------------------------------------
class Probe:
    """predict마다 기록. real이 있으면 진짜 모델로 행동을 고름 (1번 경로), 없으면 actions를 차례로 냄"""

    def __init__(self, real=None, actions=None):
        self.real, self.actions, self.trace = real, actions, []
        self.policy = real.policy if real is not None else None

    def predict(self, obs, deterministic=False, **kw):
        pb = CURRENT["pb"]
        screen = obs["screen"] if isinstance(obs, dict) else obs
        goal = obs.get("goal") if isinstance(obs, dict) else None
        prev = self.trace[-1]["clock"] if self.trace else None
        i = len(self.trace)
        if self.real is not None:
            a, _ = self.real.predict(obs, deterministic=False)
            a = int(a)
        else:
            a = self.actions[i] if i < len(self.actions) else 0
        self.trace.append({
            "i": i, "pos": list(mm.read_position(pb)), "clock": pb.clock, "dframes": None if prev is None else pb.clock - prev,
            "stale": pb.stale(), "obs": hashlib.sha1(np.ascontiguousarray(screen).tobytes()).hexdigest()[:10],
            "goal": [int(x) for x in np.flatnonzero(goal)] if goal is not None else None,
            "tod": int(pb.memory[1, TIME_OF_DAY]), "action": a, "beyond": self.real is None and i >= len(self.actions),
        })
        return np.int64(a), None


# ---------------------------------------------------------------
# 세 경로
# ---------------------------------------------------------------
def path_train(segment, seed, real):
    """학습 경로: train_ppo가 쓰는 MultiSegmentEnv (실제 시작만). SB3는 reset/step 관찰을 그대로 정책에 넣음"""
    from envs.curriculum import MultiSegmentEnv
    from envs.segments import SEGMENTS
    from train.train_ppo import load_env_config
    cfg = load_env_config(Path(MODEL).parent)
    env_config = {k: cfg[k] for k in ("move_mode", "obs_memory", "obs_goal", "fast_battle", "auto_event")}
    seg = SEGMENTS[segment]
    if Path(seg.demo_path).exists():
        env = MultiSegmentEnv(segments=[segment], real_start_prob=1.0, lateral_prob=0.0, **env_config)
        kind = "MultiSegmentEnv"
    else:   # 평가 전용 구간 (시연 없음): MultiSegmentEnv 실제 시작과 같은 GoldEnv (구간 설정 + 학습 env_config)
        env = ge.GoldEnv(**{**seg.env_kwargs(), **env_config})
        kind = "GoldEnv(평가 전용 구간, 시연 없음)"
    torch.manual_seed(seed)
    probe = Probe(real=real)
    obs, info = env.reset(seed=seed)
    end = None
    while True:
        a, _ = probe.predict(obs)
        obs, _, term, trunc, info = env.step(a)
        if term or trunc:
            end = {"step": env.step_count, "goal": bool(info["reached_goal"]), "left": bool(info["left_area"]),
                   "whiteout": info["whiteouts"] > 0, "truncated": bool(trunc), "battles": info["battles"], "events": info["events"]}
            break
    env.close()
    return probe, end, kind


def path_diagnose(segment, seed, actions):
    import train.diagnose as dg
    from envs.segments import SEGMENTS
    probe = Probe(actions=actions)
    dg._models.clear()
    orig = dg._load
    dg._load = lambda p: probe
    try:
        r = dg.run_one((segment, MODEL, "start", SEGMENTS[segment].start_state, seed))
    finally:
        dg._load = orig
    end = {"step": r["steps"], "goal": r["goal"], "left": r["left"], "whiteout": r["whiteout"], "timeout": r["timeout"],
           "battles": r["battles"]}
    return probe, end


def path_story(segment, actions):
    import train.run_story as rs
    from envs.segments import SEGMENTS
    seg = SEGMENTS[segment]
    probe = Probe(actions=actions)
    orig_load, orig_argv = rs._load, sys.argv
    rs._load = lambda p: probe
    g = seg.goal_maps[0]
    sys.argv = ["run_story", MODEL, "--headless", "--quiet", "--episodes", "1", "--max-steps", str(seg.max_steps),
                "--start", seg.start_state, "--targets", f"{g[0]},{g[1]}"]
    before = set(Path("runs/story").glob("m2a_*")) if Path("runs/story").exists() else set()
    rs.stop = False
    err = None
    try:
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            rs.main()
    except Exception as e:   # run_story 자체 오류도 발견 사항으로 기록하고 다음 사례로
        import traceback
        err = f"{type(e).__name__}: {e} @ {traceback.extract_tb(e.__traceback__)[-1].name}:{traceback.extract_tb(e.__traceback__)[-1].lineno}"
        CURRENT["pb"].stop(save=False)
    finally:
        rs._load, sys.argv = orig_load, orig_argv
    new = [p for p in Path("runs/story").glob("m2a_*") if p not in before]
    if err:
        for p in new:
            p.unlink()
        return probe, {"error": err, "step": len(probe.trace)}
    res = json.loads(next(p for p in new if p.suffix == ".json" and not p.name.endswith("_meta.json")).read_text(encoding="utf-8"))[0]
    for p in new:    # 시험 산출물은 남기지 않음
        p.unlink()
    end = {"step": res["steps"], "goal": res["ok"], "whiteout": res["whiteouts"] > 0, "stop_reason": res["stop_reason"],
           "fail_kind": res["fail_kind"], "battles": res["battles"], "late_battles": res["late_battles"],
           "events": res["events"], "scripted": res["scripted"], "heals": res["heals"], "stale_obs": res["stale_obs"]}
    return probe, end


# ---------------------------------------------------------------
# 비교
# ---------------------------------------------------------------
FIELDS = ("pos", "dframes", "obs", "goal", "tod")


def first_diff(a, b):
    """처음 갈라지는 예측 번호와 항목. 길이만 다르면 짧은 쪽 끝"""
    for x, y in zip(a, b):
        if y["beyond"]:
            return {"i": x["i"], "fields": ["행동열 끝 (앞 경로가 먼저 끝남)"]}
        diff = [f for f in FIELDS if x[f] != y[f]]
        if diff:
            return {"i": x["i"], "fields": diff, "a": {f: x[f] for f in diff}, "b": {f: y[f] for f in diff},
                    "pos": x["pos"], "action_before": a[x["i"] - 1]["action"] if x["i"] else None}
    if len(a) != len(b):
        return {"i": min(len(a), len(b)), "fields": ["길이"], "a": len(a), "b": len(b)}
    return None


def stale_summary(trace):
    s = [t for t in trace if t["stale"] != 0]
    return {"n": len(s), "first": [(t["i"], t["stale"], t["pos"]) for t in s[:5]]}


def first_frame_palette(segment):
    """게임 시간: reset 직후 첫 관찰 vs 1·5·30프레임 그린 뒤 화면 밝기 (시간대 팔레트가 첫 관찰에 들어갔나)"""
    from envs.segments import SEGMENTS
    env = ge.GoldEnv(**{**SEGMENTS[segment].env_kwargs(), "obs_goal": True, "move_mode": "tile"})
    obs, _ = env.reset()
    m0 = float(obs["screen"].mean())
    out = {"first_mean": round(m0, 1), "tod_ram": int(env.pyboy.memory[1, TIME_OF_DAY])}
    done = 0
    for k in (1, 5, 30):
        env.pyboy.tick(k - done, True)
        done = k
        out[f"after_{k}"] = round(float(env._screen().mean()) - m0, 1)
    env.close()
    return out


def check_records():
    """결과 기록: diagnose summary를 raw에서 다시 계산 (도착률·이탈·전멸·멈춤) + run_story meta의 성공 수"""
    bad, n, old = [], 0, []
    for s in sorted(Path("runs/diagnose").glob("*_summary.json")):
        raw = s.with_name(s.name.replace("_summary.json", "_raw.json"))
        if not raw.exists():
            continue
        rows, res = json.loads(s.read_text(encoding="utf-8")), json.loads(raw.read_text(encoding="utf-8"))
        if not res or not all(k in res[0] for k in ("goal", "left", "whiteout")) or not all("goal_rate" in r for r in rows):
            old.append(s.name)   # 옛 형식 (다른 키 이름) — 대조 안 함, 수만 기록
            continue
        for row in rows:
            # 옛 raw (10-06 무렵)엔 start 키가 없음 → 'start' 하나로 봄
            rs = [r for r in res if r["policy"] == row["policy"] and r.get("start", "start") == row.get("start", "start")]
            n += 1
            calc = {"runs": len(rs), "goal_rate": float(np.mean([r["goal"] for r in rs])),
                    "left_rate": float(np.mean([r["left"] for r in rs])), "whiteout_rate": float(np.mean([r["whiteout"] for r in rs]))}
            if "timeout_rate" in row:
                calc["timeout_rate"] = float(np.mean([r.get("timeout", False) for r in rs]))
            d = {k: (row.get(k), v) for k, v in calc.items() if row.get(k) is None or abs(row[k] - v) > 1e-9}
            if d:
                bad.append({"file": s.name, "policy": row["policy"], "start": row["start"], "diff": d})
    story_bad, ns = [], 0
    for m in sorted(Path("runs/story").glob("*_meta.json")):
        res_f = m.with_name(m.name.replace("_meta.json", ".json"))
        if not res_f.exists():
            continue
        meta, res = json.loads(m.read_text(encoding="utf-8")), json.loads(res_f.read_text(encoding="utf-8"))
        ns += 1
        ok = sum(r["ok"] for r in res)
        if meta.get("ok") != ok or meta.get("episodes") != len(res):
            story_bad.append({"file": m.name, "meta_ok": meta.get("ok"), "ok": ok, "meta_eps": meta.get("episodes"), "eps": len(res)})
    return {"diagnose_rows": n, "diagnose_bad": bad, "diagnose_old_format": len(old), "story_files": ns, "story_bad": story_bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", default="newbark_labdoor,route29,route30,cherrygrove_pc")
    ap.add_argument("--clocks", default="lock,day,night")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--skip-records", action="store_true")
    ap.add_argument("--records-only", action="store_true", help="결과 기록 대조만")
    ap.add_argument("--dump", action="store_true", help="세 경로의 걸음별 기록 전체를 결과에 넣음 (원인 확인용)")
    args = ap.parse_args()
    if args.records_only:
        r = check_records()
        OUT.mkdir(parents=True, exist_ok=True)
        f = OUT / f"records_{time.strftime('%m%d_%H%M%S')}.json"
        f.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"diagnose 행 {r['diagnose_rows']} (어긋남 {len(r['diagnose_bad'])}, 옛 형식 건너뜀 {r['diagnose_old_format']}), "
              f"run_story 파일 {r['story_files']} (어긋남 {len(r['story_bad'])})\n저장: {f}")
        return
    from stable_baselines3 import PPO
    real = PPO.load(MODEL, device="cpu")
    # diagnose 시작 흘림(0~29프레임)이 0인 시드만 씀 → 세 경로가 같은 프레임에서 시작
    seeds = [s for s in range(1000) if np.random.default_rng(s).integers(0, 30) == 0][:args.seeds]
    out = {"model": MODEL, "seeds": seeds, "cases": [], "palette": {}, "started": time.strftime("%m-%d %H:%M:%S")}
    for clock in args.clocks.split(","):
        os.environ["POKE_CLOCK"] = clock
        for segment in args.segments.split(","):
            out["palette"][f"{segment}|{clock}"] = first_frame_palette(segment)
            for seed in seeds:
                t0 = time.time()
                p1, e1, kind = path_train(segment, seed, real)
                acts = [t["action"] for t in p1.trace]
                p2, e2 = path_diagnose(segment, seed, acts)
                p3, e3 = path_story(segment, acts)
                case = {"segment": segment, "clock": clock, "seed": seed, "train_env": kind, "n": len(acts),
                        "end": {"train": e1, "diagnose": e2, "story": e3},
                        "diff_train_diagnose": first_diff(p1.trace, p2.trace),
                        "diff_train_story": first_diff(p1.trace, p3.trace),
                        "stale": {"train": stale_summary(p1.trace), "diagnose": stale_summary(p2.trace), "story": stale_summary(p3.trace)},
                        "seconds": round(time.time() - t0, 1)}
                if args.dump:
                    case["trace"] = {"train": p1.trace, "diagnose": p2.trace, "story": p3.trace}
                out["cases"].append(case)
                print(f"{segment:<16}{clock:<6}s{seed:<4} 걸음 {len(acts):>4}  학습↔진단 {fmt(case['diff_train_diagnose'])}  "
                      f"학습↔통합 {fmt(case['diff_train_story'])}  stale 학습/진단/통합 "
                      f"{case['stale']['train']['n']}/{case['stale']['diagnose']['n']}/{case['stale']['story']['n']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{time.strftime('%m%d_%H%M%S')}.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")   # 기록 대조 전에 먼저 저장
    if not args.skip_records:
        out["records"] = check_records()
        f.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장:", f)


def fmt(d):
    return "같음" if d is None else f"#{d['i']} {','.join(d['fields'])}"


if __name__ == "__main__":
    main()
