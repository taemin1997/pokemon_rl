"""
E26 2단계: 평가 상태 고정 + 누출 검사 (10-14)

make: 평가 상태 2개를 만들어 states/e26/eval/에 저장하고 index.json에 기록 (sha1, 위치, 방향, 이벤트 플래그 해시, 목표, 만든 방법)
  - E-lab: 통합 실행(run_story)을 시계 고정 + 시드 오프셋 1000으로 돌려 연구소 이벤트 뒤 연두마을에 처음 나온 순간 저장
           (A·B 학습용 통합 실행은 다른 시드를 씀 → 같은 판을 다시 만들지 않음)
  - E-violet: violet_pc 도착 상태(센터 문 안)에서 회복 매크로(heal_at_pc) 직후 = 간호사 앞. 전이 평가 전용
leak: 평가 상태와 "의미가 같은" 학습 시작 상태가 학습 데이터에 있나 검사
  - sha1 같음 + 의미적 중복: (맵, 좌표, 방향, 이벤트 플래그 해시, 목표 맵)이 같은 상태
  - 검사 대상: 모든 학습 구간의 시작 상태, 시연(demo.pkl, demo_goexplore.pkl) 상태, 경로 밖(offpath.pkl) 상태, 추가로 주는 폴더
  - 메모리 주소 (10-14 메모리 비교로 찾음, [후보]):
    방향 D2BA (아래 0 / 위 4 / 왼 8 / 오 12, D2BF·D2CF도 같은 값)
    이벤트 플래그 D8B4~D9B3 (영문판 wEventFlags D7B7 + 0xFD. 스토리 순서 상태 7개에서 켜진 비트 218 → 226으로 늘기만 함)

실행: uv run python -m train.e26_eval_states make
      uv run python -m train.e26_eval_states leak [추가 폴더 ...]
"""

import hashlib
import io
import json
import os
import pickle
import subprocess
import sys
import zlib
from pathlib import Path

from pyboy import PyBoy

from envs import memory_map as mm
from envs.segments import SEGMENTS

OUT = Path("states/e26/eval")
FACING = 0xD2BA
EVENT_FLAGS, EVENT_LEN = 0xD8B4, 0x100
EVAL = {"e_lab": {"segment": "newbark_labdoor", "goal": [24, 3]},
        "e_violet": {"segment": "violet_pc_counter", "goal": [10, 5]}}


def _pb():
    pb = PyBoy("roms/gold_rom.gb", window="null")
    pb.rtc_lock_experimental(True)
    return pb


def key_of(pb, state_bytes):
    """의미 키: (맵 그룹, 맵 번호, x, y, 방향, 이벤트 플래그 해시)"""
    pb.load_state(io.BytesIO(state_bytes))
    pos = mm.read_position(pb)
    ev = bytes(pb.memory[1, EVENT_FLAGS + i] for i in range(EVENT_LEN))
    return (*pos, pb.memory[1, FACING], hashlib.sha1(ev).hexdigest()[:12])


def _unz(b):
    try:
        return zlib.decompress(b)
    except Exception:
        return b


def make():
    from envs.heal import heal_at_pc
    from train.evalrec import code_version, sha1_file
    OUT.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "POKE_CLOCK": "lock", "PYTHONIOENCODING": "utf-8"}
    lab = OUT / "e_lab.state"
    made_by = None
    for seed in range(1000, 1010):     # 연구소 이벤트까지 못 간 판이면 다음 시드
        if lab.exists():
            break
        r = subprocess.run([sys.executable, "-m", "train.run_story", "models/M2B_multi/final.zip", "--headless", "--episodes", "1",
                            "--targets", "26,10;24,5;10,7", "--no-heal-at", "10,5", "--max-steps", "3000",
                            "--seed-offset", str(seed), "--snap-lab-exit", str(lab)],
                           env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        made_by = {"how": "run_story --snap-lab-exit", "seed": seed, "clock": "lock", "model": "models/M2B_multi/final.zip"}
        print(f"시드 {seed}: {'저장됨' if lab.exists() else '연구소 전 실패'}")
    assert lab.exists(), "E-lab을 못 만듦"
    pb = _pb()
    pb.load_state(open("states/segments/violet_pc/reached_goal.state", "rb"))
    pb.tick(1, False)
    h = heal_at_pc(pb)
    buf = io.BytesIO(); pb.save_state(buf)
    (OUT / "e_violet.state").write_bytes(buf.getvalue())
    index = {"made": code_version(), "states": {}}
    for name, info in EVAL.items():
        p = OUT / f"{name}.state"
        k = key_of(pb, p.read_bytes())
        index["states"][name] = {**info, "path": str(p), "sha1": sha1_file(p), "position": list(k[:4]), "facing": k[4],
                                 "event_sha1": k[5], "party_hp": mm.read_party_hp_total(pb),
                                 "made_by": made_by if name == "e_lab" else {"how": "violet_pc/reached_goal.state + heal_at_pc", "heal": h}}
        print(name, index["states"][name]["position"], "방향", k[4], "이벤트", k[5], "HP", index["states"][name]["party_hp"])
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장:", OUT / "index.json")


def training_sources(extra_dirs=()):
    """(출처 이름, 목표 맵, 상태 bytes) — 학습 구간(평가 전용 제외)의 모든 시작 후보"""
    for name, seg in SEGMENTS.items():
        if seg.eval_only or not seg.goal_maps:
            continue
        goal = list(seg.goal_maps[0])
        if Path(seg.start_state).exists():
            yield f"{name}:start", goal, Path(seg.start_state).read_bytes()
        folder = Path(seg.demo_path).parent
        for f in ("demo.pkl", "demo_goexplore.pkl"):
            if (folder / f).exists():
                d = pickle.load(open(folder / f, "rb"))
                for i, s in enumerate(d["states"]):
                    yield f"{name}:{f}[{i}]", goal, _unz(s)
        if (folder / "offpath.pkl").exists():
            d = pickle.load(open(folder / "offpath.pkl", "rb"))
            for pos, t in d["tiles"].items():
                yield f"{name}:offpath{pos}", goal, _unz(t["state"])
    for d in extra_dirs:
        for f in Path(d).rglob("*.state"):
            yield f"extra:{f}", None, f.read_bytes()
        for f in Path(d).rglob("*.pkl"):          # 트랙 A의 Go-Explore 경로 (시연 형식)
            try:
                dd = pickle.load(open(f, "rb"))
            except Exception:
                continue
            for i, st in enumerate(dd.get("states", [])):
                yield f"extra:{f}[{i}]", None, _unz(st)


def leak(extra_dirs=()):
    index = json.loads((OUT / "index.json").read_text(encoding="utf-8"))
    pb = _pb()
    ev = {n: (tuple(s["position"]) + (s["facing"], s["event_sha1"]), s["goal"], s["sha1"]) for n, s in index["states"].items()}
    hits = {n: [] for n in ev}
    near = {n: [] for n in ev}
    n_src = 0
    for src, goal, b in training_sources(extra_dirs):
        n_src += 1
        sha = hashlib.sha1(b).hexdigest()
        k = key_of(pb, b)
        for n, (ek, eg, es) in ev.items():
            if sha == es or (k == ek and (goal is None or goal == eg)):
                hits[n].append(src)
            elif k[:4] == ek[:4]:
                near[n].append(f"{src} (위치 같음, 방향 {k[4]} / 이벤트 {k[5]}, 목표 {goal})")
    res = {"sources": n_src, "extra_dirs": list(extra_dirs),
           "leak": {n: v for n, v in hits.items()}, "same_position_only": {n: v[:20] for n, v in near.items()},
           "pass": all(not v for v in hits.values())}
    Path("runs/e26").mkdir(parents=True, exist_ok=True)
    out = Path("runs/e26/stage2_leak.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"검사한 학습 상태 {n_src}개 → 누출 {'없음 ✅' if res['pass'] else '있음 ❌'}")
    for n in ev:
        print(f"  {n}: 같은 상태 {len(hits[n])}개, 위치만 같음 {len(near[n])}개")
    print("저장:", out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "make":
        make()
    elif cmd == "leak":
        leak(sys.argv[2:])
