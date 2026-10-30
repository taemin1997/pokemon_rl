"""
열린 문제 1 측정 (10-19): 막힌 걸음마다 대화 매크로(run_event)가 도는 비용 — 관찰만, 행동·모델·규칙 변경 없음

run_event를 감싸서 부를 때마다 (위치, SCRIPT_LOCK 값, 누른 횟수, 끝남 여부, 걸린 실제 시간)을 적음. 원래 함수를 그대로 부르므로 판 내용은 같음
  - diag: train.diagnose.run_one을 그대로 씀 (같은 시작 상태·시드·판당 5분 상한) → E-violet·E-lab·16구간
  - story: train.run_story.main을 그대로 씀 (판정 3과 같은 인자, 판 수만 줄임)
결과: runs/e26/macro_probe/<이름>.json (판별 기록 + 매크로 호출 목록), 요약은 `report`

  uv run python -m train.macro_probe diag violet_pc_counter --runs 50
  uv run python -m train.macro_probe story models/A_E26_s1_r2/final.zip --episodes 5
  uv run python -m train.macro_probe report
"""

import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

OUT = Path("runs/e26/macro_probe")
CALLS = []   # 이 프로세스에서 일어난 run_event 호출 기록


def _install():
    """envs.gold_env(학습·평가 환경의 잠김 처리)와 train.run_story(규칙·잠김 처리)가 import한 run_event를 감쌈"""
    import envs.gold_env as ge
    from envs import memory_map as mm
    if getattr(ge.run_event, "_probe", False):
        return
    orig = ge.run_event

    def make(src):
        def wrapped(pb, *a, **k):
            pos, lock = mm.read_position(pb), mm.read_byte(pb, mm.SCRIPT_LOCK)
            t0 = time.perf_counter()
            r = orig(pb, *a, **k)
            CALLS.append({"src": src, "pos": list(pos), "lock": lock, "presses": r["presses"], "done": r["done"],
                          "battle": r["battle"], "sec": round(time.perf_counter() - t0, 3),
                          "lock_after": mm.read_byte(pb, mm.SCRIPT_LOCK)})
            return r
        wrapped._probe = True
        return wrapped

    ge.run_event = make("env")
    if "train.run_story" in sys.modules:
        sys.modules["train.run_story"].run_event = make("story")


def _diag_one(job):
    from train.diagnose import run_one
    _install()
    CALLS.clear()
    t0 = time.perf_counter()
    r = run_one(job)
    r["wall"] = round(time.perf_counter() - t0, 1)
    r["macro"] = list(CALLS)
    return r


def diag(seg, paths, runs, workers=6):
    from train.diagnose import SEGMENTS
    os.environ.setdefault("POKE_EP_TIMEOUT", "300")   # 최종 평가와 같은 판당 상한
    s = SEGMENTS[seg]
    jobs = [(seg, p, name, path, seed) for p in paths for name, path in s.all_eval_states().items() for seed in range(runs)]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_diag_one, jobs))
    _write(f"diag_{seg}_{os.environ.get('POKE_CLOCK', 'wall')}", {"kind": "diag", "segment": seg, "runs": runs, "results": res})


def story(model, episodes, seed_offset=5000):
    import train.run_story as rs
    _install()
    # run_story는 판 결과를 리스트로만 들고 있다가 끝에 파일로 씀 → 판 경계는 print를 가로채서 나눔
    per_ep, mark = [], [0]
    real_print = print

    def tap(*a, **k):
        s = " ".join(map(str, a))
        if s.startswith("[") and ("] 성공" in s or "] 실패" in s):
            per_ep.append(CALLS[mark[0]:])
            mark[0] = len(CALLS)
        real_print(*a, **k)
    rs.print = tap
    sys.argv = ["run_story", model, "--headless", "--episodes", str(episodes), "--quiet", "--targets", "26,10;24,5;10,7",
                "--no-heal-at", "10,5", "--max-steps", "6000", "--seed-offset", str(seed_offset)]
    t0 = time.perf_counter()
    rs.main()
    _write(f"story_{Path(model).parent.name}_{os.environ.get('POKE_CLOCK', 'wall')}",
           {"kind": "story", "model": model, "episodes": episodes, "wall": round(time.perf_counter() - t0), "macro_per_episode": per_ep})


def _write(name, d):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}_{time.strftime('%m%d_%H%M%S')}.json"
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    print("저장:", p, flush=True)


def report():
    out = ["# 열린 문제 1 측정: 대화 매크로(run_event) 비용 (자동 생성 `train.macro_probe report`)", "",
           "관찰만. 판 내용은 최종 평가와 같은 조건 (diagnose: 같은 시드·판당 5분 상한 / story: 판정 3 인자, 판 수만 줄임)", ""]
    for f in sorted(OUT.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d["kind"] == "diag":
            out += [f"## {f.stem}", "| 모델 | 판 | 도착 | 멈춤 | 매크로 호출/판 | 끝 못 냄 | 누름 합 | 매크로 시간 / 판 전체 시간 | 많이 불린 위치 (map,x,y: 횟수, lock 값) |",
                    "|---|---|---|---|---|---|---|---|---|"]
            by = {}
            for r in d["results"]:
                by.setdefault(r["policy"], []).append(r)
            for pol, rs in by.items():
                calls = [c for r in rs for c in r["macro"]]
                msec, wall = sum(c["sec"] for c in calls), sum(r["wall"] for r in rs)
                top = Counter((tuple(c["pos"][:3]), c["lock"]) for c in calls).most_common(3)
                out.append(f"| {pol} | {len(rs)} | {sum(r['goal'] for r in rs)} | {sum(r['timeout'] for r in rs)} | {len(calls) / len(rs):.1f} | "
                           f"{sum(not c['done'] for c in calls)} | {sum(c['presses'] for c in calls)} | {msec:.0f}초 / {wall:.0f}초 ({msec / wall:.0%}) | "
                           + ", ".join(f"{p}: {n} (lock {l})" for (p, l), n in top) + " |")
            out.append("")
        else:
            eps = d["macro_per_episode"]
            calls = [c for e in eps for c in e]
            msec = sum(c["sec"] for c in calls)
            top = Counter((c["src"], tuple(c["pos"][:3]), c["lock"]) for c in calls).most_common(5)
            out += [f"## {f.stem}", f"- {len(eps)}판, 전체 {d['wall']}초, 매크로 {len(calls)}번 · {msec:.0f}초 ({msec / max(d['wall'], 1):.0%}), "
                    f"끝 못 냄 {sum(not c['done'] for c in calls)}, 누름 합 {sum(c['presses'] for c in calls)}",
                    "- 판별 매크로 호출: " + ", ".join(str(len(e)) for e in eps),
                    "- 많이 불린 곳 (출처, map,x,y, lock): " + ", ".join(f"{s} {p} lock {l}: {n}" for (s, p, l), n in top), ""]
    p = OUT / "report.md"
    p.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "diag":
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd"); ap.add_argument("segment"); ap.add_argument("policies", nargs="+")
        ap.add_argument("--runs", type=int, default=50); ap.add_argument("--workers", type=int, default=6)
        a = ap.parse_args()
        diag(a.segment, a.policies, a.runs, a.workers)
    elif cmd == "story":
        story(sys.argv[2], int(sys.argv[sys.argv.index("--episodes") + 1]) if "--episodes" in sys.argv else 5)
    else:
        report()
