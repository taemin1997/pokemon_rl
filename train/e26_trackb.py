"""
E26 트랙 B 경계 출발점 자동 수집 (10-14) — 사람이 설계한 방법: "통합 실행에서 구간이 바뀌는 순간 상태를 전부 모아 시작점으로"

harvest: M2B로 통합 실행 (A와 같은 조건·같은 시드 묶음: 시계 고정, 도라지체육관 목표, 도라지시티 회복 생략, 시드 2000~)
         → run_story --harvest-dir: 정책이 새 구간을 맡는 순간마다 상태 저장
build:   (구간, 맵·좌표·방향·이벤트 해시) 기준 중복 제거 → 평가 상태 누출 필터 → 구간별 추가 시작 상태 목록 + 학습 설정
         누출 필터: 평가 상태와 sha1 같음 → 뺌 / E-violet과 의미 키 같음 → 뺌 (전이 평가 전용) /
                   E-lab과 의미 키 같음 → --elab-semantic (keep: 남기고 개수 기록, drop: 뺌) — 사용자 결정 대기 항목
train:   기존 16구간 섞기 (M2B 가중치 그대로) + 추가 시작 상태가 있는 구간은 extra_prob 확률로 거기서 시작, 15만
격리: 결과물 states/e26/b/, 모델 B_E26_s<시드>. 트랙 A 데이터 안 씀

실행: uv run python -m train.e26_trackb harvest
      uv run python -m train.e26_trackb build
      uv run python -m train.e26_trackb train --seed 0
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from train.selfrepair import M2B_PARTS, M2B_WEIGHTS, TARGETS, TRAIN_FLAGS

ROOT = Path("states/e26/b")


def harvest(a):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "POKE_CLOCK": a.clock}
    (ROOT / "harvest").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "train.run_story", a.init, "--headless", "--episodes", str(a.episodes), "--quiet",
                        "--targets", TARGETS, "--no-heal-at", "10,5", "--max-steps", "6000",
                        "--seed-offset", str(a.seed_offset), "--harvest-dir", str(ROOT / "harvest")],
                       env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (ROOT / "harvest.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    n = sum(1 for _ in open(ROOT / "harvest" / "manifest.jsonl", encoding="utf-8")) if (ROOT / "harvest" / "manifest.jsonl").exists() else 0
    info = {"episodes": a.episodes, "seed_offset": a.seed_offset, "clock": a.clock, "states": n, "seconds": round(time.time() - t0),
            "returncode": r.returncode, "summary": [l for l in r.stdout.splitlines() if l.startswith("성공") or l.startswith("저장")]}
    (ROOT / "harvest.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(info)


def build(a):
    index = json.loads(Path("states/e26/eval/index.json").read_text(encoding="utf-8"))
    ev = index["states"]
    rows = [json.loads(l) for l in open(ROOT / "harvest" / "manifest.jsonl", encoding="utf-8")]
    from train.evalrec import sha1_file
    seen, extras, log = set(), {}, {"total": len(rows), "dup": 0, "drop_sha1": 0, "drop_violet": 0, "elab_semantic": 0,
                                    "elab_semantic_kept": 0}
    for r in rows:
        key = (r["segment"], tuple(r["position"]), r["facing"], r["event_sha1"])
        if key in seen:
            log["dup"] += 1
            continue
        seen.add(key)
        sha = sha1_file(r["path"])
        if sha in {v["sha1"] for v in ev.values()}:
            log["drop_sha1"] += 1
            continue
        sem = lambda e: (r["position"] == e["position"] and r["facing"] == e["facing"] and r["event_sha1"] == e["event_sha1"]
                         and r["goal"] == e["goal"])
        if sem(ev["e_violet"]):
            log["drop_violet"] += 1
            continue
        if sem(ev["e_lab"]):
            log["elab_semantic"] += 1
            if a.elab_semantic == "drop":
                continue
            log["elab_semantic_kept"] += 1
        extras.setdefault(r["segment"], []).append(r["path"])
    log["kept"] = {k: len(v) for k, v in extras.items()}
    log["elab_semantic_mode"] = a.elab_semantic
    presets = {"e26b": {"base": "newbark", "parts": M2B_PARTS, "weights": M2B_WEIGHTS, "_class": "multi_seg",
                        "extra_starts": extras, "extra_prob": a.extra_prob}}
    (ROOT / "presets.json").write_text(json.dumps(presets, ensure_ascii=False, indent=1), encoding="utf-8")
    (ROOT / "build.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(log, ensure_ascii=False))


def train(a):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "POKE_EXTRA_PRESETS": str((ROOT / "presets.json").resolve())}
    env.pop("POKE_EXTRA_SEGMENTS", None)
    name = a.name or f"B_E26_s{a.seed}"
    t0 = time.time()
    log = ROOT / f"train_{name}.log"
    r = subprocess.run([sys.executable, "-m", "train.train_ppo", "--env", "e26b", *TRAIN_FLAGS, "--init", a.init,
                        "--steps", str(a.steps), "--name", name, "--seed", str(a.seed)],
                       env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    info = {"model": f"models/{name}/final.zip", "ok": Path(f"models/{name}/final.zip").exists(), "steps": a.steps,
            "seconds": round(time.time() - t0), "returncode": r.returncode}
    (ROOT / f"train_{name}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(info)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["harvest", "build", "train"])
    ap.add_argument("--init", default="models/M2B_multi/final.zip")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed-offset", type=int, default=2000)
    ap.add_argument("--clock", default="lock")
    ap.add_argument("--elab-semantic", choices=["keep", "drop"], default="keep")
    ap.add_argument("--extra-prob", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--name", default=None)
    ap.add_argument("--root", default=None, help="시험용 다른 폴더")
    a = ap.parse_args()
    if a.root:
        ROOT = Path(a.root)
    {"harvest": harvest, "build": build, "train": train}[a.cmd](a)
