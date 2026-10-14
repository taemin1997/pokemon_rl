"""
E26 트랙 A 자동 복구 (10-14): 실패 감지 → Go-Explore → 학습 데이터 → 정책 적응 → 재도전을 사람 없이 이음

흐름 (시도 k = 1, 2, ...)
1. 실행: 통합 실행 (run_story, 시계 고정, 도라지체육관 목표, 도라지시티 회복 생략 — 판정 3과 같은 조건), --detect stop
   → 감지 없이 끝나면 "재도전 통과"로 종료
2. 실패 감지 (train/detector.py): 경계 상태 (그 목표가 시작된 순간) + 이유 저장
3. 탐색: 경계 상태에서 Go-Explore (BFS·거리표·지도 없음). 허용 맵 = 감지된 시작 맵, 목표 = 상위 계획이 준 다음 목표 맵
4. 학습 데이터: 발견 경로를 시연으로 새 구간 자동 등록 (POKE_EXTRA_SEGMENTS)
5. 적응: 기존 16구간 + 새 구간(가중치 3) 섞어 이어 학습. 예산 = budgets[k-1] (합계 ≤ PPO 15만, 설계 보강 2)
6. 재도전: 1로 (새 모델)
예산을 다 쓰고도 감지되면 "예산 소진"으로 끝 (마지막 시도는 확인만)

사람이 미리 정한 것 (결과 JSON의 provided에 기록): 최종 목표 맵 순서, 감지 기준값, 예산, 탐색 반복 수, 허용 맵 규칙
("감지된 시작 맵 하나"), 새 구간 가중치, 시계 조건, 이벤트·전투 매크로, 도라지시티 회복 생략
격리: 결과물은 --root 아래 (기본 states/e26/a/s<시드>), 모델은 A_E26_s<시드>_r<k>. 트랙 B 데이터·평가 상태는 안 씀

실행: uv run python -m train.selfrepair --seed 0
시험: uv run python -m train.selfrepair --seed 0 --root states/e26/a_smoke --budgets 2048 --story-episodes 2 --ge-iters 300 --name-prefix SMOKE_A
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

M2B_PARTS = ["house2f", "newbark", "route29", "cherrygrove", "cherrygrove_pc", "cherrygrove_pc_exit", "route30",
             "route30_rev", "cherrygrove_rev", "route29_rev", "newbark_lab",
             "route30_north", "route31", "violet_gym", "violet_pc", "violet_pc_exit"]
M2B_WEIGHTS = [1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 2, 2, 2, 1, 1]
TRAIN_FLAGS = ["--reward", "goal_only", "--move", "tile", "--goal", "--hidden", "256,256", "--fast-battle", "--auto-event", "--no-live"]
TARGETS = "26,10;24,5;10,7"


def run(cmd, env, log):
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", *cmd], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (log).write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    return r, round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="models/M2B_multi/final.zip")
    ap.add_argument("--root", default=None)
    ap.add_argument("--name-prefix", default=None)
    ap.add_argument("--budgets", default="100000,50000", help="시도별 PPO 예산 (합계 ≤ 15만)")
    ap.add_argument("--story-episodes", type=int, default=20)
    ap.add_argument("--story-seed-offset", type=int, default=2000, help="A·B 같은 시드 묶음 (평가 상태 E-lab은 1000)")
    ap.add_argument("--ge-iters", type=int, default=4000)
    ap.add_argument("--ge-after", type=int, default=1000)
    ap.add_argument("--new-weight", type=float, default=3.0)
    ap.add_argument("--clock", default="lock")
    ap.add_argument("--detect-rule-cap", action="store_true", help="규칙 상한 끊김도 감지 (10-14 추가 기준, 사용자 확인 항목)")
    a = ap.parse_args()
    budgets = [int(b) for b in a.budgets.split(",")]
    assert sum(budgets) <= 150000, "PPO 예산 합계 15만 초과"
    root = Path(a.root or f"states/e26/a/s{a.seed}")
    root.mkdir(parents=True, exist_ok=True)
    prefix = a.name_prefix or f"A_E26_s{a.seed}"
    seg_file, preset_file = root / "segments.json", root / "presets.json"
    segs = json.loads(seg_file.read_text(encoding="utf-8")) if seg_file.exists() else []
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "POKE_CLOCK": a.clock,
           "POKE_EXTRA_SEGMENTS": str(seg_file.resolve()), "POKE_EXTRA_PRESETS": str(preset_file.resolve())}
    model = a.init
    out = {"seed": a.seed, "init": a.init, "budgets": budgets, "attempts": [], "human_interventions": 0,
           "provided": {"targets": TARGETS, "no_heal_at": "10,5", "detector": "train/detector.py 300/60/2" + (" + rule_cap" if a.detect_rule_cap else ""), "budgets": budgets,
                        "ge_iters": a.ge_iters, "ge_after": a.ge_after, "allowed_rule": "감지된 시작 맵 하나",
                        "new_segment_weight": a.new_weight, "clock": a.clock, "macros": "run_event·자동 전투·회복·연구소 규칙"},
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    totals = {"ge_moves": 0, "ppo_steps": 0, "story_steps": 0, "seconds": {"story": 0, "ge": 0, "train": 0}}

    for k in range(1, len(budgets) + 2):
        att = {"k": k, "model": model}
        d = root / f"r{k}"
        d.mkdir(exist_ok=True)
        # 1) 실행 + 감지
        r, sec = run(["train.run_story", model, "--headless", "--episodes", str(a.story_episodes), "--quiet",
                      "--targets", TARGETS, "--no-heal-at", "10,5", "--max-steps", "6000",
                      "--seed-offset", str(a.story_seed_offset), "--detect", "stop", "--detect-out", str(d / "detect")] + (["--detect-rule-cap"] if a.detect_rule_cap else []),
                     env, d / "story.log")
        totals["seconds"]["story"] += sec
        m = re.findall(r"^저장: (.+)$", r.stdout, re.M)
        story = json.loads(Path(m[-1].strip()).read_text(encoding="utf-8")) if m else []
        totals["story_steps"] += sum(x.get("steps", 0) for x in story)
        att["story"] = {"file": m[-1].strip() if m else None, "episodes": len(story), "ok": sum(x["ok"] for x in story),
                        "seconds": sec, "returncode": r.returncode}
        det_file = d / "detect" / "detection.json"
        if r.returncode != 0:
            att["stage"] = "실행 오류 (실험 무효 후보)"
            out["attempts"].append(att)
            break
        if not det_file.exists():
            att["stage"] = "감지 없음 → 재도전 통과" if k > 1 else "감지 없음 (실패 재현 안 됨)"
            out["attempts"].append(att)
            break
        det = json.loads(det_file.read_text(encoding="utf-8"))
        att["detection"] = det
        if k > len(budgets):
            att["stage"] = "예산 소진 (감지됨, 더 적응 안 함)"
            out["attempts"].append(att)
            break
        # 2~4) 새 구간 등록 + Go-Explore
        name = f"e26a_s{a.seed}_r{k}"
        sys.path.insert(0, ".")
        from envs.segments import SEGMENTS
        dist_of = next((n for n, sg in SEGMENTS.items() if not sg.eval_only and list(sg.goal_maps[0]) == det["goal"]
                        and tuple(det["start_map"]) in {tuple(x) for x in sg.allowed_maps}), None)   # 진단 지표용 거리표만
        segs.append({"name": name, "allowed_maps": [det["start_map"]], "goal_maps": [det["goal"]],
                     "start_state": str(d / "detect" / "boundary.state"), "trainers": True, "dist_of": dist_of,
                     "demo_file": str(d / "demo_goexplore.pkl"),
                     "note": f"E26 트랙 A 자동 생성 (시도 {k}, 감지 {det['reason']})"})
        seg_file.write_text(json.dumps(segs, ensure_ascii=False, indent=1), encoding="utf-8")
        r, sec = run(["train.go_explore", name, "--iters", str(a.ge_iters), "--after-goal", str(a.ge_after),
                      "--seed", str(a.seed), "--out", str(d / "demo_goexplore.pkl")], env, d / "go_explore.log")
        totals["seconds"]["ge"] += sec
        st = re.search(r"\[탐색\] (\{.*\})", r.stdout)
        stats = __import__("ast").literal_eval(st.group(1)) if st else {}
        totals["ge_moves"] += int(stats.get("moves", 0))
        att["explore"] = {"segment": name, "found": (d / "demo_goexplore.pkl").exists(), "stats": stats, "seconds": sec,
                          "demo": re.findall(r"\[시연\] (.+)", r.stdout)}
        if not att["explore"]["found"]:
            att["stage"] = "자동 탐색 실패 (목표 경로 못 찾음)"
            segs.pop(); seg_file.write_text(json.dumps(segs, ensure_ascii=False, indent=1), encoding="utf-8")
            out["attempts"].append(att)
            break
        # 5) 적응
        preset = f"e26a_s{a.seed}_r{k}"
        presets = json.loads(preset_file.read_text(encoding="utf-8")) if preset_file.exists() else {}
        presets[preset] = {"base": "newbark", "parts": M2B_PARTS + [s["name"] for s in segs],
                           "weights": M2B_WEIGHTS + [a.new_weight] * len(segs), "_class": "multi_seg"}
        preset_file.write_text(json.dumps(presets, ensure_ascii=False, indent=1), encoding="utf-8")
        mname = f"{prefix}_r{k}"
        r, sec = run(["train.train_ppo", "--env", preset, *TRAIN_FLAGS, "--init", model, "--steps", str(budgets[k - 1]),
                      "--name", mname, "--seed", str(a.seed)], env, d / "train.log")
        totals["seconds"]["train"] += sec
        totals["ppo_steps"] += budgets[k - 1]
        new_model = f"models/{mname}/final.zip"
        att["adapt"] = {"preset": preset, "model": new_model, "steps": budgets[k - 1], "seconds": sec,
                        "ok": Path(new_model).exists(), "returncode": r.returncode}
        if not Path(new_model).exists():
            att["stage"] = "학습 오류 (실험 무효 후보)"
            out["attempts"].append(att)
            break
        att["stage"] = "감지 → 탐색 → 적응 완료, 재도전으로"
        out["attempts"].append(att)
        model = new_model
        (root / "selfrepair.json").write_text(json.dumps({**out, "totals": totals, "final_model": model}, ensure_ascii=False, indent=1),
                                              encoding="utf-8")

    totals["env_steps_total"] = totals["ge_moves"] + totals["ppo_steps"] + totals["story_steps"]
    out.update({"totals": totals, "final_model": model, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                "stages": {"자동 감지": any("detection" in t for t in out["attempts"]),
                           "자동 탐색": any(t.get("explore", {}).get("found") for t in out["attempts"]),
                           "정책 적응": any(t.get("adapt", {}).get("ok") for t in out["attempts"]),
                           "무인 재도전 통과": out["attempts"][-1].get("stage", "").startswith("감지 없음 → 재도전 통과")}})
    (root / "selfrepair.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"stages": out["stages"], "final_model": model, "totals": totals}, ensure_ascii=False))


if __name__ == "__main__":
    main()
