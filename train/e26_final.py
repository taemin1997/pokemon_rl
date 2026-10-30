"""
E26 6단계 최종 평가 + 보고서 (10-14)

eval: 모델 5개 (M2B 기준, A 시드 0·1 최종, B 시드 0·1)를 같은 조건으로
  - E-lab / E-violet × 아침·낮·밤 × 50판 (diagnose, 여러 모델 한 번에)
  - 16구간 × 아침·낮·밤 × 25판 + 감지기 관찰 (POKE_DETECT=1, 행동 영향 없음) → 보존 + 오탐률
  - 판정 3 통합 실행 20판 × 아침·낮·밤 (판정 3과 같은 조건, 규칙 그대로) — 시간대별로 병렬 3개
  결과 경로는 runs/e26/final_runs.json (끊겨도 끝난 항목은 건너뜀)
report: runs/e26/final_report.md — 단계별 성공, 복구율, 비용, 보존, 전이, 판정 3, H-A / H-B 판정 (설계 기준)
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from train.selfrepair import M2B_PARTS, TARGETS

LIST = Path("runs/e26/final_runs.json")
CLOCKS = ["morning", "day", "night"]
DIAG_TIMEOUT = 2400   # 한 번의 diagnose 상한 (초). 5모델×50판 violet이 정상이면 20분 안쪽
STORY_TIMEOUT = 3 * 3600   # 통합 실행 20판 상한 (10-14 수집 20판이 약 20분)


def models():
    m = {"M2B": "models/M2B_multi/final.zip"}
    for s in (0, 1):
        j = Path(f"states/e26/a/s{s}/selfrepair.json")
        if j.exists():
            m[f"A_s{s}"] = json.loads(j.read_text(encoding="utf-8"))["final_model"]
        b = f"models/B_E26_s{s}/final.zip"
        if Path(b).exists():
            m[f"B_s{s}"] = b
    return m


def _save(runs):
    LIST.write_text(json.dumps(runs, ensure_ascii=False, indent=1), encoding="utf-8")


def _diag(seg, clock, paths, runs_n, detect=False):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "POKE_CLOCK": clock}
    env.pop("POKE_EXTRA_SEGMENTS", None)
    if detect:
        env["POKE_DETECT"] = "1"
    # 10-15: violet 평가에서 diagnose의 프로세스 풀이 교착(워커 6개·부모 모두 대기)해 6시간 멈춘 적 있음
    #   → 시간 제한을 넘기면 프로세스 트리째 죽이고 다시 시도. 실패 원인 보려고 stderr를 파일로 남김
    cmd = [sys.executable, "-m", "train.diagnose", seg, *paths, "--runs", str(runs_n), "--workers", "6"]
    log = Path(f"runs/e26/diag_{seg}_{clock}.err")
    for attempt in range(3):
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, encoding="utf-8", errors="replace")
        try:
            out, err = p.communicate(timeout=DIAG_TIMEOUT)
        except subprocess.TimeoutExpired:
            # Windows: venv python은 실제 python을 자식으로 띄우므로 /T로 손자(풀 워커)까지 정리
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
            out, err = p.communicate()
            err += f"\n[시간 초과 {DIAG_TIMEOUT}초, 시도 {attempt + 1}]"
        with log.open("a", encoding="utf-8") as f:
            f.write(f"== {time.strftime('%m-%d %H:%M:%S')} 시도 {attempt + 1} returncode={p.returncode}\n{err[-4000:]}\n")
        line = [l for l in out.splitlines() if l.startswith("저장:")]
        if line:
            return line[-1].split("저장:")[1].strip()
        print(f"{time.strftime('%H:%M:%S')} {seg}|{clock} 시도 {attempt + 1} 실패 → {log}", flush=True)
    return None


def eval_all():
    runs = json.loads(LIST.read_text(encoding="utf-8")) if LIST.exists() else {}
    m = models()
    runs["models"] = m
    paths = list(m.values())
    jobs = [(f"{seg}|{c}", seg, c, 50, False) for seg in ("newbark_labdoor", "violet_pc_counter") for c in CLOCKS]
    jobs += [(f"{seg}|{c}", seg, c, 25, True) for seg in M2B_PARTS for c in CLOCKS]
    for key, seg, c, n, det in jobs:
        if runs.get(key):   # 실패(None)로 기록된 항목은 다시 돌림
            continue
        t0 = time.time()
        runs[key] = _diag(seg, c, paths, n, det)
        print(f"{time.strftime('%H:%M:%S')} {key} {round(time.time() - t0)}초 → {runs[key]}", flush=True)
        _save(runs)
    # 통합 실행: 모델마다 시간대 3개를 동시에 (각각 단일 프로세스)
    for name, path in m.items():
        todo = [c for c in CLOCKS if not runs.get(f"story|{name}|{c}")]
        procs = {}
        for c in todo:
            env = {**os.environ, "PYTHONIOENCODING": "utf-8", "POKE_CLOCK": c}
            env.pop("POKE_EXTRA_SEGMENTS", None)
            procs[c] = subprocess.Popen([sys.executable, "-m", "train.run_story", path, "--headless", "--episodes", "20", "--quiet",
                                         "--targets", TARGETS, "--no-heal-at", "10,5", "--max-steps", "6000", "--seed-offset", "5000"],
                                        env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace")
        for c, p in procs.items():
            try:
                outp = p.communicate(timeout=STORY_TIMEOUT)[0]
            except subprocess.TimeoutExpired:   # 10-15: 멈추면 트리째 정리하고 None으로 남김 (다음 실행 때 다시 돌림)
                subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
                outp = p.communicate()[0]
            line = [l for l in outp.splitlines() if l.startswith("저장:")]
            runs[f"story|{name}|{c}"] = line[-1].split("저장:")[1].strip() if line else None
            print(f"{time.strftime('%H:%M:%S')} story {name} {c} → {runs[f'story|{name}|{c}']}", flush=True)
            _save(runs)


def _raw(summary):
    return json.loads(Path(summary.replace("_summary.json", "_raw.json")).read_text(encoding="utf-8"))


def _nav(rows):
    n = len(rows)
    g = sum(r["goal"] for r in rows)
    w = sum(r["whiteout"] for r in rows)
    return (n - g - w) / n if n else None, g, n


def report():
    runs = json.loads(LIST.read_text(encoding="utf-8"))
    m = runs["models"]
    label = {v: k for k, v in m.items()}
    out = ["# E26 최종 보고서 (자동 생성 `train/e26_final.py report`)", "",
           "> 기준은 `notes/experiments/E26_SELFREPAIR.md` 확정 설계 + 보강. 시드별 따로, 합치지 않음. +10%p는 운영 기준 (통계적 비열등성 아님)",
           "> 10-15 설계 밖 변경: diagnose 판당 실제 시간 5분 상한 추가 (E-violet에서 막힌 걸음마다 대화 매크로가 돌아 한 판이 3시간 가까이 걸림). "
           "넘은 판은 '멈춤'으로 표시하고 길 찾기 실패로 셈. E-lab 3개 시간대는 상한 도입 전에 측정 (모두 1분 안에 끝나 영향 없음)", ""]
    by = {}   # (seg, clock, model) -> (nav, goal, n, rows)
    for key, summ in runs.items():
        if key in ("models",) or key.startswith("story|") or not summ:
            continue
        seg, c = key.split("|")
        for r in _raw(summ):
            lab = next((k for k, v in m.items() if Path(v).parent.name == r["policy"].split("@")[0]), r["policy"])
            by.setdefault((seg, c, lab), []).append(r)
    # 1) 단계별 (A)
    out += ["## 1. 트랙 A 단계별", "| 시드 | 자동 감지 | 자동 탐색 | 정책 적응 | 무인 재도전 통과 | 시도 | 감지 이유 | 사람 개입 |", "|---|---|---|---|---|---|---|---|"]
    costs = []
    for s in (0, 1):
        j = Path(f"states/e26/a/s{s}/selfrepair.json")
        if not j.exists():
            continue
        d = json.loads(j.read_text(encoding="utf-8"))
        st = d.get("stages", {})
        reasons = [t.get("detection", {}).get("reason") for t in d["attempts"] if t.get("detection")]
        out.append(f"| {s} | {'✅' if st.get('자동 감지') else '❌'} | {'✅' if st.get('자동 탐색') else '❌'} | {'✅' if st.get('정책 적응') else '❌'} | "
                   f"{'✅' if st.get('무인 재도전 통과') else '❌'} | {len(d['attempts'])} | {', '.join(map(str, reasons))} | {d['human_interventions']} |")
        t = d.get("totals", {})
        costs.append((f"A_s{s}", t.get("ge_moves"), t.get("ppo_steps"), t.get("story_steps"), t.get("env_steps_total"), t.get("seconds")))
    for s in (0, 1):
        j = Path(f"states/e26/b/train_B_E26_s{s}.json")
        h = Path("states/e26/b/harvest.json")
        if j.exists():
            tj, hj = json.loads(j.read_text(encoding="utf-8")), json.loads(h.read_text(encoding="utf-8")) if h.exists() else {}
            costs.append((f"B_s{s}", 0, tj["steps"], None, None, {"harvest": hj.get("seconds"), "train": tj["seconds"]}))
    out += ["", "- 10-20: 재도전·감지는 지난 화면 버그가 있던 실행기로 돌았음. 시드 1 마지막 실패(연구소 문 앞 첫걸음)는 그 버그 "
            "(같은 상태를 diagnose로 25/25 도착), 시드 0 마지막 실패(cherrygrove)는 잊어버림 (diagnose 9/25). '무인 재도전 통과 ❌'는 기록 그대로 둠"]
    out += ["", "## 2. 비용 (분리 측정)", "| 모델 | Go-Explore 걸음 | PPO 스텝 | 통합 실행 걸음 | 총 환경 걸음 | 시간(초) |", "|---|---|---|---|---|---|"]
    out += [f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]} | {c[5]} |" for c in costs]
    # 3) 복구 / 전이
    for seg, title in (("newbark_labdoor", "3. 복구 (E-lab 길 찾기 실패율)"), ("violet_pc_counter", "4. 미학습 경계 전이 (E-violet, 학습 금지)")):
        out += ["", f"## {title}", "| 모델 | " + " | ".join(CLOCKS) + " |", "|---|" + "---|" * len(CLOCKS)]
        for lab in m:
            cells = []
            for c in CLOCKS:
                rows = by.get((seg, c, lab), [])
                f, g, n = _nav(rows)
                to = sum(r.get("timeout", False) for r in rows)   # 10-15: 판당 5분 상한 초과 = 멈춤 실패 (실패율에 포함)
                cells.append(f"{f:.0%} ({g}/{n} 도착{f', 멈춤 {to}' if to else ''})" if f is not None else "-")
            out.append(f"| {lab} | " + " | ".join(cells) + " |")
    # 5) 보존
    out += ["", "## 5. 기존 16구간 보존 (길 찾기 실패율, M2B 대비 변화. 25판, 시간대별 최악)", "| 구간 | " + " | ".join(m) + " |", "|---|" + "---|" * len(m)]
    worst = {lab: (0, 0) for lab in m}       # (최대 실패율, 최대 증가)
    fp = {lab: [0, 0] for lab in m}
    for seg in M2B_PARTS:
        cells = []
        for lab in m:
            vals = []
            for c in CLOCKS:
                rows = by.get((seg, c, lab), [])
                f, _, _ = _nav(rows)
                b, _, _ = _nav(by.get((seg, c, "M2B"), []))
                if f is not None:
                    vals.append((f, f - (b or 0)))
                for r in rows:
                    if isinstance(r.get("detect"), dict) and r["goal"]:
                        fp[lab][0] += 1
                    fp[lab][1] += 1 if r["goal"] else 0
            if vals:
                fmax, dmax = max(v[0] for v in vals), max(v[1] for v in vals)
                worst[lab] = (max(worst[lab][0], fmax), max(worst[lab][1], dmax))
                cells.append(f"{fmax:.0%} ({dmax:+.0%})")
            else:
                cells.append("-")
        out.append(f"| {seg} | " + " | ".join(cells) + " |")
    out += ["", "- 감지기 오탐 (도착한 판인데 감지가 울림 / 도착한 판): " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in fp.items())]
    # 6) 판정 3
    # 10-20: headless run_story가 규칙·매크로 뒤 예측에 지난 화면을 줬음 → 옛 통합 실행은 정책 측정으로 무효.
    #   판정은 고친 실행기(--fresh-obs 기본)로 다시 잰 runs/e26/story_fresh.json으로 하고, 옛 표는 참고로만 남김
    fresh_p = Path("runs/e26/story_fresh.json")
    fresh = json.loads(fresh_p.read_text(encoding="utf-8")) if fresh_p.exists() else {}

    def story_table(get):
        rows, got = [], {}
        for lab in m:
            cells = []
            for c in CLOCKS:
                f = get(lab, c)
                if f and Path(f).exists():
                    rs = json.loads(Path(f).read_text(encoding="utf-8"))
                    ok = sum(r["ok"] and r.get("valid", True) for r in rs)
                    inv = sum(not r.get("valid", True) for r in rs)
                    kinds = {}
                    for r in rs:
                        kinds[r.get("fail_kind")] = kinds.get(r.get("fail_kind"), 0) + 1
                    got[(lab, c)] = ok
                    cells.append(f"{ok}/{len(rs)} {kinds}{' 무효 ' + str(inv) if inv else ''}")
                else:
                    cells.append("-")
            rows.append(f"| {lab} | " + " | ".join(cells) + " |")
        return rows, got

    head = ["| 모델 | " + " | ".join(CLOCKS) + " |", "|---|" + "---|" * len(CLOCKS)]
    rows, story = story_table(lambda lab, c: fresh.get(f"{lab}|{c}"))
    out += ["", "## 6. 판정 3 (도라지체육관까지 통합 실행 20판, 시계 고정 시간대별) — 10-20 고친 실행기로 다시 잼",
            "> 10-20 설계 밖 변경: headless `run_story`에서 규칙·매크로 뒤 첫 예측이 매크로 전 화면을 봤음 (판당 약 550번). "
            "예측 전 1프레임 그리기(`--fresh-obs`)를 기본으로 바꾸고 같은 인자·시드로 다시 잼. 확인 시험: `progress/HANDOFF.md` '10-20 발견'",
            *head, *rows,
            "", "### 6-옛. 10-19 측정 (지난 화면 버그, 정책 측정으로 **무효** — 참고로만)", *head,
            *story_table(lambda lab, c: runs.get(f"story|{lab}|{c}"))[0]]
    # 7) 판정
    out += ["", "## 7. 판정 (설계 기준, 시드별)"]
    for lab in m:
        if lab == "M2B":
            continue
        # 10-19 고침: 예전엔 `(실패율 or 1)`이라 실패율 0%가 1로 바뀌어 복구 ❌로 나왔음 (A_s0 E-lab 0%)
        rec = all((f := _nav(by.get(("newbark_labdoor", c, lab), []))[0]) is not None and f <= 0.2 for c in CLOCKS)
        keep = worst[lab][0] <= 0.2 and worst[lab][1] < 0.10
        integ = all(story.get((lab, c), 0) >= 16 for c in CLOCKS)
        auto = True
        if lab.startswith("A"):
            d = json.loads(Path(f"states/e26/a/s{lab[-1]}/selfrepair.json").read_text(encoding="utf-8"))
            auto = d["human_interventions"] == 0 and d.get("stages", {}).get("정책 적응", False)
        verdict = "지지" if (auto and rec and keep and integ) else "부분 지지" if (auto and rec) else "기각"
        out.append(f"- {lab}: 자동화 {'✅' if auto else '❌'} / 복구 (E-lab ≤ 20% 시간대 전부) {'✅' if rec else '❌'} / "
                   f"보존 (≤ 20%, +10%p 미만) {'✅' if keep else '❌'} (최악 {worst[lab][0]:.0%}, 증가 {worst[lab][1]:+.0%}) / "
                   f"통합 (≥ 16/20 시간대 전부) {'✅' if integ else '❌'} → **{verdict}**")
    leak = Path("runs/e26/final_leak.json")
    if leak.exists():
        lk = json.loads(leak.read_text(encoding="utf-8"))
        out += ["", f"## 8. 누출 검사 (A·B 데이터 포함): 같은 상태 " + ", ".join(f"{k} {len(v)}개" for k, v in lk["leak"].items()),
                "- E-lab과 의미 키가 같은 상태는 각 방법이 자기 실행에서 얻은 경계 상태 (사용자 결정 항목). E-violet은 0이어야 함"]
    rp = Path("runs/e26/final_report.md")
    rp.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    {"eval": eval_all, "report": report}[sys.argv[1]]()
