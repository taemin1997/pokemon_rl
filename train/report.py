"""
29번도로 일반화 리포트: 모델마다 같은 지표를 한 표로 (진단 여러 개를 차례로 돌리고 결과를 모음)

왜 필요한가
- "집 → 무궁 96%"만 보면 잘 되는 것 같지만, 반대 방향 0%, 벽 고집 45%면 해석이 완전히 달라짐 (외부 검토, 10-07)
- 지표가 진단 명령 5개에 흩어져 있어서 모델끼리 비교할 때 빠뜨리기 쉬움 → 한 명령, 한 표
- 다음 단계 목표 문장: "29번도로 어디서 시작해도, 지정된 출구를 향해 스스로 복구하며 이동한다"를 재는 표

모으는 지표 (괄호: 출처 명령)
- 경로 위 도착 start / S2 / S3 (train.diagnose route29)
- 반대 방향 도착 (train.diagnose route29_rev): 목표가 연두마을일 때
- 경로 밖 도착 3~5 / 6~9 / 10+칸, 빼둔 지역, 혼동 통로 3칸 (train.offpath eval)
- 벽 고집: 막힌 직후 같은 버튼, 최장 연속 (train.offpath stuck)
- 목표 바꿔치기: 목표만 바꾸면 행동이 바뀌는 비율, 반대 목표에서 맞는 방향 비율 (train.goal_probe)
- (선택 --perturb) 교란 복구 (train.offpath perturb)

모델 여러 개를 주면 각 진단에 같이 넣음 (같은 실행 = 공정한 비교). 결과는 표 + runs/diagnose/report_*.md / .json

실행
- uv run python -m train.report models/GOAL1_route29/final.zip models/LAT1_route29/final.zip
- 학습과 같이 돌릴 땐: --workers 2  (전부 돌리면 4프로세스 기준 약 40~50분)
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from train.diagnose import policy_label

# 혼동 통로 3칸 (LAT1 두 시드 모두 0~1/4. 목표 반대 방향 막다른 끝에서 벽 밀기, notes/04 LAT1)
CONFUSED = [(24, 3, 16, 10), (24, 3, 20, 11), (24, 3, 15, 12)]


def run(cmd):
    """진단 명령 실행 → 저장된 summary 경로 (출력의 '저장: …' 줄)"""
    print("$ " + " ".join(cmd[2:]), flush=True)
    # 자식 프로세스 출력을 utf-8로 (윈도우 기본 cp949면 '저장:' 줄을 못 읽음)
    import os
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    m = re.findall(r"저장: (\S+_summary\.json)", out.stdout)
    if not m:
        print(out.stdout[-2000:], out.stderr[-2000:])
        raise RuntimeError("진단 결과 파일을 못 찾음")
    return Path(m[-1])


def pct(x):
    return "-" if x is None else f"{x:.0%}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--perturb", action="store_true", help="교란 복구도 (약 +10분)")
    parser.add_argument("--quick", action="store_true", help="동작 확인용: 판 수를 아주 적게 (숫자는 의미 없음)")
    args = parser.parse_args()
    py = [sys.executable, "-m"]
    labels = [policy_label(m) for m in args.models]
    q_diag = ["--runs", "2"] if args.quick else []
    q_off = ["--per-band", "2", "--runs", "1"] if args.quick else []
    q_probe = ["--per-band", "3"] if args.quick else []
    R = {lab: {} for lab in labels}

    # 1) 경로 위 / 반대 방향
    for seg, key in (("route29", "on"), ("route29_rev", "rev")):
        rows = json.loads(run(py + ["train.diagnose", seg, *args.models, "--workers", str(args.workers), *q_diag]).read_text())
        for r in rows:
            R[r["policy"]][f"{key}_{r['start']}"] = r["goal_rate"]
    # 2) 경로 밖 (구간별, 빼둔 지역, 혼동 통로)
    p = run(py + ["train.offpath", "eval", "route29", *args.models, "--workers", str(args.workers), *q_off])
    for r in json.loads(p.read_text()):
        if "band" in r:
            R[r["policy"]][f"off_{r['band']}"] = r["goal_off"]
        elif r.get("region") == "빼둔 지역":
            R[r["policy"]]["holdout"] = r["goal_off"]
    raw = json.loads(Path(str(p).replace("_summary.json", "_raw.json")).read_text())
    for lab in labels:
        rs = [x for x in raw if x["policy"] == lab and x["role"] == "off" and tuple(x["pos"]) in CONFUSED]
        R[lab]["confused"] = sum(x["goal"] for x in rs) / len(rs) if rs else None
    # 3) 벽 고집
    for r in json.loads(run(py + ["train.offpath", "stuck", "route29", *args.models,
                                  "--workers", str(args.workers), *q_off[:2]]).read_text()):
        R[r["policy"]].update({"repeat_after_block": r["repeat_after_block"], "longest_max": r["longest_max"],
                               "longest_p50": r["longest_p50"]})
    # 4) 목표 바꿔치기 (모델마다)
    for m, lab in zip(args.models, labels):
        s = json.loads(run(py + ["train.goal_probe", m, *q_probe]).read_text())
        R[lab].update({"flipped": s["all"]["flipped"], "rev_top_correct": s["all"]["route29_rev"],
                       "fwd_top_correct": s["all"]["route29"]})
    # 5) 교란 (선택)
    if args.perturb:
        for r in json.loads(run(py + ["train.offpath", "perturb", "route29", *args.models,
                                      "--workers", str(args.workers)]).read_text()):
            if r["k"] == 6:
                R[r["policy"]].setdefault("perturb6", []).append(r["goal"])
        for lab in labels:
            v = R[lab].pop("perturb6", None)
            R[lab]["perturb6"] = sum(v) / len(v) if v else None

    cols = [
        ("경로 위 start/S2/S3", lambda r: f"{pct(r.get('on_start'))} / {pct(r.get('on_S2_deadend'))} / {pct(r.get('on_S3_fork'))}"),
        ("반대 방향", lambda r: pct(r.get("rev_start"))),
        ("경로 밖 3~5/6~9/10+", lambda r: f"{pct(r.get('off_3-5'))} / {pct(r.get('off_6-9'))} / {pct(r.get('off_10+'))}"),
        ("빼둔 지역", lambda r: pct(r.get("holdout"))),
        ("혼동 통로 3칸", lambda r: pct(r.get("confused"))),
        ("막힌 직후 같은 버튼", lambda r: pct(r.get("repeat_after_block"))),
        ("최장 연속 막힘", lambda r: f"{r.get('longest_p50', 0):.0f} / {r.get('longest_max', '-')}"),
        ("목표 바꾸면 행동 바뀜", lambda r: pct(r.get("flipped"))),
        ("반대 목표 맞는 방향", lambda r: pct(r.get("rev_top_correct"))),
    ]
    if args.perturb:
        cols.append(("교란 k=6 복구", lambda r: pct(r.get("perturb6"))))
    lines = ["| 지표 | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for name, f in cols:
        lines.append(f"| {name} | " + " | ".join(f(R[lab]) for lab in labels) + " |")
    table = "\n".join(lines)
    print("\n" + table)
    out = Path("runs/diagnose")
    stem = f"report_route29_{time.strftime('%m%d_%H%M')}_{'+'.join(labels)[:100]}" + ("_quick" if args.quick else "")
    (out / f"{stem}.md").write_text(table + "\n", encoding="utf-8")
    (out / f"{stem}.json").write_text(json.dumps(R, ensure_ascii=False, indent=1))
    print(f"\n저장: {out / (stem + '.md')}")


if __name__ == "__main__":
    main()
