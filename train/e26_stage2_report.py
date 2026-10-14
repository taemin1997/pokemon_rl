"""
E26 2단계 기준선 보고서 (scripts/e26_stage2.sh 결과 → runs/e26/stage2_report.md)

판 분류 (설계 보강 4)
- 성공: 목표 맵 도착
- 정책 실패: 연구소(또는 다른 맵) 재진입 = 이탈 / 시간초과 (평가 구간은 규칙 없이 정책만 걸음 → 이탈·시간초과는 정책 행동 결과)
- 전멸: 지표로만 (길 찾기 실패에서 제외, 원칙)
- 규칙 개입 실패: 자동 이벤트/전투 매크로가 상한에 걸림 → runs/stuck에 새 덤프로 드러남 (판 단위로는 못 붙여서 개수만)
- 무효 (실험 무효, H-A 기각 아님): 실행 기록의 시계 조건·게임 시각·시작 상태 sha1이 의도와 다르면 그 실행 전체 무효
전제 조건 (설계 2단계): E-lab 길 찾기 실패율 ≥ 30% (시간대마다 보고, 전부 미만이면 재설계)
"""

import json
import math
import sys
from pathlib import Path

EXPECT = {"morning": ("7:", "아침"), "day": ("14:", "낮"), "night": ("23:", "밤")}
NAMES = {"newbark_labdoor": "E-lab", "violet_pc_counter": "E-violet"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def main():
    runs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
    index = json.loads(Path("states/e26/eval/index.json").read_text(encoding="utf-8"))
    sha = {v["segment"]: v["sha1"] for v in index["states"].values()}
    out = ["# E26 2단계 기준선 (M2B_multi, 학습·모델·규칙 변경 없음)", "",
           f"> 자동 생성 (`train/e26_stage2_report.py`). 평가 상태 `states/e26/eval/index.json`. 판 수 50, 신뢰구간 95% (Wilson)", "",
           "| 상태 | 시간대 | 유효 | 성공 | 길 찾기 실패 (95% 구간) | 연구소/다른 맵 재진입 | 시간초과 | 전멸 | 게임 시각 |",
           "|---|---|---|---|---|---|---|---|---|"]
    pre = []
    for key, summ in runs.items():
        seg, clock = key.rsplit("_", 1)
        raw = json.loads(Path(summ.replace("_summary.json", "_raw.json")).read_text(encoding="utf-8"))
        meta = json.loads(Path(summ.replace("_summary.json", "_meta.json")).read_text(encoding="utf-8"))
        times = meta.get("game_times", [])
        why = []
        if meta.get("clock") != clock:
            why.append(f"시계 {meta.get('clock')}")
        if not all(t.startswith(EXPECT[clock][0]) for t in times) or meta.get("times_of_day") != [EXPECT[clock][1]]:
            why.append(f"게임 시각 {times} {meta.get('times_of_day')}")
        if meta["starts"]["start"]["sha1"] != sha[seg]:
            why.append("시작 상태 sha1 다름")
        n = len(raw)
        goal = sum(r["goal"] for r in raw)
        wo = sum(r["whiteout"] for r in raw)
        left = sum(r["left"] and not r["whiteout"] for r in raw)
        timeout = n - goal - wo - left
        nav = left + timeout
        lo, hi = wilson(nav, n)
        ends = {}
        for r in raw:
            if r["left"] and not r["whiteout"]:
                m = tuple(r["end_pos"][:2])
                ends[m] = ends.get(m, 0) + 1
        out.append(f"| {NAMES.get(seg, seg)} | {clock} | {'✅' if not why else '❌ ' + ', '.join(why)} | {goal}/{n} | "
                   f"**{nav / n:.0%}** ({lo:.0%}~{hi:.0%}) | {left} {dict(ends) if ends else ''} | {timeout} | {wo} | {', '.join(times)} |")
        if seg == "newbark_labdoor" and not why:
            pre.append((clock, nav / n))
    stuck = sorted(p.name for p in Path("runs/stuck").glob("*.state")) if Path("runs/stuck").exists() else []
    out += ["", f"- 규칙 개입 실패 단서: `runs/stuck` 덤프 {len(stuck)}개 (실행 중 새로 생긴 것 포함, 이름에 시각)",
            "", "## 전제 조건 (E-lab 길 찾기 실패율 ≥ 30%)"]
    for c, f in pre:
        out.append(f"- {c}: {f:.0%} → {'충족' if f >= 0.3 else '미달'}")
    if pre:
        out.append(f"- 판정: {'충족 (E26 진행 가능)' if any(f >= 0.3 for _, f in pre) else '전부 미달 → 실패가 상태 하나로 재현 안 됨, 재설계'}")
    rp = Path("runs/e26/stage2_report.md")
    rp.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print("보고서:", rp)


if __name__ == "__main__":
    main()
