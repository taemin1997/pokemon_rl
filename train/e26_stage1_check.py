"""
E26 1단계 재현성 검증 판정 (scripts/e26_stage1.sh가 만든 결과를 읽어서 runs/e26/stage1_report.md로)

검증 항목 (notes/experiments/E26_SELFREPAIR.md 1단계)
- V1 결정성: 시계 고정 + 같은 시드로 같은 진단을 두 번 → 판별 결과가 전부 같아야 함
- V2 과거 재현: 새 기록 코드로 10-12 17:1x 시계 고정 M2B route30 결과를 다시 → 판별로 같아야 함 (코드 변경이 결과를 안 바꿈)
- V3 시간대: morning/day/night 각각 게임 시각·시간대가 조건대로 기록되고, 같은 조건 두 번은 같음
- V4 실제 시계: wall은 게임 시각이 컴퓨터 시각을 따라감 (고정이 아님을 대조로 확인)
- V5 통합 실행 기록: 지나간 구간, 도착 목표, 종료 위치, 실패 종류, 유효성이 남고, 같은 조건 두 번이 같음
- V6 무효 판정: "성공인데 의도한 목표를 다 안 지남"을 무효로 표시하나 (가짜 결과로 시험)

실행: uv run python -m train.e26_stage1_check <결과 목록 json>
"""

import json
import sys
from pathlib import Path

KEYS = ["start", "goal", "goal_step", "left", "progress", "battles", "exp", "whiteout", "n_move", "n_blocked", "n_button",
        "end_pos", "seed"]


def raw_of(summary_path, policy="M2B_multi"):
    p = Path(str(summary_path).replace("_summary.json", "_raw.json"))
    return [r for r in json.loads(p.read_text(encoding="utf-8-sig")) if r["policy"].startswith(policy)]


def meta_of(summary_path):
    p = Path(str(summary_path).replace("_summary.json", "_meta.json"))
    return json.loads(p.read_text(encoding="utf-8-sig")) if p.exists() else None


def same(a, b, keys=KEYS):
    a = sorted(a, key=lambda r: (r["start"], r.get("seed", 0)))
    b = sorted(b, key=lambda r: (r["start"], r.get("seed", 0)))
    if len(a) != len(b):
        return 0, max(len(a), len(b))
    n = sum(all(x.get(k) == y.get(k) for k in keys if k in x and k in y) for x, y in zip(a, b))
    return n, len(a)


def rate(rs):
    g = sum(r["goal"] for r in rs) / len(rs)
    w = sum(r["whiteout"] for r in rs) / len(rs)
    path_fail = max(0.0, 1 - g - w)
    return f"도착 {g:.0%} / 길 찾기 실패 {path_fail:.0%} / 전멸 {w:.0%} / 전투 {sum(r['battles'] for r in rs) / len(rs):.1f}"


def main():
    runs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
    out = ["# E26 1단계 재현성 검증 결과", "", f"> 자동 생성 (`train/e26_stage1_check.py`). 입력 `{sys.argv[1]}`", ""]
    verdict = {}

    # V1
    a, b = raw_of(runs["V1a"]), raw_of(runs["V1b"])
    n, t = same(a, b)
    distinct1 = Path(runs["V1a"]).resolve() != Path(runs["V1b"]).resolve()
    verdict["V1 결정성 (시계 고정 같은 진단 2번)"] = n == t and distinct1
    out += ["## V1 결정성", f"- route30, M2B, lock, 25판 두 번: 판별로 같은 판 **{n}/{t}**, 별도 파일 = {distinct1}", f"- 1번: {rate(a)}", f"- 2번: {rate(b)}", ""]

    # V2
    old = runs.get("V2_old")
    if old and Path(old).exists():
        o = raw_of(old)
        n2, t2 = same(o, a, [k for k in KEYS if k != "seed"])
        verdict["V2 과거 재현 (10-12 17:1x 시계 고정 결과)"] = n2 == t2
        out += ["## V2 과거 재현", f"- 예전 결과 `{old}` vs V1 1번: 판별로 같은 판 **{n2}/{t2}** (시드 0~24 순서 대응)",
                f"- 예전: {rate(o)}", f"- 지금: {rate(a)}", ""]
    else:
        out += ["## V2 과거 재현", "- 예전 결과 파일 없음 → 건너뜀", ""]

    # V3
    out += ["## V3 시간대 (morning / day / night)", "| 구간 | 조건 | 기록된 게임 시각 | 시간대 | 결과 | 두 번 같은 판 |", "|---|---|---|---|---|---|"]
    ok3 = True
    expect = {"morning": ("7:", "아침"), "day": ("14:", "낮"), "night": ("23:", "밤")}
    for seg in ("route30", "route29"):
        for c in ("morning", "day", "night"):
            k = f"V3_{seg}_{c}"
            if k not in runs:
                continue
            r1 = raw_of(runs[k])
            r2 = raw_of(runs[k + "_2"]) if k + "_2" in runs else None
            times = sorted({r["game_time"] for r in r1})
            tods = sorted({r["time_of_day"] for r in r1})
            good = all(t.startswith(expect[c][0]) for t in times) and tods == [expect[c][1]]
            rep = same(r1, r2) if r2 else None
            if rep:
                distinct = Path(runs[k]).resolve() != Path(runs[k + "_2"]).resolve()
                good = good and rep[0] == rep[1] and distinct
            else:
                distinct = None
            ok3 = ok3 and good
            repeat = f"{rep[0]}/{rep[1]} (별도 파일 {distinct})" if rep else "-"
            out.append(f"| {seg} | {c} | {', '.join(times)} | {', '.join(tods)} | {rate(r1)} | {repeat} |")
    verdict["V3 시간대 조건 (시각·시간대 기록 + 반복 일치)"] = ok3
    out.append("")

    # V4
    w = raw_of(runs["V4_wall"])
    wm = meta_of(runs["V4_wall"])
    times = sorted({r["game_time"] for r in w})
    verdict["V4 실제 시계 대조 (고정 아님)"] = len(times) > 1 or not times[0].startswith("8:21")
    out += ["## V4 실제 시계 (대조)", f"- wall 기록 게임 시각: {', '.join(times[:5])}{' …' if len(times) > 5 else ''} (서로 다르거나 고정 값 8:21:15와 다르면 정상)",
            f"- 기록 시각 {wm['written'] if wm else '-'}", ""]

    # V5
    s1 = json.loads(Path(runs["V5a"]).read_text(encoding="utf-8-sig"))
    s2 = json.loads(Path(runs["V5b"]).read_text(encoding="utf-8-sig"))
    keys5 = ["ok", "steps", "segments", "targets_reached", "end_pos", "fail_kind", "valid", "game_time"]
    n5 = sum(all(x.get(k) == y.get(k) for k in keys5) for x, y in zip(s1, s2))
    has = all(all(k in x for k in keys5) for x in s1)
    distinct5 = Path(runs["V5a"]).resolve() != Path(runs["V5b"]).resolve()
    verdict["V5 통합 실행 기록 + 반복 일치"] = has and n5 == len(s1) == len(s2) and distinct5
    out += ["## V5 통합 실행 기록", f"- 시계 고정, 도라지체육관 목표, {len(s1)}판 두 번: 같은 판 **{n5}/{len(s1)}**, 기록 항목 다 있음 = {has}, 별도 파일 = {distinct5}"]
    for i, x in enumerate(s1):
        out.append(f"  - 판 {i + 1}: {'성공' if x['ok'] else '실패'}, 유효 {x['valid']}, 종류 {x['fail_kind']}, 걸음 {x['steps']}, "
                   f"종료 {x['end_pos']}, 게임 {x['game_time']} {x['time_of_day']}, 구간 {' → '.join(x['segments'])}")
    out.append("")

    # V6
    v6 = json.loads(Path(runs["V6"]).read_text(encoding="utf-8-sig"))
    verdict["V6 무효 판정 (가짜 결과)"] = v6["pass"]
    out += ["## V6 무효 판정", f"- {v6['detail']}", ""]

    out = out[:4] + ["## 요약", "| 항목 | 통과 |", "|---|---|"] + [f"| {k} | {'✅' if v else '❌'} |" for k, v in verdict.items()] + [""] + out[4:]
    rp = Path("runs/e26/stage1_report.md")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[:4 + 3 + len(verdict)]))
    print("보고서:", rp)


if __name__ == "__main__":
    main()
