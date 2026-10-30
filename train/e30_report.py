"""E30 PRESERVE 집계 (notes/experiments/E30_PRESERVE.md 판단 기준 그대로)
- 길 찾기 실패율 = 1 - 도착 - 전멸 (e26_final._nav와 같음)
- 새 구간 유지: E-lab 실패 ≤ 20% 그리고 A(같은 시드) 대비 +10%p 이내
- 보존: 16구간 모두 M2B 대비 실패 증가 < +10%p (구간 안 시작점 중 최악)
- 경계 (+5 ~ +15%p) 구간은 50판 재측정 대상으로 표시만 함
실행: uv run python -m train.e30_report  → runs/e30/report.md
"""
import json
from pathlib import Path

SEGS = ["house2f", "newbark", "route29", "cherrygrove", "cherrygrove_pc", "cherrygrove_pc_exit", "route30", "route30_rev",
        "cherrygrove_rev", "route29_rev", "newbark_lab", "route30_north", "route31", "violet_gym", "violet_pc", "violet_pc_exit"]
MODELS = ["E30_A_s0", "E30_A_s1", "E30_B_s0", "E30_B_s1", "E30_C_s0", "E30_C_s1", "E30_D_s0", "E30_D_s1"]
OUT = Path("runs/e30/report.md")


def latest(seg):
    fs = sorted(Path("runs/diagnose").glob(f"{seg}_1022_*E30_A_s0*E30_D_s1_summary.json"))   # 본 평가 (9모델) 파일만
    return fs[-1] if fs else None


def fail(r):
    return 1 - r["goal_rate"] - r["whiteout_rate"]


def load(seg):
    f = latest(seg)
    by = {}
    for r in json.loads(f.read_text(encoding="utf-8")):
        by.setdefault(r["policy"], {})[r["start"]] = r
    return f, by


def main():
    out = ["# E30 PRESERVE 결과 (10-22 평가, 시계 lock)", ""]
    f, el = load("newbark_labdoor")
    out += [f"원시: `{f}`", "", "## 1. 새 구간 유지 (E-lab 50판, 첫 화면 1프레임)", "| 모델 | 실패 | A 대비 | 유지 |", "|---|---|---|---|"]
    keep, elab = {}, {}
    for m in ["M2B_multi"] + MODELS:
        elab[m] = max(fail(r) for r in el[m].values())
    for m in ["M2B_multi"] + MODELS:
        if m == "M2B_multi":
            out.append(f"| {m} | {elab[m]:.0%} | - | - |")
            continue
        a = elab[f"E30_A_s{m[-1]}"]
        keep[m] = elab[m] <= 0.20 and elab[m] - a <= 0.10
        out.append(f"| {m} | {elab[m]:.0%} | {elab[m] - a:+.0%} | {'✅' if keep[m] else '❌'} |")
    out += ["", "## 2. 보존 (16구간 25판, M2B 대비 실패 증가, 구간 안 시작점 중 최악. ⚠ = 경계 +5~+15%p → 50판 재측정 대상)",
            "| 구간 | " + " | ".join(m.replace("E30_", "") for m in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    worst = {m: 0.0 for m in MODELS}
    border = {m: [] for m in MODELS}
    segd = {m: {} for m in MODELS}
    for seg in SEGS:
        f, by = load(seg)
        cells = []
        for m in MODELS:
            d = max(fail(by[m][s]) - fail(by["M2B_multi"][s]) for s in by[m])
            segd[m][seg] = d
            worst[m] = max(worst[m], d)
            if 0.05 <= d <= 0.15:
                border[m].append(seg)
            cells.append(f"{d:+.0%}{' ⚠' if 0.05 <= d <= 0.15 else ''}")
        out.append(f"| {seg} | " + " | ".join(cells) + " |")
    out.append("| **최악** | " + " | ".join(f"**{worst[m]:+.0%}**" for m in MODELS) + " |")
    # 경계 구간 50판 재측정 (M2B 같이): 있으면 그 구간 값을 50판 값으로 바꿔 최악을 다시 셈
    rc = sorted(Path("runs/diagnose").glob("route29_1022_*_M2B_multi+E30_D_s0+E30_D_s1_summary.json"))
    if rc:
        by = {}
        for r in json.loads(rc[-1].read_text(encoding="utf-8")):
            by.setdefault(r["policy"], {})[r["start"]] = r
        out += ["", f"### 경계 재측정 50판: route29 (`{rc[-1]}`)", "| 모델 | 25판 | 50판 |", "|---|---|---|"]
        for m in [k for k in by if k != "M2B_multi"]:
            d = max(fail(by[m][s]) - fail(by["M2B_multi"][s]) for s in by[m])
            out.append(f"| {m} | {segd[m]['route29']:+.0%} | {d:+.0%} |")
            others = max(v for k, v in segd[m].items() if k != "route29")
            worst[m] = max(others, d)
            border[m] = [s for s in border[m] if s != "route29"]
    out += ["", "## 3. 판정 (시드별, 경계 구간은 50판 재측정 값)", "| 모델 | 새 구간 유지 | 보존 (최악 < +10%p) | 판정 | 경계 구간 |", "|---|---|---|---|---|"]
    for m in MODELS:
        p = worst[m] < 0.10
        v = "성공" if keep[m] and p else "보존 실패" if keep[m] else "학습 방해" if p else "둘 다 실패"
        out.append(f"| {m} | {'✅' if keep[m] else '❌'} | {'✅' if p else '❌'} ({worst[m]:+.0%}) | {v} | {', '.join(border[m]) or '-'} |")
    a_repro = [s for s in (0, 1) if worst[f"E30_A_s{s}"] >= 0.10]
    out += ["", f"- A 망각 재현: 시드 {a_repro or '없음'} (최악 +10%p 이상) — 재현 없으면 그 시드 비교 무효"]
    for s in (0, 1):   # D vs C: D 성공이고 보존 최악값 차이 ≤ 10%p면 "소수 상태만 지켜도 됨" 지지
        d, c = f"E30_D_s{s}", f"E30_C_s{s}"
        ok = keep[d] and worst[d] < 0.10 and abs(worst[d] - worst[c]) <= 0.10
        out.append(f"- 시드 {s} D vs C: D 최악 {worst[d]:+.0%} / C 최악 {worst[c]:+.0%} → '소수 상태만 지켜도 됨' {'지지' if ok else '지지 안 됨'}")
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
