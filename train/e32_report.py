"""E32 NIGHT 집계 (notes/experiments/E32_NIGHT.md 판단 기준 그대로)
- 길 찾기 실패 = 1 − 도착 − 전멸 (E26·E30과 같음). 구간 안 시작 상태는 합쳐서 셈 (50판 × 시작 상태 수)
- 1단계 밤 약점: 밤 실패 − 아침 실패 ≥ +10%p (+10~+20%p는 경계 표시)
- 2단계 (있으면): Δ = NN − MM, 시각 = NN − NM, 게임 동작 = NM − MM → 시각 주도 (시각 ≥ 0.5Δ) / 게임 동작 주도 (동작 ≥ 0.5Δ) / 섞임
실행: uv run python -m train.e32_report → runs/e32/report.md
"""
import json
from pathlib import Path

MODELS = ["M2B_multi", "E30_D_s0", "E30_D_s1"]
S1 = Path("runs/e32/stage1.json")
S2 = Path("runs/e32/stage2.json")
S4 = Path("runs/e32/stage4.json")
OUT = Path("runs/e32/report.md")


def raw(summary):
    return json.loads(Path(summary.replace("_summary.json", "_raw.json")).read_text(encoding="utf-8"))


def fail(rows):
    n = len(rows)
    return (n - sum(r["goal"] for r in rows) - sum(r["whiteout"] for r in rows)) / n if n else None


def by_model(summary):
    out = {}
    for r in raw(summary):
        out.setdefault(r["policy"], []).append(r)
    return out


def stage1():
    d = json.loads(S1.read_text(encoding="utf-8"))
    segs = list(dict.fromkeys(k.split("|")[0] for k in d))
    table, weak = {}, {m: [] for m in MODELS}
    for seg in segs:
        for m in MODELS:
            f = {}
            for c in ("morning", "day", "night"):
                k = f"{seg}|{c}"
                f[c] = fail(by_model(d[k]).get(m, [])) if k in d else None
            table[(seg, m)] = f
            if f["night"] is not None and f["morning"] is not None and f["night"] - f["morning"] >= 0.10:
                weak[m].append(seg)
    return segs, table, weak


def main():
    segs, table, weak = stage1()
    out = ["# E32 NIGHT 결과", "", "## 1단계: 시간대별 길 찾기 실패 (50판 × 시작 상태, 시작 상태 합침)",
           "| 구간 | " + " | ".join(f"{m} 아침/낮/밤 (Δ밤−아침)" for m in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    for seg in segs:
        cells = []
        for m in MODELS:
            f = table[(seg, m)]
            if None in f.values():
                cells.append("-")
                continue
            dl = f["night"] - f["morning"]
            mark = " **약점**" if dl >= 0.20 else " ⚠경계" if dl >= 0.10 else ""
            cells.append(f"{f['morning']:.0%}/{f['day']:.0%}/{f['night']:.0%} ({dl:+.0%}){mark}")
        out.append(f"| {seg} | " + " | ".join(cells) + " |")
    out += ["", "- 약점 (Δ ≥ +10%p, ⚠ = +10~+20%p 경계 포함): " + " / ".join(f"{m}: {', '.join(weak[m]) or '없음'}" for m in MODELS)]
    if S2.exists():
        d2 = json.loads(S2.read_text(encoding="utf-8"))
        out += ["", "## 2단계: 2×2 (네 칸 모두 다시 그린 화면, 50판 × 시작 상태)",
                "| 모델 | 구간 | MM | NN | NM | MN | Δ=NN−MM | 시각=NN−NM | 동작=NM−MM | 판정 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for m in MODELS:
            for seg in weak[m]:
                f = {}
                for cell in ("MM", "NN", "NM", "MN"):
                    k = f"{seg}|{cell}"
                    f[cell] = fail(by_model(d2[k]).get(m, [])) if k in d2 else None
                if None in f.values():
                    out.append(f"| {m} | {seg} | - | - | - | - | - | - | - | 측정 없음 |")
                    continue
                dl, vis, dyn = f["NN"] - f["MM"], f["NN"] - f["NM"], f["NM"] - f["MM"]
                v = "Δ ≤ 0 (다시 그린 조건에선 밤 약점 없음)" if dl <= 0 else "시각 주도" if vis >= 0.5 * dl else "게임 동작 주도" if dyn >= 0.5 * dl else "섞임"
                out.append(f"| {m} | {seg} | {f['MM']:.0%} | {f['NN']:.0%} | {f['NM']:.0%} | {f['MN']:.0%} | {dl:+.0%} | {vis:+.0%} | {dyn:+.0%} | {v} |")
    if S4.exists():
        d4 = json.loads(S4.read_text(encoding="utf-8"))
        kinds = ["policy", "transition", "rule", "detected", "invalid_scope", "interrupted"]
        out += ["", "## 4단계: 통합 실행 (고친 실행기, 20판, E26 판정 3과 같은 인자·시드)",
                "- 실패 종류: policy (걸음 상한까지 목표 못 감) / transition (계획 없음·구간 전환) / rule (규칙·매크로 멈춤). 전멸 판 = 전멸 1번 이상 (성공 판 포함)",
                "| 모델 | 시간대 | 성공 | " + " | ".join(kinds) + " | 전멸 판 | 성공 걸음 중앙값 |", "|---|---|---|" + "---|" * (len(kinds) + 2)]
        for m in MODELS:
            for c in ("morning", "night"):
                p = d4.get(f"{m}|{c}")
                if not p:
                    out.append(f"| {m} | {c} | 측정 없음 |" + " |" * (len(kinds) + 2))
                    continue
                rows = json.loads(Path(p).read_text(encoding="utf-8"))
                ok = [r for r in rows if r.get("fail_kind") == "ok"]
                st = sorted(r["steps"] for r in ok)
                med = st[len(st) // 2] if st else "-"
                out.append(f"| {m} | {c} | {len(ok)}/{len(rows)} | " + " | ".join(str(sum(r.get("fail_kind") == k for r in rows)) for k in kinds)
                           + f" | {sum(r.get('whiteouts', 0) > 0 for r in rows)} | {med} |")
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
