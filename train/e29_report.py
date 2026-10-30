"""
E29 반사실 실험 집계 → runs/e29/report.md (판단 기준은 notes/experiments/E29_CRITICAL.md, start 상태 50판)

실행: uv run python -m train.e29_report
"""

import json
from pathlib import Path

import numpy as np

LIST = Path("runs/e29/cf.json")
SEG = {"K1": "route29", "K2": "route29", "K3": "cherrygrove"}


def raw(path):
    return json.loads(Path(path.replace("_summary", "_raw")).read_text(encoding="utf-8"))


def stats(rs, k_min_dist, dist_start):
    n = len(rs)
    ok = sum(r["goal"] for r in rs)
    s = {"n": n, "ok": ok, "rate": ok / n}
    if "ref" in rs[0]:
        reach = [r for r in rs if r["ref"]["visited_over"]]
        passed = [r for r in rs if r["progress"] > dist_start - k_min_dist]
        s.update({"reach": len(reach), "pass": len(passed), "ok_after_pass": sum(r["goal"] for r in passed),
                  "kl": float(np.mean([r["ref"]["kl_mean"] for r in rs if r["ref"]["kl_mean"] is not None])),
                  "kl_over": float(np.mean([r["ref"]["kl_over_mean"] for r in rs if r["ref"]["kl_over_mean"] is not None]))
                  if any(r["ref"]["kl_over_mean"] is not None for r in rs) else None,
                  "n_override": float(np.mean([r["ref"]["n_override"] for r in rs]))})
    return s


def verdict(a, b, c, cc):
    d = c - b
    if d < 0.15:
        return "기각 (C − B < +15%p)"
    if d < 0.30:
        return "보류 (+15 ~ +30%p)"
    if cc > c - 0.20:
        return "지지 아님 (대조도 비슷하게 회복)"
    if c < a - 0.15:
        return "부분 지지 (C < A − 15%p)"
    return "**인과 지지**"


def main():
    L = json.loads(LIST.read_text(encoding="utf-8"))
    lines = ["# E29 반사실 실험 결과 (자동 생성 `train/e29_report.py`)", "",
             "> 판정 = start 상태 50판, 시계 고정. 기준: C − B ≥ +30%p, C ≥ A − 15%p, C_ctrl ≤ C − 20%p / 기각 C − B < +15%p", ""]
    lines += ["| 사례 | 시작 | A 옛 | B 새 | C 길목만 옛 | C_ctrl 대조 칸만 옛 | 판정 (start만) |", "|---|---|---|---|---|---|---|"]
    detail = ["", "## 세부 (start, B·C·C_ctrl)", "",
              "| 사례 | 조건 | 도착 | 길목 도달 | 길목 통과 | 통과 뒤 도착 | KL(옛‖새) 평균 | KL 길목 칸 | 덮어쓴 걸음 |", "|---|---|---|---|---|---|---|---|---|"]
    for k in ("K1", "K2", "K3"):
        crit = json.loads(Path(f"runs/e29/{k}.json").read_text(encoding="utf-8"))
        seg = SEG[k]
        dist_start = max(r["dist"] for r in crit["rows"])
        A, B, C, CC = (raw(L[f"A|{seg}"]), raw(L[f"{k}|B"]), raw(L[f"{k}|C"]), raw(L[f"{k}|C_ctrl"]))
        starts = sorted({r["start"] for r in A}, key=lambda s: (s != "start", s))
        for st in starts:
            sel = lambda rs, key=None: [r for r in rs if r["start"] == st and (key is None or r["policy"] == key)]
            a, b, c, cc = (stats(sel(x), crit["K_min_dist"], dist_start) for x in (A, B, C, CC))
            v = verdict(a["rate"], b["rate"], c["rate"], cc["rate"]) if st == "start" else "(참고)"
            lines.append(f"| {k} {seg} | {st} | {a['ok']}/{a['n']} | {b['ok']}/{b['n']} | {c['ok']}/{c['n']} | {cc['ok']}/{cc['n']} | {v} |")
            if st == "start":
                for name, s in (("B", b), ("C", c), ("C_ctrl", cc)):
                    kl_over = f"{s['kl_over']:.2f}" if s.get("kl_over") is not None else "-"
                    detail.append(f"| {k} | {name} | {s['ok']}/{s['n']} | {s['reach']} | {s['pass']} | {s['ok_after_pass']}/{s['pass']} | {s['kl']:.2f} | {kl_over} | {s['n_override']:.1f} |")
    detail += ["", "- 길목 도달: 길목 칸(조건 C_ctrl은 대조 칸)을 한 번이라도 밟은 판. 길목 통과: 뒤집힌 칸 중 목표에 가장 가까운 칸보다 더 가까이 간 판 (세 조건 같은 기준)",
               "- KL 길목 칸: 조건마다 그 조건의 덮어쓰기 칸에서 잰 값 (B는 K 칸)"]
    out = Path("runs/e29/report.md")
    out.write_text("\n".join(lines + detail) + "\n", encoding="utf-8")
    print("\n".join(lines + detail))


if __name__ == "__main__":
    main()
