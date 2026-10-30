"""E32 NIGHT 3단계: 짝지은 판에서 처음 갈라지는 걸음 (notes/experiments/E32_NIGHT.md 3단계)
- 같은 (모델, 시작 상태, 시드)의 두 칸 판을 걸음 기록(POKE_TRACE)으로 맞대어 봄. 기본 짝 = MM vs NN (설계), 확인용 NM vs NN·MM vs NM
- 처음 갈라지는 걸음 i = 위치·전투·이벤트·행동 중 하나라도 다른 첫 걸음
  (a) 정책: 위치·전투·이벤트 같은데 행동이 다름 (같은 게임 상태에서 보는 색만 달라 행동이 바뀜)
  (b) 게임 동작: 직전까지 행동이 같았는데 위치·전투·이벤트가 다름 (조우·NPC·잠김)
  안 갈라짐: 한쪽 기록이 끝날 때까지 같음
- 짝마다 (a)의 뜻이 다름 (갈라지기 전까지 같은 시드·같은 행동 → 같은 칸끼리는 게임 진행이 같음):
  NM vs NN = 게임 같음, 색만 다름 → (a)는 순수 색 반응 / MM vs NM = 색 같음, 게임 시간만 다름 → (a)는 화면에 보이는 게임 차이
  (밤 NPC·스프라이트 등, 위치·전투·이벤트엔 안 잡힘) / MM vs NN = 둘이 섞임
- 확인 칸 '같은 화면 (a)': 관찰 해시가 같은데 행동이 갈린 수 (같은 시드면 0이어야 함)
- 갈라진 걸음의 두 정책 분포 KL(앞 칸 ‖ 뒤 칸)도 보고 (판정엔 안 씀)
실행: uv run python -m train.e32_diverge [MM-NN ...] → runs/e32/diverge.md
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

from train.e32_report import MODELS, S2, raw

OUT = Path("runs/e32/diverge.md")


def kl(p, q):
    return sum(a * (math.log(a + 1e-8) - math.log(b + 1e-8)) for a, b in zip(p, q))


def first_split(ta, tb):
    for i, (x, y) in enumerate(zip(ta, tb)):
        same_state = x["pos"] == y["pos"] and x["b"] == y["b"] and x["ev"] == y["ev"]
        if not same_state:
            return i, "b"
        if x["a"] != y["a"]:
            return i, "a"
    return None, None


def compare(d2, seg, ca, cb):
    ka, kb = f"{seg}|{ca}", f"{seg}|{cb}"
    if ka not in d2 or kb not in d2:
        return None
    A = {(r["policy"], r["start"], r["seed"]): r for r in raw(d2[ka])}
    B = {(r["policy"], r["start"], r["seed"]): r for r in raw(d2[kb])}
    rows = []
    for key in sorted(A.keys() & B.keys()):
        a, b = A[key], B[key]
        i, kind = first_split(a["trace"], b["trace"])
        x = a["trace"][i] if i is not None else None
        y = b["trace"][i] if i is not None else None
        rows.append({"policy": key[0], "start": key[1], "seed": key[2], "i": i, "kind": kind or "none",
                     "pos": x["pos"] if x else None, "kl": kl(x["p"], y["p"]) if kind == "a" else None,
                     "same_obs": kind == "a" and x["obs"] == y["obs"],
                     "ok_a": a["goal"], "ok_b": b["goal"]})
    return rows


def main():
    pairs = [p.split("-") for p in (sys.argv[1:] or ["MM-NN", "NM-NN", "MM-NM"])]
    d2 = json.loads(S2.read_text(encoding="utf-8"))
    segs = list(dict.fromkeys(k.split("|")[0] for k in d2))
    out = ["# E32 3단계: 처음 갈라지는 걸음", "",
           "- (a) 정책 = 같은 게임 상태에서 행동이 다름 / (b) 게임 동작 = 같은 행동 뒤 위치·전투·이벤트가 다름 / 없음 = 끝까지 같음",
           "- '앞 성공·뒤 실패' = 앞 칸은 도착, 뒤 칸은 도착 못 한 짝 (밤에서 잃은 판). 그 짝만 따로 셈", ""]
    for ca, cb in pairs:
        out += [f"## {ca} vs {cb}", "",
                "| 모델 | 구간 | 짝 | (a) 정책 | (b) 게임 동작 | 없음 | 앞 성공·뒤 실패: (a)/(b)/없음 | 갈라진 걸음 중앙값 | (a) KL 중앙값 | 같은 화면 (a) | 자주 갈라진 칸 |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for m in MODELS:
            for seg in segs:
                rows = compare(d2, seg, ca, cb)
                if rows is None:
                    continue
                rows = [r for r in rows if r["policy"] == m]
                if not rows:
                    continue
                c = Counter(r["kind"] for r in rows)
                lost = Counter(r["kind"] for r in rows if r["ok_a"] and not r["ok_b"])
                steps = sorted(r["i"] for r in rows if r["i"] is not None)
                kls = sorted(r["kl"] for r in rows if r["kl"] is not None)
                tiles = Counter(tuple(r["pos"]) for r in rows if r["pos"]).most_common(2)
                med = lambda v: v[len(v) // 2] if v else None
                ms, mk = med(steps), med(kls)
                out.append(f"| {m} | {seg} | {len(rows)} | {c['a']} | {c['b']} | {c['none']} | "
                           f"{lost['a']}/{lost['b']}/{lost['none']} | {ms if ms is not None else '-'} | "
                           f"{f'{mk:.3f}' if mk is not None else '-'} | {sum(r['same_obs'] for r in rows)} | "
                           + ", ".join(f"{list(t)}×{n}" for t, n in tiles) + " |")
        out.append("")
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
