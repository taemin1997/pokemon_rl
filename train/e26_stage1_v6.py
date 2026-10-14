"""E26 1단계 V6: 통합 실행의 범위/실패 분류를 가짜 결과로 빠르게 검사한다."""

import json
import types
from pathlib import Path

from pyboy import PyBoy

import train.run_story as rs


def main():
    pb = PyBoy("roms/gold_rom.gb", window="null")
    with open("states/start.state", "rb") as state:
        pb.load_state(state)
    pb.tick(1, False)
    env = types.SimpleNamespace(pyboy=pb)
    args = types.SimpleNamespace(target_list=[(26, 10), (24, 5), (10, 7)])
    cases = []

    rs.REC.clear()
    rs.REC.update({"segments": ["newbark_lab"], "targets_reached": [[26, 10], [24, 5]], "stop_reason": None})
    a = rs._finalize({"ok": True}, args, env)
    cases.append(("연구소에서 끝난 성공", a["valid"] is False and a["fail_kind"] == "invalid_scope"))

    rs.REC.clear()
    rs.REC.update({"segments": [], "targets_reached": [[26, 10], [24, 5], [10, 7]], "stop_reason": None})
    b = rs._finalize({"ok": True}, args, env)
    cases.append(("목표 다 지난 성공", b["valid"] is True and b["fail_kind"] == "ok"))

    rs.REC.clear()
    rs.REC.update({"segments": [], "targets_reached": [[26, 10], [24, 5]], "stop_reason": "rule:lab_exit_cap"})
    c = rs._finalize({"ok": False}, args, env)
    cases.append(("규칙 상한 실패", c["valid"] is True and c["fail_kind"] == "rule"))

    rs.REC.clear()
    rs.REC.update({"segments": [], "targets_reached": [], "stop_reason": "transition:no_plan"})
    d = rs._finalize({"ok": False}, args, env)
    cases.append(("계획 없음", d["fail_kind"] == "transition"))
    pb.stop()

    result = {"pass": all(ok for _, ok in cases),
              "detail": ", ".join(f"{name} {'✅' if ok else '❌'}" for name, ok in cases)}
    path = Path("runs/e26/stage1_v6.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
