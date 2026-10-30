"""
메뉴 트랙 (비계): 이벤트 대화 + 이름 입력 화면을 매크로로 넘김

왜 필요한가
- 연구소 이벤트 ④(경찰 대화 뒤 라이벌 이름 입력)에서 A만 누르면 이름 칸에 글자만 계속 들어가고 안 끝남
  START를 섞으면 이름 화면이 아닐 때 메뉴/도감이 열려서 엉킴 (10-08 실패)
- 그래서 "지금 이름 입력 화면인가"를 메모리로 확인하고, 맞을 때만 커서를 '결정'으로 옮겨 누름

이름 입력 화면 (10-09, 메모리 비교로 찾음)
- 커서 열 NAME_CURSOR_X = C538, 줄 NAME_CURSOR_Y = C539 (0~4, 아래로 내리면 5에서 0으로 돌아감)
- 맨 아래 줄(4)이 '정정 / 결정'. 줄 4에서 오른쪽 한 번 → '결정'
- 판별: 세이브 상태에서 아래를 한 번 눌러 커서 줄 값이 (y + 1) % 5로 바뀌면 이름 화면 (대화 중엔 아래가 아무것도 안 바꿈)
  고정 플래그 값은 WRAM에서 못 찾음 → can_move처럼 "눌러보고 되돌리기"로 판별
- 이름은 A를 누른 만큼 'ㄱ'이 들어간 채로 결정됨 (예: ㄱㄱㄱㄱㄱ). 이름은 진행에 영향 없음

실행 (확인용): uv run python -m envs.menu  → probe.state에서 이벤트 끝까지
"""

import io

from envs import memory_map as mm
from envs.heal import _press, _probe_done, _snap, can_move

NAME_CURSOR_X = 0xC538
NAME_CURSOR_Y = 0xC539
NAME_ROWS = 5            # 줄 0~4, 4 = 정정/결정


def is_name_screen(pb):
    """아래를 한 번 눌러보고 커서 줄이 한 칸 내려가면 이름 화면 (상태는 되돌림)"""
    s = _snap(pb)
    f0 = pb.frame_count
    y0 = mm.read_byte(pb, NAME_CURSOR_Y)
    _press(pb, "down", wait=30, render=False)
    y1 = mm.read_byte(pb, NAME_CURSOR_Y)
    pb.load_state(io.BytesIO(s))
    _probe_done(pb, f0)   # 시험 프레임은 진행으로 안 셈 (heal.real_frames, 10-23 C1)
    return y0 < NAME_ROWS and y1 == (y0 + 1) % NAME_ROWS


def confirm_name(pb):
    """이름 화면에서 커서를 '결정'으로 옮겨 누름"""
    for _ in range(NAME_ROWS):
        if mm.read_byte(pb, NAME_CURSOR_Y) == NAME_ROWS - 1:
            break
        _press(pb, "down", wait=30)
    _press(pb, "right", wait=30)
    _press(pb, "a", wait=60)


BATTLE_CAP = 20000      # 이벤트 하나에서 전투에 쓸 최대 프레임 (트레이너 전투 약 6,000~8,000)


_dumps = 0


def _dump_stuck(pb, why):
    """안 끝나는 상황을 나중에 보려고 상태 + 화면 저장 (runs/stuck/). 프로세스당 3번까지 (10-11: 같은 상황이 750번 쌓였음)"""
    global _dumps
    _dumps += 1
    if _dumps > 3:
        return
    import os
    import time
    from pathlib import Path
    out = Path("runs/stuck")
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{why}_{time.strftime('%m%d_%H%M%S')}_{os.getpid()}"
    buf = io.BytesIO()
    pb.save_state(buf)
    stem.with_suffix(".state").write_bytes(buf.getvalue())
    try:
        pb.tick(1, True)
        pb.screen.image.save(stem.with_suffix(".png"))
    except Exception:
        pass


def run_event(pb, max_presses=300):
    """자동 이벤트를 끝까지: A를 누르고 매번 움직일 수 있나 확인, 이름 화면이면 결정
    반환: {"done", "presses", "names", "battle"(0 없음 / 1 야생 / 2 트레이너)}"""
    names, battle, battle_frames = 0, 0, 0
    for presses in range(1, max_presses + 1):
        if mm.in_battle(pb):
            # 전투 중엔 확인(이름 화면/움직임) 없이 A 연타만 (GoldEnv 자동 전투와 같은 타이밍)
            #   10-11 버그: 늦게 시작한 야생 전투를 여기서 받으면 A마다 움직임 확인(세이브 상태 4번)을 해서
            #   M2B 학습이 초당 9스텝까지 느려졌음
            #   10-11 버그 2: A 연타로 안 끝나는 전투 상태가 있어 학습 프로세스 하나가 1시간 반 멈춤
            #   → 이벤트 하나에서 전투 프레임 합을 BATTLE_CAP으로 제한, 넘으면 상태를 저장하고 포기(에이전트에게 넘김)
            #   10-12: 1번 기술 PP 0 → auto_battle이 PP 남은 기술로 넘김 (1번 PP가 있으면 예전과 같은 A 연타)
            from envs.battle import auto_battle
            battle = max(battle, mm.read_byte(pb, mm.BATTLE_MODE))
            battle_frames += auto_battle(pb, wait=20, max_frames=BATTLE_CAP - battle_frames)["frames"]
            if mm.in_battle(pb):
                _dump_stuck(pb, "battle")
                return {"done": False, "presses": presses, "names": names, "battle": battle, "stuck": True}
            continue
        if is_name_screen(pb):
            confirm_name(pb)
            names += 1
        else:
            _press(pb, "a", wait=60)
        battle = max(battle, mm.read_byte(pb, mm.BATTLE_MODE))   # 도중에 전투가 있었나 (1 야생, 2 트레이너)
        if presses >= 3 and mm.read_byte(pb, mm.SCRIPT_LOCK) == 0 and can_move(pb):
            return {"done": True, "presses": presses, "names": names, "battle": battle}
    return {"done": False, "presses": max_presses, "names": names, "battle": battle}


if __name__ == "__main__":
    from pyboy import PyBoy
    pb = PyBoy("roms/gold_rom.gb", window="null")
    pb.load_state(open("states/segments/newbark_lab/probe.state", "rb"))
    r = run_event(pb)
    print(r, "위치", mm.read_position(pb))
