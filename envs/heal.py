"""
포켓몬센터 회복 매크로 (비계): 센터 문 안쪽에서 → 카운터까지 위로 → 간호사에게 말 걸기 → 대화 끝까지

왜 매크로인가
- 회복은 "센터에 들어가서 간호사에게 말 걸고 예"라는 정해진 절차라, 도착 보상으로 RL이 배울 신호가 거의 없음
- 메뉴/이벤트 트랙의 첫 매크로. "언제 회복할지"는 상위 계획의 규칙 (새 마을 첫 도착 또는 HP 절반 미만), 나중에 RL 후보

동작 (모든 센터가 같은 구조: 문에서 위로 곧장 가면 카운터 앞, 간호사는 카운터 너머)
1. 위로 걸어서 더 안 움직일 때까지 (카운터에 닿음)
2. A를 누르고, 누를 때마다 "방향키로 움직여지나"를 세이브 상태로 시험 → 움직여지면 대화 끝
   ("회복할까요?"의 기본 선택이 '예'라 A만으로 됨)
3. 결과: 파티 HP 합이 최대가 됐는지로 확인

주의: 대화가 끝난 뒤 A를 더 누르면 간호사에게 다시 말을 검 (할아버지 집 이벤트와 같은 함정) → 매번 움직임 확인
"""

import io

from envs import memory_map as mm


# 창으로 볼 때(라이브)는 매크로 입력도 화면에 그림. 기본은 안 그림 (학습/평가 속도)
#   "눌러보고 되돌리기" 확인(can_move, is_name_screen)은 항상 안 그림 → 그려도 되돌아가서 화면이 튀기만 함
RENDER = False
# 10-23 (일관성 시험 C1): 눌러보고 되돌리는 시험(can_move, is_name_screen)이 쓴 프레임 합
#   PyBoy frame_count는 load_state로 되돌려도 안 줄어듦 → frame_count만 보면 시험만 한 걸음도 '게임이 진행됨'으로 보임
#   real_frames(pb) = frame_count − 이 값 = 실제로 게임이 진행된 프레임 (run_story 지난 화면 판별용)
PROBE_FRAMES = 0


def real_frames(pb):
    return pb.frame_count - PROBE_FRAMES


def _probe_done(pb, f0):
    global PROBE_FRAMES
    PROBE_FRAMES += pb.frame_count - f0


def _press(pb, b, wait=40, render=None):
    r = RENDER if render is None else render
    pb.button_press(b)
    pb.tick(4 if b in ("a", "b", "start") else 16, r)
    pb.button_release(b)
    pb.tick(wait, r)


def _snap(pb):
    buf = io.BytesIO()
    pb.save_state(buf)
    return buf.getvalue()


def can_move(pb):
    """세이브 상태로 네 방향을 시험해서 움직여지는 방향이 있으면 True (상태는 되돌림)"""
    s = _snap(pb)
    f0 = pb.frame_count
    p0 = mm.read_position(pb)
    ok = False
    for d in ("down", "left", "right", "up"):
        pb.load_state(io.BytesIO(s))
        _press(pb, d, render=False)
        if mm.read_position(pb)[:2] == p0[:2] and mm.read_position(pb) != p0:
            ok = True
            break
    pb.load_state(io.BytesIO(s))
    _probe_done(pb, f0)
    return ok


def party_max_hp(pb):
    total = 0
    for i in range(min(mm.read_party_count(pb), 6)):
        a = mm.PARTY_MON1_MAXHP + i * mm.PARTY_MON_SIZE
        total += mm.read_byte(pb, a) << 8 | mm.read_byte(pb, a + 1)
    return total


def heal_at_pc(pb, max_presses=120):
    """센터 안(문 근처)에서 호출. 반환: {"healed", "presses", "hp_before", "hp_after", "max_hp"}"""
    hp0 = mm.read_party_hp_total(pb)
    for _ in range(12):   # 카운터까지 위로
        before = mm.read_position(pb)
        _press(pb, "up")
        if mm.read_position(pb) == before:
            break
    presses = 0
    for presses in range(1, max_presses + 1):
        _press(pb, "a", wait=60)
        if presses >= 3 and can_move(pb):
            break
    hp1, mx = mm.read_party_hp_total(pb), party_max_hp(pb)
    return {"healed": hp1 == mx, "presses": presses, "hp_before": hp0, "hp_after": hp1, "max_hp": mx}
