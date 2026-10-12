"""
전투 실행 층 (전투 모듈 B1): "기술 2번 써" 같은 묶은 행동을 실제 버튼 입력으로 바꿔주는 매크로

왜 필요한가
- 전투를 버튼 단위(상하좌우/A/B)로 RL에 맡기면, 메뉴 커서 옮기기와 문구 넘기기까지 배워야 해서
  정작 중요한 판단("어떤 기술을 쓸까, 도망칠까")의 신호가 묻힘
- 그래서 행동을 판단 단위로 묶음: 기술 1~4 / 도망. 메뉴 조작은 이 매크로가 함 (비계, 명시)
- 판단 층(규칙 B2 → RL B3)은 "메인 메뉴가 열렸을 때 무엇을 고를까"만 정하면 됨

동작
- 문구/연출 중(battle_menu() == None): A로 넘김 (A 연타와 같은 타이밍)
- 메인 메뉴: 판단 함수를 불러 행동을 받음 → 커서를 그 칸으로 옮기고 A
- 기술 목록: 정해둔 기술 칸으로 커서를 옮기고 A
- 커서는 메모리(MENU_CURSOR_Y/X)를 보고 옮김 → 지난 턴 커서 위치가 남아 있어도 정확

주의
- 레벨업 후 "기술을 잊게 할까?" 예/아니오, 포획, 교체, 가방은 아직 다루지 않음 (문구로 보고 A를 누름 = A 연타와 같음)
- PP 0 (10-12 처리): PP가 0인 기술을 고르면 게임이 "기술의 남은 포인트가 없다!"를 띄우고 기술 목록(커서 그대로)으로 돌아옴
  (stuck_pp0 상태로 확인) → 판단 층은 PP 남은 기술만 고름 (decide_pp_attack: PP 남은 공격 기술 중 첫 번째)
  거절 뒤 기술 목록에 남은 경우는 '아래 눌러서 커서가 움직이나'로 진짜 목록인지 확인하고 옮김. 전부 0이면 게임이 발버둥을 씀
  A 연타 자동 전투(GoldEnv, run_event)는 1번 기술 PP가 0이 된 순간부터만 이 매크로로 넘김 (auto_battle)
- 판단 층(decide)은 battle_menu() == "main"일 때만 불림. "moves"는 기술을 고른 뒤 문구 중에도 잠깐 보일 수 있어서 판단 시점으로 안 씀
"""

from envs import memory_map as mm

ACTIONS = ["move1", "move2", "move3", "move4", "run"]
# 메인 메뉴 칸 (행, 열): 싸우다 (1,1), 가방 (1,2), 포켓몬 (2,1), 도망치다 (2,2)
MAIN_FIGHT = (1, 1)
MAIN_RUN = (2, 2)
MAX_FRAMES = 6000   # GoldEnv 자동 전투와 같은 안전장치


def _press(pb, button, hold=4, wait=20):
    pb.button_press(button)
    pb.tick(hold, False)
    pb.button_release(button)
    pb.tick(wait, False)
    return hold + wait


def _cursor(pb):
    return mm.read_byte(pb, mm.MENU_CURSOR_Y), mm.read_byte(pb, mm.MENU_CURSOR_X)


def _move_cursor(pb, target, cols):
    """커서를 target (행, 열)로. cols == 1이면 열은 무시. 최대 8번 누름. 쓴 프레임 반환"""
    frames = 0
    for _ in range(8):
        y, x = _cursor(pb)
        if y == target[0] and (cols == 1 or x == target[1]):
            break
        if y != target[0]:
            frames += _press(pb, "down" if y < target[0] else "up", wait=8)
        else:
            frames += _press(pb, "right" if x < target[1] else "left", wait=8)
    return frames


def num_moves(pb):
    """파티 1번 포켓몬의 기술 수 (기술 번호 0 = 빈 칸)"""
    return sum(mm.read_byte(pb, mm.PARTY_MON1_MOVES + i) != 0 for i in range(4))


def first_pp_slot(pb):
    """PP가 남은 기술 칸 (1~4): 공격 기술(ROM 위력 > 0) 중 첫 번째, 없으면 PP 남은 첫 기술. 전부 0이면 None (그땐 게임이 발버둥을 씀)
    공격 기술 우선인 이유 (10-12 시험): 할퀴기 PP 0 → "PP 남은 첫 기술" = 째려보기(위력 0)라 데미지 없이 전투가 안 끝났음"""
    from envs import rom_data

    pp = mm.read_mon1_pp(pb)
    with_pp = [i + 1 for i in range(num_moves(pb)) if pp[i] > 0]
    for slot in with_pp:
        if rom_data.move(mm.read_byte(pb, mm.PARTY_MON1_MOVES + slot - 1))["power"] > 0:
            return slot
    return with_pp[0] if with_pp else None


def needs_pp_switch(pb):
    """1번 기술 PP가 0인데 다른 기술엔 남음 = A 연타로는 전투가 안 끝나는 상황 (10-11 M2B 멈춤 원인)"""
    return mm.read_mon1_pp(pb)[0] == 0 and first_pp_slot(pb) is not None


def has_attack_pp(pb):
    """PP 남은 공격 기술(위력 > 0)이 있나"""
    from envs import rom_data

    pp = mm.read_mon1_pp(pb)
    return any(pp[i] > 0 and rom_data.move(mm.read_byte(pb, mm.PARTY_MON1_MOVES + i))["power"] > 0
               for i in range(num_moves(pb)))


def decide_pp_attack(pb):
    """PP 남은 공격 기술 중 첫 번째 (first_pp_slot). A 연타(항상 1번)와 가장 가까운 규칙 — 1번이 공격 기술이고 PP가 있으면 A 연타와 같음
    공격 기술 PP가 전부 0이면: 야생은 도망 (10-12 pp_low 시험: 째려보기만 남으면 시간초과 74%, 전멸 24%)
    트레이너 전투는 도망이 안 되니 남은 기술 → 전부 0이면 발버둥 (게임이 자동으로 씀)"""
    if not has_attack_pp(pb) and mm.read_byte(pb, mm.BATTLE_MODE) == 1:
        return "run"
    return f"move{first_pp_slot(pb) or 1}"


def run_battle(pb, decide, max_frames=MAX_FRAMES):
    """전투가 끝날 때까지 진행. decide(pb) → ACTIONS 중 하나 (메인 메뉴가 열릴 때마다 한 번 부름)

    반환: {"frames", "turns", "actions": [고른 행동들], "rejected": 없는 기술 칸을 고른 횟수 (1번으로 바꿔 씀),
           "pp_fix": PP 0 기술이 골라져 있어서 PP 남은 기술로 바꾼 횟수, "min_hp": 전투 중 파티 HP 합 최저 (전멸 판정용)}
    """
    frames, turns, rejected, pp_fix = 0, 0, 0, 0
    min_hp = mm.read_party_hp_total(pb)
    actions = []
    pending = None          # 메인 메뉴에서 고른 기술 칸 (기술 목록에서 쓸 것)
    prev = None
    while mm.in_battle(pb) and frames < max_frames:
        min_hp = min(min_hp, mm.read_party_hp_total(pb))
        menu = mm.battle_menu(pb)
        if menu == "main":
            if prev != "main":
                turns += 1
            action = decide(pb)
            actions.append(action)
            if action == "run":
                frames += _move_cursor(pb, MAIN_RUN, cols=2)
                pending = None
            else:
                # 없는 기술 칸이면 1번으로 (판단 층이 잘못 골라도 멈추지 않게, 횟수는 기록)
                slot = int(action[-1])
                if slot > num_moves(pb):
                    rejected += 1
                    slot = 1
                frames += _move_cursor(pb, MAIN_FIGHT, cols=2)
                pending = slot
            frames += _press(pb, "a")
        elif menu == "moves":
            y = mm.read_byte(pb, mm.MENU_CURSOR_Y)
            alt = first_pp_slot(pb)
            if pending is not None and alt is not None and mm.read_mon1_pp(pb)[pending - 1] == 0:
                # 판단 층이 PP 0 기술을 골랐음 → PP 남은 첫 기술로 (거절 문구 반복 방지)
                pp_fix += 1
                pending = alt
            if pending is None and 1 <= y <= 4 and alt is not None and mm.read_mon1_pp(pb)[y - 1] == 0:
                # 커서 칸 PP 0: 거절 뒤 진짜 기술 목록일 수도, 기술을 고른 뒤 문구 중(블록이 남고 커서가 1로 읽힘, 10-12 추적)일 수도 있음
                #   → 아래를 한 번 눌러봄. 진짜 목록이면 커서가 움직임 → PP 남은 기술로 옮겨 고름. 문구 중이면 아무 일 없음 → A로 넘김
                #   (문구 중 아래는 효과 없음. 그 사이 메인 메뉴로 바뀌었으면 다음 반복의 메인 처리가 커서를 메모리 보고 다시 맞춤)
                frames += _press(pb, "down", wait=8)
                if mm.battle_menu(pb) != "moves":
                    prev = menu
                    continue
                if mm.read_byte(pb, mm.MENU_CURSOR_Y) != y:
                    pp_fix += 1
                    pending = alt
            if pending is None:
                # 기술을 고른 뒤 문구가 나오는 동안에도 블록 값이 남아서 "moves"로 보임 (째려보기처럼 문구가 길면 수백 프레임)
                #   → 문구로 보고 A로 넘김. B는 누르지 않음 (문구 중 B가 다른 걸 취소할 수 있음)
                frames += _press(pb, "a")
            else:
                frames += _move_cursor(pb, (pending, 1), cols=1)
                frames += _press(pb, "a")
                pending = None
                # 주의 (10-07 버그): A 직후에도 메뉴 블록 값이 잠깐 남아서 battle_menu()가 "moves"를 돌려줌
                #   → 그걸 "거절됨"으로 보고 B를 누르면 기술 선택이 취소됨 (기술 1만 써도 승리 0%가 됐음)
                #   그래서 B는 누르지 않음. 다음 반복에서 "moves"가 또 보이고 pending이 없으면 그때 A로 넘김

        else:
            frames += _press(pb, "a")
        prev = menu
    return {"frames": frames, "turns": turns, "actions": actions, "rejected": rejected,
            "pp_fix": pp_fix, "min_hp": min(min_hp, mm.read_party_hp_total(pb))}


def auto_battle(pb, wait=20, max_frames=MAX_FRAMES, render=False):
    """A 연타 자동 전투 (GoldEnv, run_event 공용) + PP 0 처리
    1번 기술 PP가 남아 있는 동안은 예전과 똑같이 A 연타 (같은 타이밍 → 예전 전투 결과 그대로)
    1번 PP가 0이 되면 그 순간부터 run_battle(PP 남은 공격 기술)로 넘김 (10-11 M2B 멈춤 수정)
    반환: {"frames", "min_hp", "pp_switch": 넘겼으면 True}"""
    frames = 0
    min_hp = mm.read_party_hp_total(pb)
    while mm.in_battle(pb) and frames < max_frames:
        if needs_pp_switch(pb):
            r = run_battle(pb, decide_pp_attack, max_frames=max_frames - frames)
            return {"frames": frames + r["frames"], "min_hp": min(min_hp, r["min_hp"]), "pp_switch": True}
        pb.button_press("a")
        pb.tick(4, False)
        pb.button_release("a")
        pb.tick(wait, render)
        frames += 4 + wait
        min_hp = min(min_hp, mm.read_party_hp_total(pb))
    return {"frames": frames, "min_hp": min_hp, "pp_switch": False}


# ---------------------------------------------------------------
# 판단 층 (B2): 규칙. RL(B3)이 넘어야 할 비교 기준
# ---------------------------------------------------------------
def move_score(move_id, my_types, enemy_types):
    """기대 데미지 근사: 위력 × 명중 × 자속(1.5) × 상성. 데미지 없는 기술(위력 0)은 0
    위력/타입/명중/상성은 전부 ROM 표에서 읽음 (envs/rom_data.py)
    근사라서 빠진 것: 공격/방어 능력치, 레벨, 급소, 고정 데미지 기술(용의분노 등은 위력 칸이 40으로 들어 있음)"""
    from envs import rom_data

    m = rom_data.move(move_id)
    if m["power"] == 0:
        return 0.0
    stab = 1.5 if m["type"] in my_types else 1.0
    return m["power"] * m["accuracy"] * stab * rom_data.effectiveness(m["type"], enemy_types)


def decide_best_move(pb):
    """가장 센 기술 (move_score 최대). 전부 0이면 1번"""
    from envs import rom_data

    my_types = rom_data.species_types(mm.read_byte(pb, mm.PARTY_MON1_SPECIES))
    enemy_types = rom_data.species_types(mm.read_byte(pb, mm.ENEMY_SPECIES))
    scores = []
    for i in range(4):
        mid = mm.read_byte(pb, mm.PARTY_MON1_MOVES + i)
        if mid:
            scores.append((move_score(mid, my_types, enemy_types), i + 1))
    best = max(scores) if scores else (0, 1)
    return f"move{best[1] if best[0] > 0 else 1}"
