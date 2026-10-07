"""
한국어판 금 버전 메모리 주소 + 읽기 함수

왜 따로 뺐나
- 2단계에서 검증한 주소를 한 곳에만 둬야, 나중에 주소를 고칠 때 여기만 고치면 됨
- 환경(gold_env.py)은 "무슨 값을 보상으로 쓸지"만 신경 쓰고
  "그 값이 메모리 어디에 있는지"는 몰라도 되게 나눔
- 주소 검증 기록: notes/02_memory_map.md
"""

# 전부 WRAM 1번 뱅크 기준 (영문판 대비 밀린 양은 notes 참고)
MAP_GROUP = 0xDAFD
MAP_NUMBER = 0xDAFE
Y_COORD = 0xDAFF
X_COORD = 0xDB00
MONEY = 0xD626  # 3바이트, 빅엔디언 이진수
JOHTO_BADGES = 0xD62F  # [추정] 첫 배지 때 확인 필요. 0번 비트 = 윙배지(도라지)
PARTY_COUNT = 0xDB1F
PARTY_SPECIES = 0xDB20
# 파티 1번 포켓몬 구조체: 맵/파티와 같은 +0xFD [검증] 시작 상태에서 종 158(리아코), Lv5
PARTY_MON1_SPECIES = 0xDB27
PARTY_MON1_LEVEL = 0xDB46
# 경험치 3바이트 빅엔디언 [검증] 시작 135 → 꼬렛 Lv4 이긴 뒤 167 (화면 "32 경험치를 얻었다"와 일치)
# 레벨보다 이걸 보는 이유: 한 번 이겨도 레벨업까진 안 갈 수 있음 (Lv6까지 44 필요)
PARTY_MON1_EXP = 0xDB2F
# HP 2바이트 빅엔디언 [검증] 시작 상태 20 / 최대 20 (화면 "HP 20/20"과 일치). 영문판 DA4C → +0xFD
PARTY_MON1_HP = 0xDB49
PARTY_MON1_MAXHP = 0xDB4B
# 기술 번호 4칸 (영문판 DA2C → +0xFD) [검증] 시작 상태 [10, 43, 0, 0] = 할퀴기(10), 째려보기(43)
# 기술 번호 체계는 pret/영문판과 같음 → ROM 기술 표를 그대로 번호로 찾을 수 있음
PARTY_MON1_MOVES = 0xDB29
# 지닌 물건 (영문판 DA2B → +0xFD) [검증] 시작 상태 173 = 나무열매(BERRY, pret 0xAD). 아이템 번호 체계도 영문판과 같음
# 스타팅 포켓몬은 나무열매를 지니고 시작 → 전투 중 HP가 절반 이하면 게임이 자동으로 먹임 (에이전트 행동 아님)
PARTY_MON1_ITEM = 0xDB28
# 파티 포켓몬 구조체 크기. 2번째 포켓몬 = 1번 주소 + 0x30 [추정: pret 기준, 2마리 이상일 때 검증 필요]
PARTY_MON_SIZE = 0x30
# 전투 모드 (0=전투 아님, 1=야생, 2=트레이너). 영문판 D116 → +0xBD
# [검증] 29번도로 야생 꼬렛 Lv4 전투: 전투 내내 1, 끝나면 0. 집 안 대화창에서는 0 유지
# 같은 +0xBD로 적 종/레벨까지 화면과 일치함 → 이 구간의 밀린 양이 확실함
# (처음엔 D21B를 전투 플래그로 착각했는데, 대화창이 떠 있어도 1이 되는 다른 플래그였음)
# 트레이너 전투(2)는 아직 확인 못 함
BATTLE_MODE = 0xD1D3
ENEMY_SPECIES = 0xD1AC  # [검증] 꼬렛 = 19
ENEMY_LEVEL = 0xD1B9    # [검증] 4
# 적 HP 2바이트 빅엔디언 (영문판 wEnemyMonHP D0FF → +0xBD, 적 종/레벨과 같은 밀린 양)
# [검증 10-07] 꼬리선 Lv3: 15/15 → 할퀴기 한 번에 9 (화면 HP 바 약 60%), 구구 Lv3도 같은 위치
ENEMY_HP = 0xD1BC
ENEMY_MAXHP = 0xD1BE

# 2차원 메뉴 데이터 (전투 메뉴, 기술 목록 등 게임 공통 메뉴 루틴이 씀) [검증 10-07, 야생 전투 2개 × 3턴]
#   메뉴가 열릴 때 게임이 이 블록을 새로 씀 → 값으로 "지금 어떤 메뉴가 열려 있나"를 판별
#   - MENU_INIT_Y: 커서 시작 줄 (전투 메인 메뉴 14, 기술 목록 10)
#   - MENU_ROWS / MENU_COLS: 행 수 / 열 수 (메인 메뉴 2×2, 기술 목록 = 기술 수 × 1)
#   - MENU_CURSOR_Y / X: 커서 (1부터). 메인 메뉴: (1,1) 싸우다, (1,2) 가방, (2,1) 포켓몬, (2,2) 도망치다
#     기술 목록: CURSOR_Y = 기술 칸 번호 (1~4). 끝에서 아래로 누르면 처음으로 돌아감
#   주의: 메뉴가 닫혀도 블록이 남아 있음 (기술 고른 뒤엔 CURSOR_Y가 0이 됨). 판별 함수가 행/열까지 같이 봄
#   [추정] 기술이 3~4개일 때 기술 목록 시작 줄이 10 그대로인지, 트레이너 전투/가방/포켓몬 메뉴는 미확인
MENU_INIT_Y = 0xCEAD
MENU_INIT_X = 0xCEAE
MENU_ROWS = 0xCEAF
MENU_COLS = 0xCEB0
MENU_CURSOR_Y = 0xCEB5
MENU_CURSOR_X = 0xCEB6
# 지금 커서가 올라간 기술의 정보 (영문판 wPlayerMoveStruct 계열로 추정): 번호, 효과, 위력, 타입, 명중, PP
#   [관찰 10-07] 할퀴기 위 → 10, 0, 40, 0, ?, 35 / 째려보기 위 → 43, 19, 0, 0, ?, 30. 규칙 판단(B2)에선 ROM 기술표를 쓸 예정이라 참고용
CUR_MOVE_STRUCT = 0xCAEF


def read_byte(pyboy, addr):
    # 0xD000~0xDFFF는 컬러 게임보이의 WRAM 뱅크 전환 구간
    # 게임이 잠깐 다른 뱅크로 바꾼 순간에 읽어도 안전하게 항상 1번 뱅크를 지정함
    if addr >= 0xD000:
        return pyboy.memory[1, addr]
    return pyboy.memory[addr]


def read_position(pyboy):
    """(맵그룹, 맵번호, x, y) 튜플. 탐험 보상에서 '처음 밟은 칸' 판정에 씀"""
    return (
        read_byte(pyboy, MAP_GROUP),
        read_byte(pyboy, MAP_NUMBER),
        read_byte(pyboy, X_COORD),
        read_byte(pyboy, Y_COORD),
    )


def read_money(pyboy):
    return (
        read_byte(pyboy, MONEY) << 16
        | read_byte(pyboy, MONEY + 1) << 8
        | read_byte(pyboy, MONEY + 2)
    )


def read_party_count(pyboy):
    return read_byte(pyboy, PARTY_COUNT)


def read_badges(pyboy):
    """배지 비트 플래그(0~255). 켜진 비트 수 = 배지 개수"""
    return read_byte(pyboy, JOHTO_BADGES)


def in_battle(pyboy):
    """전투 중이면 True (야생/트레이너 둘 다)"""
    return read_byte(pyboy, BATTLE_MODE) != 0


def read_mon1_exp(pyboy):
    """파티 1번 포켓몬 경험치. 늘었으면 전투에서 이긴 것"""
    return (
        read_byte(pyboy, PARTY_MON1_EXP) << 16
        | read_byte(pyboy, PARTY_MON1_EXP + 1) << 8
        | read_byte(pyboy, PARTY_MON1_EXP + 2)
    )



def read_party_hp_total(pyboy):
    """파티 전체 HP 합. 0이면 전멸 (화이트아웃 직전)
    2번째 이후 포켓몬은 구조체 크기(0x30)를 더해서 읽음 [추정]"""
    total = 0
    for i in range(min(read_party_count(pyboy), 6)):
        a = PARTY_MON1_HP + i * PARTY_MON_SIZE
        total += read_byte(pyboy, a) << 8 | read_byte(pyboy, a + 1)
    return total


def read_enemy_hp(pyboy):
    """(적 현재 HP, 최대 HP). 전투 RL 보상(준 데미지), 규칙 판단(막타)에 씀"""
    hp = read_byte(pyboy, ENEMY_HP) << 8 | read_byte(pyboy, ENEMY_HP + 1)
    mx = read_byte(pyboy, ENEMY_MAXHP) << 8 | read_byte(pyboy, ENEMY_MAXHP + 1)
    return hp, mx


def battle_menu(pyboy):
    """전투 중 지금 열려 있는 메뉴: "main"(싸우다/가방/포켓몬/도망), "moves"(기술 목록), None(문구/연출 중)

    전투 실행 층(메뉴 매크로)이 "입력을 기다리는 순간"을 알아야 버튼을 누를 수 있어서 만듦
    (지금 자동 전투는 A를 그냥 계속 눌러서 이게 필요 없었음)
    판별: 메뉴 데이터 블록의 행/열 수와 커서. 메뉴가 닫혀도 블록이 남는 걸 커서 0으로 거름"""
    if not in_battle(pyboy):
        return None
    rows, cols = read_byte(pyboy, MENU_ROWS), read_byte(pyboy, MENU_COLS)
    y, x = read_byte(pyboy, MENU_CURSOR_Y), read_byte(pyboy, MENU_CURSOR_X)
    if rows == 2 and cols == 2 and 1 <= y <= 2 and 1 <= x <= 2:
        return "main"
    if cols == 1 and read_byte(pyboy, MENU_INIT_Y) == 10 and 1 <= y <= 4:
        return "moves"
    return None
