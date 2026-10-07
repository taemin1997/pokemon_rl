"""
ROM에서 게임 데이터 표 읽기: 기술표, 기본 능력치(타입), 타입 상성표 (전투 모듈 B2 규칙 판단, B3 관찰용)

왜 ROM에서 읽나 (하드코딩 최소화 원칙)
- 기술 위력/타입, 포켓몬 타입, 상성을 손으로 적으면 251종 × 251기술을 옮겨 적어야 하고 틀리기 쉬움
- 게임이 쓰는 표를 그대로 읽으면 정확하고, 한국어판에서 위치가 달라도 상관없음

위치를 어떻게 찾나
- 주소를 박아두지 않고, 표 맨 앞의 "알려진 값"(영문판 pret과 같은 데이터) 패턴으로 ROM을 검색함
  → 한국어판은 문자열 때문에 코드/데이터 위치가 밀렸지만 표 안의 값은 같음 (메모리 주소 때와 같은 상황)
- 패턴은 각각 ROM에서 정확히 한 번만 나옴 (10-07 확인): 기술표 0x4172E (뱅크 0x10), 기본 능력치 0x51BDF, 상성표 0x34D01
- 검증 (10-07): 할퀴기 위력 40/PP 35, 째려보기 효과 19/위력 0, 몸통박치기 35/명중 95%,
  리아코 물/물, 꼬리선 노말, 구구 노말/비행 → 게임 화면/알려진 값과 일치

구조 (pret pokegold 기준)
- 기술 7바이트: 애니메이션(=번호), 효과, 위력, 타입, 명중(0~255, 255 = 100%), PP, 부가 효과 확률
- 기본 능력치 32바이트: 종 번호, HP, 공격, 방어, 스피드, 특공, 특방, 타입1, 타입2, …
- 상성: (공격 타입, 방어 타입, 배율 × 10) 3바이트씩. 0xFE = 투시(Foresight) 경계, 0xFF = 끝
"""

from functools import lru_cache

ROM_PATH = "roms/gold_rom.gb"

# 표 맨 앞 패턴 (영문판과 같은 값)
MOVES_SIG = bytes([1, 0, 40, 0, 255, 35, 0, 2, 0, 50, 1, 255, 25, 0])        # 막치기, 태권당수
BASE_SIG = bytes([1, 45, 49, 49, 45, 65, 65, 0x16, 3])                      # 이상해씨
TYPES_SIG = bytes([0, 5, 5, 0, 9, 5, 0x14, 0x14, 5, 0x14, 0x15, 5, 0x14, 0x16, 20])  # 노말→바위/강철, 불→불/물/풀

NUM_MOVES = 251
NUM_SPECIES = 251

# 타입 번호 (pret constants/type_constants.asm). 이름은 사람이 읽는 용도 (분석/로그)
TYPE_NAMES = {0: "노말", 1: "격투", 2: "비행", 3: "독", 4: "땅", 5: "바위", 7: "벌레", 8: "고스트", 9: "강철",
              0x13: "???", 0x14: "불꽃", 0x15: "물", 0x16: "풀", 0x17: "전기", 0x18: "에스퍼", 0x19: "얼음",
              0x1A: "드래곤", 0x1B: "악"}
# 2세대 규칙: 0x14(불꽃) 이상 = 특수 기술 (특공/특방으로 계산), 그 아래 = 물리
SPECIAL_FROM = 0x14


def _find(rom, sig, name):
    i = rom.find(sig)
    if i < 0 or rom.count(sig) != 1:
        raise RuntimeError(f"ROM에서 {name} 표를 못 찾음 (패턴 {rom.count(sig)}번)")
    return i


@lru_cache(maxsize=1)
def tables(rom_path=ROM_PATH):
    rom = open(rom_path, "rb").read()
    m = _find(rom, MOVES_SIG, "기술")
    moves = {}
    for n in range(1, NUM_MOVES + 1):
        e = rom[m + (n - 1) * 7: m + n * 7]
        moves[n] = {"effect": e[1], "power": e[2], "type": e[3], "accuracy": e[4] / 255, "pp": e[5], "chance": e[6]}
    b = _find(rom, BASE_SIG, "기본 능력치")
    base = {}
    for sp in range(1, NUM_SPECIES + 1):
        e = rom[b + (sp - 1) * 32: b + (sp - 1) * 32 + 9]
        base[sp] = {"hp": e[1], "atk": e[2], "def": e[3], "spd": e[4], "sat": e[5], "sdf": e[6], "types": (e[7], e[8])}
    t = _find(rom, TYPES_SIG, "상성")
    chart = {}
    k = t
    while rom[k] != 0xFF:
        if rom[k] == 0xFE:   # 투시 경계 (그 뒤 고스트 면역 줄들도 일반 규칙으로 같이 씀)
            k += 1
            continue
        chart[(rom[k], rom[k + 1])] = rom[k + 2] / 10
        k += 3
    return {"moves": moves, "base": base, "chart": chart}


def move(n):
    return tables()["moves"][n]


def species_types(sp):
    return tables()["base"][sp]["types"]


def effectiveness(move_type, defender_types):
    """상성 배율 (2타입이면 곱함, 같은 타입 두 번이면 한 번만)"""
    chart = tables()["chart"]
    mult = 1.0
    for t in dict.fromkeys(defender_types):
        mult *= chart.get((move_type, t), 1.0)
    return mult
