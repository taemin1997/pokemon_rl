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
