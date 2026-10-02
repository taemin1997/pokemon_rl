"""
2단계: 게임을 직접 플레이하면서 메모리 값을 실시간으로 찍어보는 스크립트

왜 필요한가
- 3단계 Gym 환경의 "관찰"과 "보상"은 전부 메모리 값에서 나옴
  (예: 새 맵에 들어가면 +보상, 배지 비트가 켜지면 큰 보상)
- 주소가 틀리면 에이전트는 엉뚱한 숫자를 보고 학습함 → 학습이 안 되는데 원인을 못 찾음
- 그래서 학습 코드를 짜기 전에 "읽은 값 == 화면에 보이는 값"을 눈으로 확인하는 단계

영문판 vs 한국어판
- pret/pokegold 심볼은 영문판 기준임
- 한국어판은 이름 같은 문자열이 한 글자 2바이트라서 버퍼가 길어짐
  → 그 뒤에 오는 변수들이 통째로 뒤로 밀림 (구간마다 밀린 양이 다름)
- 아래 KO 주소는 화면 값(소지금 3000원, ID, 걸을 때 바뀌는 좌표)을 메모리 전체에서
  패턴 검색해서 찾은 후보임. 이 스크립트로 직접 플레이하면서 맞는지 확인하는 게 목적

조작 (PyBoy 기본 키)
- 방향키: 이동 / A: a키 / B: s키 / START: Enter / SELECT: Backspace
- 창을 닫으면 종료되고, 종료 시점 상태가 states/last_session.state 로 저장됨

실행
- 처음부터:          uv run python step2_pyboy/01_read_memory.py
- 저장한 상태에서:   uv run python step2_pyboy/01_read_memory.py states/last_session.state
"""

import sys
from pathlib import Path

from pyboy import PyBoy

ROM_PATH = "roms/gold_rom.gb"
STATE_OUT = Path("states/last_session.state")

# 몇 프레임마다 메모리를 읽을지. 게임은 초당 60프레임이라 30이면 0.5초마다 읽음
# 매 프레임 읽어도 되지만, 바뀐 것만 출력하니까 이 정도면 충분함
READ_EVERY = 30

# ---------------------------------------------------------------
# 주소표: 이름 → (영문판 주소, 한국어판 주소)
# ---------------------------------------------------------------
# 영문판 주소를 같이 두는 이유: 실행해보면 영문 주소 쪽은 엉뚱한 값이 나오는 걸
# 직접 볼 수 있음 → "심볼을 그대로 믿으면 안 된다"는 근거가 됨 (포트폴리오에도 쓸 만함)
#
# 상태 표시:
#   [검증] = 헤드리스로 화면 값과 대조해서 확인함
#   [추정] = 주변 변수들의 밀린 양으로 계산만 함. 직접 확인 필요
ADDR = {
    # 맵 그룹/번호/좌표: 영문판보다 +0xFD 밀림 [검증]
    # 주인공 방에서 24:7, 계단으로 1층 내려가면 24:6 → 영문판 맵 번호와 값 자체는 같음
    "map_group": (0xDA00, 0xDAFD),
    "map_number": (0xDA01, 0xDAFE),
    "y": (0xDA02, 0xDAFF),
    "x": (0xDA03, 0xDB00),
    # 소지금: +0xB3 밀림 [검증] (트레이너 카드 3000원 = 00 0B B8)
    # 2세대는 소지금을 3바이트 이진수(빅엔디언)로 저장함 (1세대는 BCD였음)
    "money": (0xD573, 0xD626),
    # 배지: 영문판에서 소지금+9 위치라 같은 비율로 계산 [추정]
    # 첫 배지를 따기 전엔 0이라 지금은 확인 불가. 체육관 깨는 시점에 다시 확인해야 함
    "johto_badges": (0xD57C, 0xD62F),
    # 파티: 맵과 같이 +0xFD [추정 → 반쯤 검증]
    # 포켓몬 0마리일 때 "개수 0, 바로 다음 칸 0xFF(목록 끝 표시)" 모양은 확인함
    # 연구소에서 첫 포켓몬 받으면 1이 되는지 확인 필요
    "party_count": (0xDA22, 0xDB1F),
    "party_species": (0xDA23, 0xDB20),
}


def read_byte(pyboy, addr):
    # pyboy.memory[1, addr] = "WRAM 1번 뱅크의 addr"
    # 컬러 게임보이는 0xD000~0xDFFF 구간에 WRAM 뱅크 7개를 번갈아 끼워 씀
    # 골드는 평소엔 1번 뱅크를 끼워두지만 잠깐씩 다른 뱅크로 바꾸는 순간이 있음
    # → 그 순간에 읽으면 엉뚱한 값이 나오니까 뱅크 번호를 명시해서 항상 1번에서 읽음
    if addr >= 0xD000:
        return pyboy.memory[1, addr]
    return pyboy.memory[addr]


def read_state(pyboy, which):
    """which=0이면 영문 주소, 1이면 한국어판 주소로 읽어서 dict로 돌려줌"""

    def a(name):
        return ADDR[name][which]

    # 소지금 3바이트 → 정수. 빅엔디언이라 앞 바이트가 가장 큰 자리
    money = (
        read_byte(pyboy, a("money")) << 16
        | read_byte(pyboy, a("money") + 1) << 8
        | read_byte(pyboy, a("money") + 2)
    )

    count = read_byte(pyboy, a("party_count"))
    # 종 번호 목록. 최대 6마리라 6칸만 읽고, count만큼만 씀
    # (count가 쓰레기 값이면 이상한 숫자들이 나오는데, 그게 곧 "주소 틀림" 신호임)
    species = [read_byte(pyboy, a("party_species") + i) for i in range(min(count, 6))]

    return {
        "map": (read_byte(pyboy, a("map_group")), read_byte(pyboy, a("map_number"))),
        "xy": (read_byte(pyboy, a("x")), read_byte(pyboy, a("y"))),
        "money": money,
        "party": (count, species),
        # 배지는 비트 플래그. 0번 비트 = 도라지 체육관(윙배지)
        "badges": read_byte(pyboy, a("johto_badges")),
    }


def fmt(s):
    return (
        f"맵 {s['map'][0]}:{s['map'][1]} | 좌표 x={s['xy'][0]} y={s['xy'][1]} | "
        f"돈 {s['money']} | 파티 {s['party'][0]}마리 {s['party'][1]} | "
        f"배지 {s['badges']:08b}"
    )


def main():
    pyboy = PyBoy(ROM_PATH)

    # 인자로 세이브 스테이트를 주면 그 시점부터 시작
    # (매번 오프닝부터 다시 하지 않으려고. 3단계 reset()도 같은 원리로 씀)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as f:
            pyboy.load_state(f)
        print(f"상태 불러옴: {sys.argv[1]}")

    prev = None
    frame = 0
    while pyboy.tick():
        frame += 1
        if frame % READ_EVERY:
            continue

        ko = read_state(pyboy, 1)
        # 값이 바뀔 때만 출력. 매번 찍으면 터미널이 같은 줄로 도배돼서 변화를 못 봄
        if ko != prev:
            en = read_state(pyboy, 0)
            print(f"[KO] {fmt(ko)}")
            print(f"[EN] {fmt(en)}")
            print()
            prev = ko

    # 창을 닫은 시점의 상태를 저장
    # 세이브 스테이트 = 에뮬레이터 전체 스냅샷 (게임 안의 "레포트" 저장과는 별개)
    # 3단계에서 "에피소드마다 이 시점으로 되돌리기"용으로 쓸 예정
    STATE_OUT.parent.mkdir(exist_ok=True)
    with open(STATE_OUT, "wb") as f:
        pyboy.save_state(f)
    print(f"상태 저장함: {STATE_OUT}")

    pyboy.stop()


if __name__ == "__main__":
    main()
