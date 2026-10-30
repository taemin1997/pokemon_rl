"""
게임 시계 조건 (10-12, E26 1단계 재현성)

왜 필요한가
- PyBoy의 RTC는 컴퓨터 시각을 따라감 → 같은 세이브 상태·같은 시드도 실행 시각마다 게임 시간대(아침/낮/밤)가 달라짐
  (10-10 발견: 같은 모델 통합 실행 10/10 vs 26/30, 10-12: M2A_r30 route30_rev 아침 실패 24%)
- 평가를 재현하려면 시계를 고정해야 하고, 시간대 일반화를 보려면 아침/낮/밤을 골라서 고정할 수 있어야 함

어떻게 (10-12 메모리 확인)
- rtc_lock_experimental(True): RTC가 항상 0을 보고함 → 게임 시각 = RTC 0 + 게임이 저장한 시작 시각
- 시작 시각(시) = WRAM D298 [검증: start.state 값 8 = 고정 시 게임 시각 8:21:15의 '8', 14/23으로 바꾸면 게임 시각 14:21 / 23:21]
- 게임 시각 = HRAM FF96 시 / FF98 분 / FF9A 초 [검증: 고정 8:21:15에서 안 움직임, 실제 시계에선 흐름]
- 시간대 = WRAM D214 (D622도 같음): 0 아침 / 1 낮 / 2 밤 [검증: 7시 0, 14시 1, 23시 2]
- 시간대 경계 (금·은): 아침 4~9시, 낮 10~17시, 밤 18~3시 → 각 구간 가운데쯤 시각을 고름 (게임 규칙, 사람이 고른 값)

사용: 환경변수 POKE_CLOCK = wall (기본, 실제 시계) | lock (고정, 세이브 상태 시각 그대로) | morning | day | night
- 옛 POKE_RTC_LOCK=1은 lock과 같음 (10-10~12 평가 호환)
- 병렬 진단 프로세스까지 한 번에 적용하려고 인자 대신 환경변수
- 시작 시각은 WRAM이라 load_state 때마다 덮어써짐 → 세이브 상태를 불러올 때마다 after_load() 호출 (GoldEnv.reset이 함)
"""

import os

START_HOUR = 0xD298
TIME_OF_DAY = 0xD214
GAME_HOUR, GAME_MIN, GAME_SEC = 0xFF96, 0xFF98, 0xFF9A
PRESETS = {"morning": 7, "day": 14, "night": 23}
TOD_NAMES = {0: "아침", 1: "낮", 2: "밤"}


def mode():
    m = os.environ.get("POKE_CLOCK", "").strip().lower()
    if not m:
        return "lock" if os.environ.get("POKE_RTC_LOCK") == "1" else "wall"
    assert m in ("wall", "lock", *PRESETS), f"POKE_CLOCK 값이 이상함: {m}"
    return m


def setup(pb):
    """PyBoy를 만든 직후 한 번. wall이 아니면 RTC 고정"""
    if mode() != "wall":
        pb.rtc_lock_experimental(True)


def after_load(pb):
    """세이브 상태를 불러온 직후. 시간대 조건이면 시작 시각(시)을 덮어씀 (분·초는 상태 값 그대로)"""
    m = mode()
    if m in PRESETS:
        pb.memory[1, START_HOUR] = PRESETS[m]


def render_as(pb, preset, frames=4):
    """E32 (10-23): 지금 게임 상태를 다른 시간대 색으로 그린 화면 (상태는 되돌림 → 게임 진행엔 영향 없음)
    시작 시각만 바꾸고 frames 진행 (마지막 1프레임 그림) — 시간대 팔레트가 1~3프레임 뒤 반영 (GoldEnv.RESET_FRAMES와 같은 근거)
    확인: 밤 게임을 아침으로 그린 화면 = 같은 칸 아침 게임 화면 (route29·route31 100% 같음, 연구소 문 앞 애니메이션 차 0.3)
    반환 후 PyBoy 화면 버퍼는 다시 그린 화면 → 호출한 쪽은 다음 걸음이 새로 그릴 때까지 화면 버퍼를 관찰로 쓰면 안 됨"""
    import io
    buf = io.BytesIO()
    pb.save_state(buf)
    pb.memory[1, START_HOUR] = PRESETS[preset]
    pb.tick(frames - 1, False)
    pb.tick(1, True)
    screen = pb.screen.ndarray[:, :, :3].copy()
    pb.load_state(io.BytesIO(buf.getvalue()))
    return screen


def info(pb):
    """지금 게임 시각과 시간대 (기록용). 시간대 값은 게임이 필드 루프에서 갱신하므로 불러온 직후엔 이전 값일 수 있음"""
    tod = pb.memory[1, TIME_OF_DAY]
    return {"clock": mode(), "game_time": f"{pb.memory[GAME_HOUR]}:{pb.memory[GAME_MIN]:02d}:{pb.memory[GAME_SEC]:02d}",
            "time_of_day": TOD_NAMES.get(tod, str(tod))}
