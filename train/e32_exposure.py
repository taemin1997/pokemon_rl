"""
E32 사전 확인 (10-23): 학습 때 정책이 본 게임 시간대 비중

왜
- M2A·M2B 계열은 실제 시계(wall)로 학습 → 게임 시각 = 컴퓨터 시각 + 상태별 차이 (RTC가 컴퓨터 시각을 따라감)
- 밤 성능이 나쁜 게 '밤을 거의 못 보고 배워서'인지 보려면 학습 스텝이 어느 시간대에 쌓였는지 알아야 함

어떻게
1. 상태별 차이: 학습에 쓰인 시작 상태(구간 실제 시작 + 시연 상태 표본)를 wall 시계로 불러와 게임 시각과 컴퓨터 시각 차이를 잼
2. 학습 시각: tensorboard 기록(time/total_timesteps의 wall_time)으로 스텝이 쌓인 시각 구간
3. 스텝마다 게임 시각 = 컴퓨터 시각 + 차이 → 시간대 (아침 4~9시, 낮 10~17시, 밤 18~3시, envs/clock.py와 같은 경계)

실행: uv run python -m train.e32_exposure → runs/e32/exposure.json
"""

import datetime as dt
import io
import json
import os
import pickle
import time
import zlib
from pathlib import Path

import numpy as np

LINEAGE = [("GOAL2_route29", "models/GOAL2_route29"), ("M2A_multi", "models/M2A_multi"),
           ("M2A_r30", "models/M2A_r30"), ("M2B_multi", "models/M2B_multi")]
OUT = Path("runs/e32")


def tod(hour):
    return "morning" if 4 <= hour <= 9 else "day" if 10 <= hour <= 17 else "night"


def state_offsets():
    """상태별 (게임 시각 − 컴퓨터 시각), 분 단위. wall 시계로 불러와 몇 프레임 진행 후 HRAM 시각을 읽음"""
    os.environ["POKE_CLOCK"] = "wall"
    from pyboy import PyBoy
    from envs.segments import SEGMENTS
    pb = PyBoy("roms/gold_rom.gb", window="null")
    out = {}
    blobs = []
    for n, s in SEGMENTS.items():
        if Path(s.start_state).exists():
            blobs.append((f"{n}:start", Path(s.start_state).read_bytes()))
        if Path(s.demo_path).exists():
            st = pickle.loads(Path(s.demo_path).read_bytes())["states"]
            for i in sorted({0, len(st) // 2, len(st) - 1}):
                blobs.append((f"{n}:demo{i}", zlib.decompress(st[i])))
    for name, b in blobs:
        pb.load_state(io.BytesIO(b))
        pb.tick(10, False)
        now = dt.datetime.now()
        g = pb.memory[0xFF96] * 60 + pb.memory[0xFF98]
        h = now.hour * 60 + now.minute
        out[name] = ((g - h + 720) % 1440) - 720    # −12시간 ~ +12시간
    pb.stop(save=False)
    return out


def step_times(run_dirs):
    """tensorboard에서 (wall_time, total_timesteps) 목록. 이어 학습한 파일은 모두 합침"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    pts = []
    for d in run_dirs:
        for f in sorted(Path(d).glob("events.out.tfevents.*")):
            ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
            ea.Reload()
            tags = ea.Tags()["scalars"]
            if not tags:
                continue
            # SB3는 total_timesteps를 스칼라로 안 남김 → 기록 간격이 가장 촘촘한 태그의 (wall_time, step)
            tag = "time/fps" if "time/fps" in tags else max(tags, key=lambda t: len(ea.Scalars(t)))
            pts += [(e.wall_time, e.step) for e in ea.Scalars(tag)]
    return sorted(pts)


def main():
    offs = state_offsets()
    vals = np.array(list(offs.values()))
    off = int(np.median(vals))
    res = {"offset_minutes_median": off, "offset_spread_minutes": [int(vals.min()), int(vals.max())],
           "n_states": len(offs), "offsets": offs, "models": {}}
    print(f"게임 시각 − 컴퓨터 시각: 중앙값 {off}분, 범위 {vals.min()}~{vals.max()}분 (상태 {len(offs)}개)")
    for name, mdir in LINEAGE:
        runs = sorted(Path("runs").glob(f"{name}_[0-9]*"))
        pts = step_times(runs)
        cnt = {"morning": 0.0, "day": 0.0, "night": 0.0}
        for (t0, s0), (t1, s1) in zip(pts, pts[1:]):
            ds = s1 - s0
            if ds <= 0 or t1 - t0 > 3600:   # 끊긴 구간 (재시작 사이)은 건너뜀
                continue
            mid = dt.datetime.fromtimestamp((t0 + t1) / 2) + dt.timedelta(minutes=off)
            cnt[tod(mid.hour)] += ds
        tot = sum(cnt.values())
        span = [time.strftime("%m-%d %H:%M", time.localtime(pts[0][0])), time.strftime("%m-%d %H:%M", time.localtime(pts[-1][0]))] if pts else None
        res["models"][name] = {"steps": tot, "share": {k: round(v / tot, 3) if tot else None for k, v in cnt.items()}, "span": span,
                               "runs": [str(r) for r in runs]}
        print(f"{name:<16} 스텝 {tot:>10,.0f}  아침 {cnt['morning'] / max(tot, 1):.0%} 낮 {cnt['day'] / max(tot, 1):.0%} "
              f"밤 {cnt['night'] / max(tot, 1):.0%}  ({span})")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exposure.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장:", OUT / "exposure.json")


if __name__ == "__main__":
    main()
