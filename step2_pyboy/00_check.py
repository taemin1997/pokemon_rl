from pyboy import PyBoy

# ROM 경로를 주면 에뮬레이터 창이 뜸 (파일명은 본인 것에 맞추면 됨)
pyboy = PyBoy("roms/gold_rom.gb")

# tick() = 게임을 1프레임 진행시킴
# 창을 닫으면 False를 돌려줘서 루프가 끝남
while pyboy.tick():
    pass

# 에뮬레이터 정리. 이때 ROM 옆에 .ram 세이브 파일이 생김 (.gitignore에 이미 넣어둠)
pyboy.stop()