"""
커밋 전 저작물 검사 (10-14): 게임 ROM·세이브·상태·화면·영상·소리가 저장소에 들어가지 않게 막음

검사 (스테이징된 파일 대상, 하나라도 걸리면 커밋 중단)
1. 확장자: ROM/세이브/상태/모델/영상/소리 (.gb .gbc .gba .sav .srm .rtc .state .pkl .zip .mp4 .gif ...)
2. 이미지: 허용 목록(ALLOWED_IMAGES, docs/figures/)에 없는 이미지 — 직접 만든 그래프만 허용, 게임 화면 금지
3. 크기: 2MB 넘는 파일
4. 내용: 게임보이 ROM 헤더(0x104의 닌텐도 로고 바이트)가 있는 파일 — 확장자를 바꿔도 잡음

설치 (한 번): python scripts/check_assets.py --install   → .git/hooks/pre-commit 생성
직접 검사:    python scripts/check_assets.py            (스테이징된 파일)
             python scripts/check_assets.py --all      (추적 중인 모든 파일)
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BLOCK_EXT = {".gb", ".gbc", ".gba", ".sgb", ".sav", ".srm", ".rtc", ".state", ".ss", ".pkl", ".zip", ".bin",
             ".mp4", ".gif", ".avi", ".mkv", ".webm", ".mov", ".wav", ".mp3", ".ogg", ".flac"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
ALLOWED_IMAGES = {"notes/img/v1_explore_progress.png", "notes/img/v2_ent003_progress.png"}   # 직접 만든 이동 지도 그래프
ALLOWED_IMAGE_DIRS = ("docs/figures/",)
MAX_BYTES = 2 * 1024 * 1024
# 게임보이 카트리지 헤더 0x104~0x133의 닌텐도 로고 앞부분 (모든 정식 ROM에 있음)
GB_LOGO = bytes([0xCE, 0xED, 0x66, 0x66, 0xCC, 0x0D, 0x00, 0x0B])


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


def problems(paths):
    bad = []
    for rel in paths:
        p = REPO / rel
        ext = p.suffix.lower()
        if ext in BLOCK_EXT:
            bad.append((rel, f"금지 확장자 {ext} (ROM·세이브·상태·모델·영상·소리)"))
            continue
        if ext in IMAGE_EXT and rel not in ALLOWED_IMAGES and not rel.startswith(ALLOWED_IMAGE_DIRS):
            bad.append((rel, "허용 안 된 이미지 (게임 화면 금지. 직접 만든 그래프는 docs/figures/에)"))
            continue
        if not p.is_file():
            continue
        size = p.stat().st_size
        if size > MAX_BYTES:
            bad.append((rel, f"큰 파일 {size / 1e6:.1f}MB"))
            continue
        with open(p, "rb") as f:
            head = f.read(0x10C)
        if len(head) >= 0x10C and head[0x104:0x10C] == GB_LOGO:
            bad.append((rel, "게임보이 ROM 헤더가 있음"))
    return bad


def main():
    if "--install" in sys.argv:
        hook = REPO / ".git" / "hooks" / "pre-commit"
        # 'python'이 Windows 스토어 안내 스텁일 수 있어서 (10-14) 프로젝트 가상환경 파이썬 → 없으면 uv run
        hook.write_text("#!/bin/sh\n# 저작물 검사 (scripts/check_assets.py)\n"
                        "if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe\n"
                        "elif [ -x .venv/bin/python ]; then PY=.venv/bin/python\n"
                        "else PY=\"uv run python\"; fi\n"
                        "$PY scripts/check_assets.py || exit 1\n", encoding="utf-8")
        print("설치:", hook)
        return 0
    if "--all" in sys.argv:
        paths = [l for l in git("ls-files").splitlines() if l]
    else:
        paths = [l for l in git("diff", "--cached", "--name-only", "--diff-filter=ACMR").splitlines() if l]
    bad = problems(paths)
    for rel, why in bad:
        print(f"[막음] {rel}: {why}")
    if bad:
        print("저작물 검사 실패 → 커밋 중단. 정말 필요한 직접 만든 그림이면 docs/figures/에 두거나 ALLOWED_IMAGES에 추가")
        return 1
    print(f"저작물 검사 통과 ({len(paths)}개 파일)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
