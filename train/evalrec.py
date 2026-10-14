"""
평가 기록 (10-12, E26 1단계 재현성)

왜 필요한가
- 10-12에 "도라지체육관까지 20판 19/20"이 실제로는 연구소까지만 잰 결과였음 (run_story가 연구소에서 판을 끝냄)
  → 평가마다 "무슨 코드로, 어떤 게임 시각에, 어떤 시작 상태에서, 실제로 어느 구간을 지나 어디서 끝났나"를 같이 남기고
    의도한 범위와 다르면 무효로 표시
- 결과 숫자 파일 옆에 <같은 이름>_meta.json으로 저장 (기존 summary/raw 형식은 안 바꿈 → train.report 등 그대로 동작)
"""

import hashlib
import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_version():
    """git 커밋 + 수정 파일 목록 + 전체 작업 코드 해시.

    추적 중인 diff뿐 아니라 새로 만든 미추적 코드도 해시에 포함한다. E26 1단계처럼
    검증 코드 자체가 아직 커밋 전인 경우에도 실행에 사용한 정확한 코드 상태를 구분한다.
    """
    def git_text(*a):
        try:
            return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=20).stdout
        except Exception as e:      # git이 없거나 실패해도 평가는 계속
            return f"(git 실패: {e})"
    commit = git_text("rev-parse", "HEAD").strip()
    status = git_text("status", "--porcelain", "--untracked-files=all").splitlines()
    dirty = [ln[3:] for ln in status if ln.strip()]

    h = hashlib.sha1()
    try:
        diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=REPO, capture_output=True,
                              timeout=20, check=False).stdout
        h.update(b"tracked-diff\0" + diff)
        for line in status:
            if not line.startswith("?? "):
                continue
            rel = line[3:]
            p = REPO / rel
            if p.is_file():
                h.update(b"untracked\0" + rel.encode("utf-8", "surrogatepass") + b"\0")
                h.update(p.read_bytes())
    except Exception as e:
        h.update(f"hash 실패: {e}".encode("utf-8", "replace"))
    return {"commit": commit, "dirty_files": dirty,
            "diff_sha1": h.hexdigest() if dirty else None}


def write_meta(stem_path, meta):
    """stem_path: 결과 파일 경로에서 확장자 뺀 것 (예: runs/diagnose/route30_1012_2000_M2B_multi)"""
    meta = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "code": code_version(), **meta}
    p = Path(f"{stem_path}_meta.json")
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return p
