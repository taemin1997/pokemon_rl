#!/bin/bash
# E26 2단계 기준선 (10-14): M2B로 E-lab / E-violet × 아침·낮·밤 × 50판. 학습·모델·규칙 변경 없음, 평가만
# 실행: nohup bash scripts/e26_stage2.sh > runs/e26/stage2.log 2>&1 &
cd /c/pokemon_rl
export PYTHONIOENCODING=utf-8
stamp () { echo "== $(date '+%m-%d %H:%M:%S') $*"; }
M=models/M2B_multi/final.zip
LIST=runs/e26/stage2_runs.json
echo "{}" > $LIST
stamp "E26 2단계 기준선 시작 (코드 $(git rev-parse --short HEAD), 수정 파일 $(git status --porcelain --untracked-files=no | wc -l)개)"
for seg in newbark_labdoor violet_pc_counter; do
  for c in morning day night; do
    stamp "$seg $c"
    out=$(POKE_CLOCK=$c uv run python -m train.diagnose $seg $M --runs 50 --workers 6 2>&1 | grep "저장:" | sed 's/저장: //' | tr -d '\r')
    echo "  → $out"
    uv run python -c "import json,sys; p='$LIST'; d=json.load(open(p, encoding='utf-8-sig')); d['${seg}_$c']=sys.argv[1]; json.dump(d, open(p,'w'), indent=1)" "$out"
  done
done
stamp "판정"
uv run python -m train.e26_stage2_report $LIST
stamp "끝"
