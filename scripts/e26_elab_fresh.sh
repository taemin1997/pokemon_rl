#!/bin/bash
# 10-21 설계 밖 추가 측정: E-lab을 첫 예측 전 1프레임 그린 뒤 시간대별로 다시 잼 (6단계 E-lab 결과는 그대로 둠)
#   이유: reset 직후 화면은 e_lab.state(아침에 저장) 팔레트 그대로 → 낮·밤 조건에서도 첫 화면이 아침 (progress/log/2026-10-21.md)
#   켬(POKE_FRESH_FIRST=1) 3시간대 + 대조로 끔 3시간대 (끔은 10-15 최종 평가와 판 단위로 같아야 함)
# 실행: nohup bash scripts/e26_elab_fresh.sh > runs/e26/elab_fresh.log 2>&1 &
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
ts() { date '+%m-%d %H:%M:%S'; }
LIST=runs/e26/elab_fresh.json
# 판정 3 재측정이 끝날 때까지 기다림 (같이 돌리면 느려짐)
until grep -q '전체 종료' runs/e26/story_fresh.log; do sleep 60; done
echo "== $(ts) 시작"
M="models/M2B_multi/final.zip models/A_E26_s0_r2/final.zip models/B_E26_s0/final.zip models/A_E26_s1_r2/final.zip models/B_E26_s1/final.zip"
echo "{" > $LIST.tmp
first=1
for mode in on off; do
  for c in morning day night; do
    if [ $mode = on ]; then f=1; else f=0; fi
    out=$(POKE_CLOCK=$c POKE_FRESH_FIRST=$f uv run python -m train.diagnose newbark_labdoor $M --runs 50 --workers 6 2>runs/e26/elab_fresh_${mode}_${c}.err)
    path=$(echo "$out" | grep '저장:' | tail -1 | sed 's/.*저장: //' | tr '\\' '/')
    echo "== $(ts) 끝 $mode $c $path"
    [ $first = 1 ] || echo "," >> $LIST.tmp
    first=0
    echo "\"$mode|$c\": \"$path\"" >> $LIST.tmp
  done
done
echo "}" >> $LIST.tmp
mv $LIST.tmp $LIST
echo "== $(ts) 전체 종료 → $LIST"
