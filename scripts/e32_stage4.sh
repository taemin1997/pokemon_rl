#!/bin/bash
# E32 NIGHT 4단계 (notes/experiments/E32_NIGHT.md): 통합 실행 아침·밤 × 3모델 × 20판 (고친 실행기, --fresh-obs 기본)
#   인자·시드는 E26 판정 3 (scripts/e26_story_fresh.sh)과 같음. 10-23 C4 수정(판 시작 3프레임 이동)으로 옛 결과와 판 단위 비교는 안 함 → 전부 새로 잼
#   결과 목록: runs/e32/stage4.json (끝난 항목은 건너뜀) → 집계는 train.e32_report의 4단계 표
# 실행: nohup bash scripts/e32_stage4.sh >> runs/e32/stage4.log 2>&1 &
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
LIST=runs/e32/stage4.json
OUT=runs/e32/stage4
mkdir -p $OUT
ts() { date '+%m-%d %H:%M:%S'; }
[ -f $LIST ] || echo "{}" > $LIST

run() {   # 이름 모델경로 시간대
  local lab=$1 m=$2 c=$3 log=$OUT/$1_$3.log
  echo "== $(ts) 시작 $lab $c"
  POKE_CLOCK=$c timeout 14400 uv run python -m train.run_story $m --headless --episodes 20 --quiet \
    --targets "26,10;24,5;10,7" --no-heal-at 10,5 --max-steps 6000 --seed-offset 5000 > $log 2>&1
  local code=$? path=$(grep '저장:' $log | tail -1 | sed 's/저장: //' | tr '\\' '/' | tr -d '\r')
  if [ -z "$path" ]; then echo "== $(ts) 실패 $lab $c code=$code ($log)"; return; fi
  echo "== $(ts) 끝 $lab $c code=$code $path"
  until mkdir $LIST.lock 2>/dev/null; do sleep 1; done   # 동시에 끝난 실행끼리 목록을 덮어쓰지 않게
  uv run python -c "
import json
p = '$LIST'; d = json.load(open(p, encoding='utf-8')); d['$lab|$c'] = '$path'
json.dump(d, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)"
  rmdir $LIST.lock
}

MODELS="M2B_multi=models/M2B_multi/final.zip E30_D_s0=models/E30_D_s0/final.zip E30_D_s1=models/E30_D_s1/final.zip"
jobs=()
for mm in $MODELS; do
  lab=${mm%%=*}; m=${mm#*=}
  for c in morning night; do
    if grep -q "\"$lab|$c\"" $LIST; then echo "== $(ts) 건너뜀 $lab $c"; else jobs+=("$lab $m $c"); fi
  done
done
echo "== $(ts) 남은 항목 ${#jobs[@]}개"
# 코어 8개: 동시에 4개까지
for j in "${jobs[@]}"; do
  while [ $(jobs -rp | wc -l) -ge 4 ]; do sleep 20; done
  run $j &
  sleep 2   # 목록 파일 쓰기가 겹치지 않게 시작 시각을 조금 벌림
done
wait
uv run python -m train.e32_report > /dev/null 2>&1
echo "== $(ts) 전체 종료 → runs/e32/report.md"
