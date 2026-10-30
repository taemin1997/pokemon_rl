#!/bin/bash
# 10-20 밤: 판정 3 다시 (지난 화면 버그 고친 run_story, --fresh-obs 기본) 5모델 × 아침·낮·밤 × 20판
# 판정 3 인자·시드는 6단계와 같음. 결과 목록: runs/e26/story_fresh.json (끝난 항목은 건너뜀)
# 10-20 낮 확인 시험의 켬 실행 4개(A_s1·M2B × 낮·밤)는 같은 조건이라 그대로 씀
# 실행: nohup bash scripts/e26_story_fresh.sh > runs/e26/story_fresh.log 2>&1 &
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
LIST=runs/e26/story_fresh.json
OUT=runs/e26/story_fresh
mkdir -p $OUT
ts() { date '+%m-%d %H:%M:%S'; }
[ -f $LIST ] || cat > $LIST <<'EOF'
{"A_s1|day": "runs/story/m2a_1020_105330_468483700_A_E26_s1_r2.json",
 "A_s1|night": "runs/story/m2a_1020_114227_292730600_A_E26_s1_r2.json",
 "M2B|day": "runs/story/m2a_1020_121842_134766300_M2B_multi.json",
 "M2B|night": "runs/story/m2a_1020_121244_424536700_M2B_multi.json"}
EOF

run() {   # 이름 모델경로 시간대
  local lab=$1 m=$2 c=$3 log=$OUT/$1_$3.log
  echo "== $(ts) 시작 $lab $c"
  POKE_CLOCK=$c timeout 14400 uv run python -m train.run_story $m --headless --episodes 20 --quiet \
    --targets "26,10;24,5;10,7" --no-heal-at 10,5 --max-steps 6000 --seed-offset 5000 > $log 2>&1
  local code=$? path=$(grep '저장:' $log | tail -1 | sed 's/저장: //' | tr '\\' '/')
  echo "== $(ts) 끝 $lab $c code=$code $path"
  [ -z "$path" ] && return
  until mkdir $LIST.lock 2>/dev/null; do sleep 1; done   # 동시에 끝난 실행끼리 목록을 덮어쓰지 않게
  uv run python -c "
import json
p = '$LIST'; d = json.load(open(p, encoding='utf-8')); d['$lab|$c'] = '$path'
json.dump(d, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)"
  rmdir $LIST.lock
}

MODELS="M2B=models/M2B_multi/final.zip A_s0=models/A_E26_s0_r2/final.zip B_s0=models/B_E26_s0/final.zip A_s1=models/A_E26_s1_r2/final.zip B_s1=models/B_E26_s1/final.zip"
jobs=()
for mm in $MODELS; do
  lab=${mm%%=*}; m=${mm#*=}
  for c in morning day night; do
    grep -q "\"$lab|$c\"" $LIST || jobs+=("$lab $m $c")
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
uv run python -m train.e26_final report > /dev/null 2>&1
echo "== $(ts) 전체 종료 → runs/e26/final_report.md"
