#!/bin/bash
# 10-20 화면 버그 확인 시험: run_story --fresh-obs 있음/없음, A_s1_r2·M2B × 낮·밤 × 20판 (판정 3 인자·시드 그대로)
# 실행: nohup bash scripts/fresh_obs_test.sh > runs/e26/fresh_obs/main.log 2>&1 &
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
OUT=runs/e26/fresh_obs
mkdir -p $OUT
ts() { date '+%m-%d %H:%M:%S'; }
run() {   # 모델 시간대 플래그(off/on)
  local m=$1 c=$2 f=$3 name=$(basename $(dirname $1))_$2_$3
  local extra=""; [ "$f" = on ] && extra="--fresh-obs"
  echo "== $(ts) 시작 $name"
  POKE_CLOCK=$c timeout 14400 uv run python -m train.run_story $m --headless --episodes 20 --quiet \
    --targets "26,10;24,5;10,7" --no-heal-at 10,5 --max-steps 6000 --seed-offset 5000 $extra > $OUT/$name.log 2>&1
  echo "== $(ts) 끝 $name code=$? $(grep '저장:' $OUT/$name.log)"
}
# 코어 8개: 한 번에 4개씩 (같은 모델·시간대의 끔/켬을 같이 띄워 조건을 맞춤)
for m in models/A_E26_s1_r2/final.zip models/M2B_multi/final.zip; do
  for c in day night; do
    for f in off on; do run $m $c $f & done
  done
  wait
done
echo "== $(ts) 전체 종료"
