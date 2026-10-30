#!/bin/bash
# E30 PRESERVE 본 실행 (notes/experiments/E30_PRESERVE.md, 10-21 사용자 확정)
#   4군 × 2시드 × PPO 10만: A 기준선(E26 r1 그대로) / B 새 구간 가중치 1 / C 전체 KL β=1 / D 확신 상태 517칸 KL β=1
#   학습 2개씩 동시 → 평가 (E-lab 50판 첫 화면 1프레임, 16구간 25판, 시계 lock, M2B 같이)
#   끝난 학습(models/E30_*/final.zip 있음)은 건너뜀 → 끊기면 같은 명령으로 재개
# 실행: nohup bash scripts/e30_run.sh >> runs/e30/run.log 2>&1 &
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 POKE_CLOCK=lock
ts() { date '+%m-%d %H:%M:%S'; }
FLAGS="--reward goal_only --move tile --goal --hidden 256,256 --fast-battle --auto-event --no-live --init models/M2B_multi/final.zip --steps 100000"
OLD=models/M2B_multi/final.zip
train() {   # 군 시드
  local g=$1 s=$2 name=E30_$1_s$2
  [ -f models/$name/final.zip ] && { echo "== $(ts) 건너뜀 $name (있음)"; return; }
  local preset=e30_s${s}_base extra=""
  [ $g = B ] && preset=e30_s${s}_w1
  [ $g = C ] && extra="--anchor-model $OLD --anchor-coef 1 --anchor-mode rollout"
  [ $g = D ] && extra="--anchor-model $OLD --anchor-coef 1 --anchor-mode states --anchor-states runs/e30/anchor_states.npz"
  echo "== $(ts) 학습 시작 $name ($preset $extra)"
  POKE_EXTRA_SEGMENTS=$(pwd)/states/e26/a/s$s/segments.json POKE_EXTRA_PRESETS=$(pwd)/runs/e30/presets_s$s.json \
    uv run python -m train.train_ppo --env $preset $FLAGS --seed $s --name $name $extra > runs/e30/train_$name.log 2>&1
  echo "== $(ts) 학습 끝 $name code=$? $( [ -f models/$name/final.zip ] && echo ok || echo 없음)"
}
echo "== $(ts) 시작"
for pair in "A B" "C D"; do
  for g in $pair; do
    for s in 0 1; do
      while [ $(jobs -rp | wc -l) -ge 2 ]; do sleep 20; done
      train $g $s &
      sleep 5
    done
  done
done
wait
M="$OLD"
for g in A B C D; do for s in 0 1; do M="$M models/E30_${g}_s$s/final.zip"; done; done
echo "== $(ts) 평가 시작"
out=$(POKE_FRESH_FIRST=1 uv run python -m train.diagnose newbark_labdoor $M --runs 50 --workers 6 2>runs/e30/eval_elab.err)
echo "== $(ts) 평가 끝 elab $(echo "$out" | grep '저장:' | tail -1)"
for seg in house2f newbark route29 cherrygrove cherrygrove_pc cherrygrove_pc_exit route30 route30_rev cherrygrove_rev route29_rev newbark_lab route30_north route31 violet_gym violet_pc violet_pc_exit; do
  out=$(uv run python -m train.diagnose $seg $M --runs 25 --workers 6 2>runs/e30/eval_$seg.err)
  echo "== $(ts) 평가 끝 $seg $(echo "$out" | grep '저장:' | tail -1)"
done
echo "== $(ts) 전체 종료"
