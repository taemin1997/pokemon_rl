#!/bin/bash
# E32 NIGHT 2·3단계 (notes/experiments/E32_NIGHT.md): 약점 구간마다 2×2 (게임 시간 × 정책이 보는 색) + 걸음 기록
#   네 칸 모두 다시 그린 화면 (10-23 설계 밖 변경: 다시 그리기의 애니메이션 차이를 네 칸에 똑같이)
#   MM = 아침 게임·아침 색 / NN = 밤·밤 / NM = 밤 게임·아침 색 / MN = 아침 게임·밤 색. 50판, POKE_TRACE=1 (3단계 처음 갈라지는 걸음)
#   약점 구간 = 1단계 집계(train.e32_report)에서 어느 모델이든 Δ ≥ +10%p인 구간. 끝난 칸은 runs/e32/stage2.json → 재개 가능
# 실행: nohup bash scripts/e32_stage2.sh >> runs/e32/stage2.log 2>&1 &
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 POKE_TRACE=1
ts() { date '+%m-%d %H:%M:%S'; }
M="models/M2B_multi/final.zip models/E30_D_s0/final.zip models/E30_D_s1/final.zip"
LIST=runs/e32/stage2.json
[ -f $LIST ] || echo "{}" > $LIST
SEGS=$(uv run python -c "
from train.e32_report import stage1
_, _, weak = stage1()
print(' '.join(dict.fromkeys(s for v in weak.values() for s in v)))" 2>/dev/null | tr -d '\r')
echo "== $(ts) 시작 (약점 구간: ${SEGS:-없음})"
for seg in $SEGS; do
  for cell in MM NN NM MN; do
    if grep -q "\"$seg|$cell\"" $LIST; then echo "== $(ts) 건너뜀 $seg $cell"; continue; fi
    g=$([ ${cell:0:1} = M ] && echo morning || echo night); o=$([ ${cell:1:1} = M ] && echo morning || echo night)
    out=$(POKE_CLOCK=$g POKE_OBS_CLOCK=$o uv run python -m train.diagnose $seg $M --runs 50 --workers 6 2>runs/e32/s2_${seg}_${cell}.err)
    f=$(echo "$out" | grep '저장:' | tail -1 | sed 's/저장: //' | tr -d '\r')
    if [ -n "$f" ]; then
      uv run python -c "import json,sys; p=sys.argv[1]; d=json.load(open(p)); d[sys.argv[2]]=sys.argv[3]; json.dump(d,open(p,'w'),indent=1)" $LIST "$seg|$cell" "$f"
      rm -f runs/e32/s2_${seg}_${cell}.err
      echo "== $(ts) 끝 $seg $cell $f"
    else
      echo "== $(ts) 실패 $seg $cell (runs/e32/s2_${seg}_${cell}.err)"
    fi
  done
done
echo "== $(ts) 전체 종료"
