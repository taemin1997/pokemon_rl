#!/bin/bash
# E32 NIGHT 1단계 (notes/experiments/E32_NIGHT.md, 10-23 사용자 확정): 밤 약점 구간 찾기
#   모델 3개 (M2B, E30_D_s0, E30_D_s1) × 18구간 (16 + E-lab + E-violet) × 아침·낮·밤 × 50판 (시드 0~49)
#   끝난 (구간, 시계)는 runs/e32/stage1.json에 기록 → 끊기면 같은 명령으로 재개
# 실행: nohup bash scripts/e32_stage1.sh >> runs/e32/stage1.log 2>&1 &
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
ts() { date '+%m-%d %H:%M:%S'; }
M="models/M2B_multi/final.zip models/E30_D_s0/final.zip models/E30_D_s1/final.zip"
LIST=runs/e32/stage1.json
[ -f $LIST ] || echo "{}" > $LIST
echo "== $(ts) 시작"
for c in morning day night; do
  for seg in newbark_labdoor violet_pc_counter house2f newbark route29 cherrygrove cherrygrove_pc cherrygrove_pc_exit route30 route30_rev cherrygrove_rev route29_rev newbark_lab route30_north route31 violet_gym violet_pc violet_pc_exit; do
    if grep -q "\"$seg|$c\"" $LIST; then echo "== $(ts) 건너뜀 $seg $c"; continue; fi
    out=$(POKE_CLOCK=$c uv run python -m train.diagnose $seg $M --runs 50 --workers 6 2>runs/e32/s1_${seg}_${c}.err)
    f=$(echo "$out" | grep '저장:' | tail -1 | sed 's/저장: //' | tr -d '\r')
    if [ -n "$f" ]; then
      uv run python -c "import json,sys; p=sys.argv[1]; d=json.load(open(p)); d[sys.argv[2]]=sys.argv[3]; json.dump(d,open(p,'w'),indent=1)" $LIST "$seg|$c" "$f"
      rm -f runs/e32/s1_${seg}_${c}.err
      echo "== $(ts) 끝 $seg $c $f"
    else
      echo "== $(ts) 실패 $seg $c (runs/e32/s1_${seg}_${c}.err)"
    fi
  done
done
echo "== $(ts) 전체 종료"
