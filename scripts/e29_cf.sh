#!/bin/bash
# E29 반사실 실험 (학습 없음): 사례 K1~K3 × 조건 A·B·C·C_ctrl × 50판, 시계 고정
#   A 옛 정책 / B 새 정책 (칸 도달·KL만 기록) / C 새 정책 + 뒤집힌 칸 K에서만 옛 정책 / C_ctrl 대조 칸에서만 옛 정책
#   길목 목록: runs/e29/K*.json (train/e29_critical.py, 결과 보기 전 규칙). 기준: notes/experiments/E29_CRITICAL.md
# 실행: nohup bash scripts/e29_cf.sh > runs/e29/cf.log 2>&1 &
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 POKE_CLOCK=lock
ts() { date '+%m-%d %H:%M:%S'; }
OLD=models/M2B_multi/final.zip
LIST=runs/e29/cf.json
echo "{" > $LIST.tmp
first=1
add() { [ $first = 1 ] || echo "," >> $LIST.tmp; first=0; echo "\"$1\": \"$2\"" >> $LIST.tmp; }
diag() {   # 키 구간 모델 [환경변수...]
  local key=$1 seg=$2 m=$3; shift 3
  out=$(env "$@" uv run python -m train.diagnose $seg $m --runs 50 --workers 6 2>runs/e29/cf_${key}.err)
  path=$(echo "$out" | grep '저장:' | tail -1 | sed 's/.*저장: //' | tr '\\' '/')
  echo "== $(ts) 끝 $key $path"
  add "$key" "$path"
}
echo "== $(ts) 시작"
diag "A|route29" route29 $OLD
diag "A|cherrygrove" cherrygrove $OLD
for c in "K1 route29 models/A_E26_s0_r1/final.zip" "K2 route29 models/A_E26_s1_r1/final.zip" "K3 cherrygrove models/A_E26_s0_r2/final.zip"; do
  set -- $c; k=$1 seg=$2 new=$3
  diag "$k|B" $seg $new POKE_REF_POLICY=$OLD POKE_OVERRIDE_TILES=runs/e29/$k.json:K POKE_OVERRIDE_DRY=1
  diag "$k|C" $seg $new POKE_REF_POLICY=$OLD POKE_OVERRIDE_TILES=runs/e29/$k.json:K
  diag "$k|C_ctrl" $seg $new POKE_REF_POLICY=$OLD POKE_OVERRIDE_TILES=runs/e29/$k.json:K_ctrl
done
echo "}" >> $LIST.tmp
mv $LIST.tmp $LIST
echo "== $(ts) 전체 종료 → $LIST"
