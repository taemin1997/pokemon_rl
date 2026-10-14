#!/bin/bash
# E26 3~6단계 본 실행 (10-14): 트랙 B 수집·학습 → 트랙 A 자동 복구 → 최종 평가
# 실행: nohup bash scripts/e26_main.sh > runs/e26/main.log 2>&1 &
# 끊기면 다시 실행: 끝난 단계(결과 파일 있음)는 건너뜀. 학습은 한 번에 하나 (CPU)
cd /c/pokemon_rl
export PYTHONIOENCODING=utf-8
stamp () { echo "== $(date '+%m-%d %H:%M:%S') $*"; }
RULECAP=${RULECAP:---detect-rule-cap}      # 사용자 확인 항목 (규칙 상한 끊김도 감지). 빼려면 RULECAP= 로
ELAB=${ELAB:-keep}                          # 사용자 확인 항목 (B 수집에서 E-lab과 의미 키 같은 상태: keep|drop)
stamp "E26 본 실행 시작 (코드 $(git rev-parse --short HEAD), 수정 파일 $(git status --porcelain --untracked-files=no | wc -l)개, rule_cap=${RULECAP:-없음}, elab=$ELAB)"

# 3단계: 트랙 B
[ -f states/e26/b/harvest.json ] || { stamp "B 수집 (통합 실행 20판, 시드 2000~)"; uv run python -m train.e26_trackb harvest; }
[ -f states/e26/b/presets.json ] || { stamp "B 목록 (중복·누출 필터)"; uv run python -m train.e26_trackb build --elab-semantic $ELAB; }
for s in 0 1; do
  [ -f models/B_E26_s$s/final.zip ] || { stamp "B 학습 시드 $s (15만)"; uv run python -m train.e26_trackb train --seed $s; }
done

# 4단계: 트랙 A
for s in 0 1; do
  if ! grep -q '"finished"' states/e26/a/s$s/selfrepair.json 2>/dev/null; then
    stamp "A 자동 복구 시드 $s"
    uv run python -m train.selfrepair --seed $s $RULECAP
  fi
done

# 누출 검사 (A·B 학습 데이터 폴더 포함)
stamp "누출 검사 (A·B 데이터 포함)"
uv run python -m train.e26_eval_states leak states/e26/a states/e26/b
cp runs/e26/stage2_leak.json runs/e26/final_leak.json

# 6단계: 최종 평가
stamp "최종 평가"
uv run python -m train.e26_final eval
stamp "보고서"
uv run python -m train.e26_final report
stamp "끝"
