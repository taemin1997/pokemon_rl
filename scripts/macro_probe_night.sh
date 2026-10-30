#!/bin/bash
# 10-19 밤: E26 최종 보고서가 나온 뒤 열린 문제 1 측정 (관찰만, 학습 없음)
# 실행: nohup bash scripts/macro_probe_night.sh > runs/e26/macro_probe/pipeline.log 2>&1 &
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
ts() { date '+%m-%d %H:%M:%S'; }

until grep -q "report 종료" runs/e26/main.log; do sleep 60; done
echo "== $(ts) E26 보고서 확인 → 측정 시작"

MODELS=$(uv run python -c "import json; print(' '.join(json.load(open('runs/e26/final_runs.json', encoding='utf-8'))['models'].values()))")
PARTS=$(uv run python -c "from train.selfrepair import M2B_PARTS; print(' '.join(M2B_PARTS))")
echo "모델: $MODELS"

# 1) 전이·복구 평가 상태: 최종 평가와 같은 판 수·시드·5분 상한, 3시간대
for seg in violet_pc_counter newbark_labdoor; do
  for c in morning day night; do
    echo "== $(ts) diag $seg $c"
    POKE_CLOCK=$c timeout 3600 uv run python -m train.macro_probe diag $seg $MODELS --runs 50 || echo "!! $seg $c 실패 code=$?"
  done
done
# 2) 학습 구간 16개: 학습에도 같은 비용이 숨어 있나 (낮만, 25판)
for seg in $PARTS; do
  echo "== $(ts) diag $seg day"
  POKE_CLOCK=day timeout 2400 uv run python -m train.macro_probe diag $seg $MODELS --runs 25 || echo "!! $seg 실패 code=$?"
done
# 3) 통합 실행: 모델마다 낮 5판 (출력은 파일로 — 파이프 막힘 없음)
for m in $MODELS; do
  echo "== $(ts) story $m day"
  POKE_CLOCK=day timeout 5400 uv run python -m train.macro_probe story $m --episodes 5 > "runs/e26/macro_probe/story_$(basename $(dirname $m)).log" 2>&1 || echo "!! story $m 실패 code=$?"
done
uv run python -m train.macro_probe report > /dev/null 2>&1
echo "== $(ts) 측정 종료 → runs/e26/macro_probe/report.md"
