# 커리큘럼 난수 시드 미연결 + resume이 live.zip을 고를 수 있음 (외부 검토)

- 날짜: 2026-10-08
- 분류: 버그

## 내용
DemoReverseEnv의 self.rng가 시드 없는 default_rng → 같은 --seed라도 시작 위치 순서가 달랐음. self.np_random(gymnasium 시드)으로 연결. 또 --resume latest가 live.zip을 고를 수 있었고, 강제 종료 시 모델(최대 5만 스텝 전)과 전선(최신)이 어긋남 → 정식 체크포인트만 + demo_ptr_history.jsonl로 그 스텝 시점 전선 복원. 지난 결과는 '같은 조건 성능 재현'으로 유효, GOAL1까지 예전 조건. 시험: 같은 시드 시작 순서 일치, 다른 시드 다름

## 갱신 기록
- 2026-10-08: 파일 생성
