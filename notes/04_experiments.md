# 04. 보상 설계 실험 기록

## 공통 조건 (바꾸면 여기에 적기)
- 시작 상태: `states/start.state` (리아코 받은 직후, 집 1층 24:6 (9,2))
- 행동: 상하좌우 + A + B (START/SELECT 제외), 1스텝 = 24프레임
- 관찰: 화면 흑백 72x80
- 에피소드: 최대 2048스텝, 배지 1개 획득 시 종료
- PPO: CnnPolicy, 환경 6개, n_steps 512, batch 256, epochs 4, gamma 0.995, ent_coef 0.01, lr 2.5e-4, seed 0
- 속도: 학습 포함 약 230스텝/초 → 100만 스텝 약 1.2시간 (CPU)

- **10-08부터 (조건 변경)**: 커리큘럼 시작 위치 난수가 시드에 연결됨 (예전엔 시드 없음 → 같은 --seed라도 시작 위치 순서가 매번 달랐음). GOAL1(E21)까지는 예전 조건, 그 뒤 실험(시드 재현 포함)은 새 조건. `--resume latest`는 정식 체크포인트만 고르고 전선 상태도 그 스텝 시점으로 복원
- **신경망**: 기본은 은닉층 없음 (CnnPolicy와 같은 모양). 10-08부터 `--hidden 256,256` 옵션 (GOAL1 교훈: 새 입력이 화면과 엮여야 하면 은닉층 필요)
- 보상 `goal_only`에도 기본 배지 보상(+100)은 남아 있음 → 배지가 있는 구간(체육관)에 들어갈 때 명시적으로 정리

## 기준점: 랜덤 행동
| 항목 | 값 |
|---|---|
| 1000스텝 보상 | 3.6 |
| 탐험 칸 | 37 |
| 맵 | 1 (집 밖으로 못 나감) |

- 주의: 위 값은 1000스텝 기준이라 2048스텝 에피소드와 직접 비교가 안 됨
- 실제 기준점은 v1 학습 첫 기록(12k 스텝, 거의 랜덤 상태)인 **visited_maps 2.83**으로 보는 게 맞음
  (집 1층, 2층, 마을 정도는 2048스텝이면 랜덤으로도 들어감)

## 실험 목록 (실험마다 파일 하나: `notes/experiments/`)

> 새 실험은 `notes/experiments/E<번호>_<이름>.md` 새 파일 (가설과 판정 기준을 학습 전에 먼저). 아래 표에 한 줄 추가
> 10-08 이전 한 파일 버전: `notes/archive/04_experiments_until_2026-10-08.md`

| # | 실험 | 바꾼 것 | 결과 |
|---|---|---|---|
| E01 | [v1_explore](experiments/E01_v1_explore.md) | 탐험 보상, 집 시작 | ✅ 마을까지, 550k부터 정체 |
| E02 | [v2_ent003](experiments/E02_v2_ent003.md) | 엔트로피 0.03 | ✅ 못 넘음 → 탐험 부족은 주원인 아님 |
| E03 | [A_route29](experiments/E03_A_route29.md) | 29번도로 시작, 이탈 실패, 전투 자동 | ❌ reward hacking (무효) |
| E04 | [A2_route29](experiments/E04_A2_route29.md) | A + 버그 수정 | ✅ 풀숲 탈출 100%, 무궁 0% |
| E05 | [B_route29](experiments/E05_B_route29.md) | + 거리 보상 | ✅ 무궁 64~72% |
| E06 | [A3_route29](experiments/E06_A3_route29.md) | A2 + 도착 보너스 | ✅ 무궁 0% (결과만 칭찬하면 신호 없음) |
| E07 | [B2_route29](experiments/E07_B2_route29.md) | 보상 역할 정리 | ✅ 무궁 88~96% (거리 보상 기준선) |
| E08 | [RC_route29](experiments/E08_RC_route29.md) | 역방향 커리큘럼 (무작위 확장) | ❌ 전선 9걸음 정체 |
| E09 | [RC2_route29](experiments/E09_RC2_route29.md) | RC 개선 | ❌ 전선 5걸음 |
| E10 | [AC_route29](experiments/E10_AC_route29.md) | Go-Explore식 아카이브 | ❌ 너무 빨리 넓어짐 |
| E11 | [AC2_route29](experiments/E11_AC2_route29.md) | 잘하는 자리만 아카이브 | ❌ 전선 4.5 |
| E12 | [AC3_route29](experiments/E12_AC3_route29.md) | 공간 전선 | ⚠️ U턴(약 50)에서 정체 |
| E13 | [DEMO_route29](experiments/E13_DEMO_route29.md) | 시연 경로 따라 전선 | ⚠️ 전선 76, 입구 0% (tap 회전) |
| E14 | [DEMO_TILE_route29](experiments/E14_DEMO_TILE_route29.md) | DEMO + 1칸 조작 | ✅ **88~100%, 시드 2개** (M1 거리 보상 없이) |
| E15 | [DEMO_TILE_MEM](experiments/E15_DEMO_TILE_MEM_route29.md) | DEMO_TILE + 기억 | — 안 돌림 (E20 LAT1_MEM으로 대체) |
| E16 | [DEMO_GE_route29](experiments/E16_DEMO_GE_route29.md) | 시연을 Go-Explore 경로로 | ⏸ 보류 (경로는 찾아둠) |
| E17 | [CHAIN1 + T1](experiments/E17_CHAIN1.md) | 연두마을 구간, 이어 붙이기, 가중치 이어받기 | ✅ 이어 붙이기 96% / 이어받기는 부정적 전이 + 잊음 |
| E18 | [OFFPATH1](experiments/E18_OFFPATH1.md) | (진단) 경로 밖 시작, 교란 | ✅ 3칸+ 급락 (B2도), 1~3칸 교란은 회복 |
| E19 | [LAT1_route29](experiments/E19_LAT1_route29.md) | 경로 옆 시작 + 폭 커리큘럼 | ✅ **빼둔 지역 77~84%, 시드 2개** (M1 대표) |
| E20 | [LAT1_MEM_route29](experiments/E20_LAT1_MEM_route29.md) | LAT1 + 짧은 기억 | ✅ 혼동 통로 6/12, 벽 고집 그대로 |
| E21 | [GOAL1_route29](experiments/E21_GOAL1.md) | 목표 정보 + 양방향 (정책 하나) | ❌ 정방향 0%, 목표 사용 출렁임 → 원인: 은닉층 없는 선형 결합 |
| E25 | [NEWMAP](experiments/E25_NEWMAP.md) | **[A]** 새 맵을 Go-Explore 경로로 학습, 사전학습/랜덤 초기 비교 | ✅ Go-Explore 8/8 경로 발견. 고정 출발 100%; 경로 밖은 랜덤 초기 쪽이 우세 |
| E24 | [ZS30](experiments/E24_ZS30.md) | **[A]** (학습 없음) GOAL2를 처음 보는 30번도로에 | ❌ 0% (random 이하): 능력이 학습한 맵에 묶임 |
| E23 | [M2A_multi](experiments/E23_M2A.md) | **[B]** 여러 구간을 정책 하나로, route30 보강 | ✅ M2a 11구간 실패 ≤4% (시드 0), 시드 1은 0% |
| E27 | [M2B_multi](experiments/E27_M2B.md) | **[B]** M2a 11 + M2b 5구간, M2A_r30에서 이어서, auto_event | ⚠️ 구간 성능·보존 통과, 통합 8/20·9/20 미달 (연구소 경계 상태/규칙 실패) |
| E26 | [SELFREPAIR](experiments/E26_SELFREPAIR.md) | **[A]** 실패 감지 → 재탐색(Go-Explore) → 재학습 → 재도전 (감독기), 첫 무대 M2b | 🔄 설계 확정, 1단계 재현성 검증 완료. 2단계 기준선 전 |
| E22 | [GOAL2_route29](experiments/E22_GOAL2.md) | GOAL1 + 은닉층 256-256 | ✅ **정방향 96~100%, 반대 100% (시드 2개)**, 빼둔 지역 71~100% (시드 편차 큼) |
