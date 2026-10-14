# 모델 카드 목록

> 모델마다 파일 하나. 새 모델은 `progress/make_docs.py`의 MODELS에 한 줄 추가 후 실행 (있는 카드는 안 덮어씀)

| 모델 | 시작 | 상태 | 결과 |
|---|---|---|---|
| [v1_explore](v1_explore.md) | 10-02 | ✅ 1M | 시작 근처 맵 7개, 550k부터 정체 |
| [v2_ent003](v2_ent003.md) | 10-02 | ✅ 1M | 엔트로피 유지, 못 넘음 |
| [A_route29](A_route29.md) | 10-02 | ❌ | 무효 |
| [A2_route29](A2_route29.md) | 10-02 | ✅ 250k | 풀숲 탈출 0 → 100%, 무궁 0% |
| [A3_route29](A3_route29.md) | 10-06 | ✅ 250k | 무궁 0% |
| [B_route29](B_route29.md) | 10-05 | ✅ 250k | 무궁 64~72% |
| [B2_route29](B2_route29.md) | 10-06 | ✅ 250k | 무궁 88~96%, 약 140걸음 |
| [RC_route29](RC_route29.md) | 10-06 | ❌ 67k 중단 | 전선 9걸음 정체 |
| [RC2_route29](RC2_route29.md) | 10-06 | ❌ 101k | 전선 5걸음 |
| [AC_route29](AC_route29.md) | 10-06 | ❌ 77k 중단 | 도착 67 → 14%, 빙글빙글 |
| [AC2_route29](AC2_route29.md) | 10-06 | ❌ | 전선 4.5, 입구 0% |
| [AC3_route29](AC3_route29.md) | 10-06 | ⚠️ 250k | U턴(약 50)에서 정체, 입구 0% |
| [DEMO_route29](DEMO_route29.md) | 10-06 | ⚠️ 250k | 전선 76, 입구 0% |
| [DEMO_TILE_route29](DEMO_TILE_route29.md) | 10-06 | ✅ 250k | 무궁 88/92/100% |
| [DEMO_TILE_route29_seed1](DEMO_TILE_route29_seed1.md) | 10-06 | ✅ 250k | 80/96/92% |
| [NB_scratch](NB_scratch.md) | 10-07 | ✅ 100k | 100%, 24걸음 (최단) |
| [NB_init29](NB_init29.md) | 10-07 | ⚠️ 7.4만 (프로세스 꺼짐) | 5만에 24% (scratch 100%), 끝까지 28% |
| [LAT1_route29](LAT1_route29.md) | 10-07 | ✅ 300k | 경로 위 96/100/100%, 빼둔 지역 79% |
| [LAT1_route29_seed1](LAT1_route29_seed1.md) | 10-07 | ✅ 300k | 빼둔 지역 80%, 경로 위 96/100/92% |
| [LAT1_MEM_route29](LAT1_MEM_route29.md) | 10-07 | ✅ 300k | 혼동 통로 50% (LAT1 8%), 빼둔 지역 89% |
| [GOAL1_route29](GOAL1_route29.md) | 10-08 | ❌ 500k | 정방향 0%, 반대 80%, 분기점 바뀜 18% |
| [GOAL2_route29](GOAL2_route29.md) | 10-08 | ✅ 500k | 정방향 96%, 반대 100%, 빼둔 지역/혼동 통로 100% |
| [GOAL2_route29_seed1](GOAL2_route29_seed1.md) | 10-08 | ✅ 500k | 정방향 96/100/96%, 반대 100%, 빼둔 지역 71%, 10칸+ 69% |
| [A_NM1B_route30rev](A_NM1B_route30rev.md) | 10-08 | ✅ 200k | 고정 출발 100% (10만부터), 경로 밖 1-2/3-5/6-9/10+ = 100/88/81/34% |
| [A_NMD_route30rev](A_NMD_route30rev.md) | 10-08 | ✅ 200k | 고정 출발 100% (10만부터), 경로 밖 100/88/88/100% (전체 94%) |
| [A_NM1B_route30rev_seed1](A_NM1B_route30rev_seed1.md) | 10-09 | ✅ 200k | 고정 출발 100% (5만부터), 경로 밖 100/88/84/91% (전체 91%) |
| [A_NM3_route30rev](A_NM3_route30rev.md) | 10-09 | ✅ 200k | 고정 출발 100% (5만부터), 경로 밖 100% 전부 → 전이 효과 안 보임 |
| [A_NM1B_cherrygroverev](A_NM1B_cherrygroverev.md) | 10-09 | ✅ 200k | 고정 출발 100% (30걸음), 경로 밖 30% |
| [A_NM1B_cherrygroverev_seed1](A_NM1B_cherrygroverev_seed1.md) | 10-09 | ✅ 200k | 고정 출발 100% (30걸음), 경로 밖 38% |
| [M2A_multi](M2A_multi.md) | 10-09 | ⚠️ 2M | 11구간 중 10개 100%, route30 48~56% (출발 칸에서 무궁시티로 되돌아감) |
| [M2A_r30](M2A_r30.md) | 10-10 | ✅ 300k | 11구간 모두 길 찾기 실패 ≤ 4% (route30 90%, 실패는 전멸뿐) |
| [A_NM3_cherrygroverev](A_NM3_cherrygroverev.md) | 10-10 | ✅ 200k | 고정 출발 100%, 경로 밖 88% (GOAL2 초기 30~38%) |
| [A_NM3_route30rev_seed1](A_NM3_route30rev_seed1.md) | 10-10 | ✅ 200k | 고정 출발 100%, 경로 밖 100% |
| [M2A_multi_seed1](M2A_multi_seed1.md) | 10-10 | ✅ 2M | M2A_r30_seed1의 재료 (완료) |
| [M2A_r30_seed1](M2A_r30_seed1.md) | 10-10 | ✅ 300k | 11구간 길 찾기 실패 0% (실패는 전멸뿐), 통합 실행 19/20 |
| [M2B_multi](M2B_multi.md) | 10-10 | ⚠️ 1M 완료 | 구간 성능·보존 통과, 통합 8/20·9/20 미달 |
