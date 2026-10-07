# 새 세션 인계 프롬프트 (2026-10-07 기준)

> 새 대화를 시작할 때 아래 블록을 그대로 붙여넣기. "지금 상황" 부분만 최신으로 고쳐서 쓰면 됨

```text
포켓몬 골드(한국어판) 강화학습 프로젝트(C:\pokemon_rl)를 이어서 하려고 해. 기록은 전부 폴더 안에 있어. 먼저 아래 파일들을 읽고 지금 상태를 파악한 다음 이어가줘.

## 먼저 읽을 것 (이 순서로)
1. progress/README.md — 전체 현황, 원칙, 마일스톤, 갱신 기록(맨 아래가 최신)
2. progress/step4_training.md — 실험 표, 알게 된 것, 다음 할 일, 도구 목록
3. notes/04_experiments.md — 실험별 가설, 판정 기준, 결과 (결과 칸이 빈 실험 = 아직 안 돌렸거나 진행 중). 특히 DEMO_TILE, CHAIN1 블록
4. notes/06_related_work.md — 다른 포켓몬 RL 프로젝트와 비교 (출처 포함)
5. progress/segments.md — 구간 구조, 알려진 문제
6. progress/RESUME.md — 학습 이어서 하기, 프로세스 정리

## 지금 상황 (2026-10-07 기준)
- 최종 목표: 전당등록. M1(29번도로 → 무궁시티) 달성
- **DEMO_TILE: 거리 보상 없이 M1 달성, 시드 2개 재현** (start / S2 / S3 = 시드0 88/96/92%, 시드1 80/96/92%. B2(거리 보상)는 96/84/96%)
  - 방법: 시연 경로 1개(BFS)로 역방향 커리큘럼 (경로 끝부터 시작, 잘하면 입구 쪽으로 당김) + 보상은 도착 +10만 + 1칸 조작(--move tile)
  - 결정적 요인: 조작 방식. 같은 설정에서 tap(다른 방향이면 몸만 돎) 0% → tile 88%
  - 비계: 시연 경로(BFS), 세이브 상태 시작, 전투 자동(A 연타), 마을 나가면 판 종료. RL: 화면 → 버튼 정책 전체 (U턴, 수풀 주머니 탈출, 풀숲 통과를 도착 보상만으로)
- **다음 실험 CHAIN1 + T1 (준비 완료, 아직 안 돌림)**
  - 연두마을 구간 `newbark` (집 1층 → 29번도로, BFS 24걸음, 시연 25칸) 추가
  - 학습 두 개 (각 10만, 15~25분 예상, 하나씩):
    uv run python -m train.train_ppo --env newbark_demo --reward goal_only --move tile --steps 100000 --name NB_scratch
    uv run python -m train.train_ppo --env newbark_demo --reward goal_only --move tile --steps 100000 --name NB_init29 --init models/DEMO_TILE_route29/final.zip
  - 끝나면 진단:
    - 연두마을: uv run python -m train.diagnose newbark models/NB_scratch/final.zip models/NB_init29/final.zip (전이: 전선이 24에 닿은 스텝도 TensorBoard에서 비교)
    - 집 1층 → 무궁시티 한 판: uv run python -m train.diagnose newbark_cherrygrove "chain:newbark=models/NB_scratch/final.zip,route29=models/DEMO_TILE_route29/final.zip" models/NB_init29/final.zip
    - 판정 기준은 notes/04 CHAIN1 블록에 이미 적어둠
- 보류: DEMO_GE (Go-Explore가 찾은 시연으로 BFS 비계 걷기. --env route29_demo_ge, 시연은 states/segments/route29/demo_goexplore.pkl)
- 이후 후보: 경로 옆 시작(옆으로 넓히기), MEM(짧은 기억, --memory 구현 완료), 전선 당기기 속도, 30번도로 구간 + 자동 구간 사슬(사람은 목표 맵 순서만 적고 나머지 자동, 구간마다 전이 효율 기록)
- 알려진 문제: 집 1층부터 한 번에 BFS를 돌리면 결과가 흔들림 (평가 구간은 compose로 우회). 정책이 최단 경로 한 줄만 알아서 줄을 벗어나면 벽을 밂. 다른 맵 일반화는 미측정

## 원칙 (중요)
- RL 중심: 목적은 "AI를 만드는 것". 규칙/BFS/세이브 스테이트는 RL을 가르치는 비계로만 쓰고, 하나씩 걷어내며 효과를 측정. 무엇을 RL이 했고 무엇이 비계인지 항상 명시
- 한 번에 하나만 바꾸기. 학습 전에 notes/04에 가설과 판정 기준을 먼저 적기 (정립 기준 같은 것도 결과 보기 전에)
- 평가는 학습 보상이 아니라 train.diagnose의 실제 진행 지표로. 25판 기준 ±10%p는 오차
- 하드코딩 최소화. 확장성 우선 (envs/segments.py 구간 구조)
- 전멸은 지표로만. **수풀/전투/전멸에 벌점 안 줌** (레벨업 안 하는 정책이 됨). 갇힘은 조작/관찰/커리큘럼으로 고침
- tile 조작이 기본 (--move tile). 진단/라이브는 models/<실험>/env_config.json을 자동으로 따름
- 정리하면서 가기: 실험/결정마다 같은 턴에 notes/04, progress/README.md(지금 하는 일, 마일스톤, 갱신 기록), progress/step4_training.md를 갱신

## 진행 방식
- 한국어로, "~임/~함" 식의 짧은 반말체. 명령조("~해봐") 말고 "~하면 됨", "~해보면 될 듯"
- 나는 강화학습 초보. 코드는 완성본으로 주되 왜 그렇게 하는지 주석 충분히. 설명은 "왜 필요한지, 앞 단계의 어떤 문제를 해결하는지" 흐름 위주로
- 내 결론에 무조건 동의하지 말고, 데이터로 반증되는 건 적극적으로 지적
- 학습(오래 걸림)은 내가 터미널에서 직접 돌림. 명령만 알려줘. 라이브 창은 약 1만 스텝 뒤 자동으로 뜸 (전선 위치에서 시작, 4판에 1번 실제 시작)
- 창 없는 짧은 작업(진단, 거리표, 짧은 시험)은 네가 직접 돌려도 됨. 학습과 동시에 무거운 작업은 피하기 (CPU 8코어, 학습 하나가 거의 다 씀)
- 백그라운드 대기 루프를 걸면 감시 대상이 죽어도 끝나게 짜기 (전에 영원히 도는 루프가 남았음)
- git은 내가 직접 함. 커밋하자는 말은 안 해도 됨. ROM/세이브가 올라갈 위험이 보이면 그것만 짚어줘
- uv만 사용 (uv run, uv add). pip 금지

## 환경 메모
- Windows, PyBoy 2.7, Gymnasium 1.3, SB3 2.9 PPO, torch CPU. tile 조작은 걸음당 40프레임이라 초당 30~150스텝 (전투 많으면 느림)
- TensorBoard: uv run tensorboard --logdir runs → http://localhost:6006 (game/demo_front, game/real_goal 위주)
- 라이브: uv run python -m train.live_watch <실험> --env <구간>_demo
```

## 쓸 때 팁
- 새 세션에서도 이 프로젝트 폴더의 기억(Claude 메모리)이 자동으로 같이 불러와짐. 그래도 위 프롬프트를 주면 첫 턴부터 정확히 이어감
- "지금 상황"의 숫자는 마지막으로 본 값이라, AC3 결과가 나온 뒤면 그 결과로 바꿔서 붙이면 됨
