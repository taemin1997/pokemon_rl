# 06. 관련 사례 (다른 포켓몬 RL 프로젝트와 비교)

> 조사일: 2026-10-06. 출처를 직접 확인한 내용만 적음. 확인 못 한 숫자는 "확인 안 됨"으로 둠
> 목적: 우리 결과를 어디에 놓을지 (포트폴리오에서 "이 분야에서 이 정도 어려움은 정상인가"를 설명하는 근거)

## 한 줄 요약
포켓몬 RL은 어디서나 **길 찾기, 맴돌기, 보상 구멍**으로 고생함. 대부분 **촘촘한 탐험 보상 + 게임 단순화(스크립트) + 세이브 상태**로 해결했음. 우리 DEMO 계열은 그중 **보상을 가장 드물게(도착만) 준 조건**이라 더 어려운 쪽임.

## 사례별 정리

### 1. Peter Whidden — PokemonRedExperiments (2023~)
- 포켓몬 레드, 화면 입력 + 버튼 출력 PPO. 유튜브 영상으로 유명해짐
- 보상: 레벨업, 새 지역 탐험, 전투 승리, 체육관 등 **여러 항목을 섞은 보상**. 탐험 보상은 처음엔 화면 유사도(frame KNN), V2부터 **좌표 기반 탐험 보상**
- 결과: V2 기준 회색시티 → 블루시티(Cerulean)까지
- 보고된 어려움: 극단적인 보상 한 번이 행동에 오래 남음 (예: 한 번 전멸하면 포켓몬센터를 피하게 됨)
- 우리와 연결: 탐험 보상 = 우리 v1/A2의 새 칸 보상. 이상한 습관 = 우리 B의 "목표 앞 뜸 들이기", A의 reward hacking과 같은 종류
- 출처: [GitHub](https://github.com/PWhiddy/PokemonRedExperiments), [X 게시글](https://x.com/computerender/status/1711208834495058160), [80.lv 기사](https://80.lv/articles/ai-learns-to-play-pokemon-red)

### 2. Pleines, Addis, Rubinstein, Zimmer, Preuss, Whidden — "Pokemon Red via Reinforcement Learning" (arXiv 2025)
- 단순화한 환경 + PPO로 **블루시티(Cerulean) 완료**까지 가는 기준 에이전트
- 보상 항목 25개 이상, 관찰에 **가본 칸 지도(72×80)** 채널 추가 (다른 논문의 정리 기준)
- 어려움으로 꼽은 것: 멀티태스킹, 수만 걸음의 긴 호흡, **어려운 탐험**, 보상 구멍을 파고드는 행동
- 우리와 연결: 관찰 크기 72×80이 우리와 같음. 가본 칸 지도는 우리 "긴 기억" 후보와 같은 발상
- 출처: [arXiv 2502.19920](https://arxiv.org/abs/2502.19920)

### 3. David Rubinstein 등 — pokemonred_puffer / PokeRL 문서 (PufferLib, 2024~2025)
- **포켓몬 레드 클리어 성공** (1천만 파라미터 미만 정책). 단, 스크립트를 전부 끄고는 아직 안정적으로 못 깸 (작성자 표현)
- 남겨둔 단순화: 가방이 차면 아이템 버리기, 무한 돈, 괴력/파도타기/플래시/포켓몬피리 자동 사용 등. 개발 중에 쓰다 뺀 것: 풀베기 자동, 야생 전투 끄기, 엘리베이터 스크립트 등
- **스웜(swarming)**: 어떤 에이전트가 필수 목표를 달성하면 그 순간의 세이브 상태를 모든 에이전트가 불러옴. **Go-Explore에서 영감**. 대신 일반화를 해칠 수 있다고 작성자가 직접 적음
- 보상: 좌표 방문, 맵 번호, NPC, 비전머신 쓸 수 있는 자리 등 **탐험 보상 중심**. "탐험 최적화는 아주 강력하지만 조정이 어렵다", "보상 설계가 아주 중요하다"
- 사파리존은 스크립트 + 남은 걸음에 비례한 보상을 넣고 수천 번 시도해서 통과
- 속도: 엔지니어링으로 초당 최대 1만 스텝, 학습 한 번 1일 → 7시간
- 우리와 연결: 세이브 상태 공유 = 우리 커리큘럼(AC, DEMO)과 같은 계열. "단순화는 비계, 하나씩 걷어냄"도 우리 원칙과 같음. 규모는 우리(CPU, 초당 60~300)의 수십~수백 배
- 출처: [GitHub](https://github.com/drubinstein/pokemonred_puffer), [결과](https://drubinstein.github.io/pokerl/docs/chapter-4/results/), [환경 설정](https://drubinstein.github.io/pokerl/docs/chapter-2/env-setup/), [스웜](https://drubinstein.github.io/pokerl/docs/chapter-3/swarm/), [GIGAZINE 기사](https://gigazine.net/gsc_news/en/20250310-pokemon-rl-edition/)

### 4. Mudireddy, Patibandla — "PokeRL: Reinforcement Learning for Pokémon Red" (arXiv 2026)
- **우리와 조건이 가장 비슷함**: SB3 PPO, 환경 4개, CNN 약 107만 파라미터 (우리 CnnPolicy 약 106만)
- 행동: 방향 4 + A + B + 아무것도 안 함. START/SELECT 뺌 (우리와 같은 이유)
- **"한 번 누르면 한 칸" 이동 추상화** (두 번 누르기) → 우리 TILE과 같은 문제를 같은 방식으로 해결
- 맵별 가본 칸 지도 관찰 → 고유 칸 방문 +40.6%. 맴돌기 방지 장치(벌점 등)로 맴도는 판 41.2% → 4.7%
- 보상: 3단계 (작은 이동/새 칸 +0.2~1.0, 맵 이동 +5~10, 큰 목표 +20~50) + 맴돌기 벌점
- 과제별 세이브 상태로 따로 학습 (집 나가기, 풀숲까지, 첫 라이벌전)
- 결과: 집 나가기 15만 스텝에 약 65%, 풀숲 도달 50만 스텝에 약 60%, 라이벌전 승률 약 50%
- 우리와 연결: 비슷한 규모로 **훨씬 짧은 과제**(마을 안 몇십 걸음)를 촘촘한 보상으로 풀었는데 60% 수준. 우리 29번도로(103걸음, 도착 보상만)는 이보다 어려운 과제
- 출처: [arXiv 2604.10812](https://arxiv.org/html/2604.10812v1)

### 5. LLM 에이전트 — Claude Plays Pokémon (Anthropic, 2025), Gemini Plays Pokémon
- RL이 아니라 언어 모델이 화면을 보고 버튼을 고름
- 길 찾기에서 같은 문제: 달맞이산에서 수십 시간 갇힘, 같은 곳을 맴돌기, **있을 수 없는 출구를 찾아 벽을 계속 밀기**, 메모를 지웠다가 다시 헤매기
- 우리와 연결: 우리 에이전트가 (22,11)에서 같은 벽을 47번 미는 것과 같은 현상. "막힌 걸 기억하지 못함"은 방식과 상관없는 공통 난제
- 출처: [TechCrunch](https://techcrunch.com/2025/02/25/anthropics-claude-ai-is-playing-pokemon-on-twitch-slowly), [LessWrong 분석](https://www.lesswrong.com/posts/HyD3khBjnBhvsp8Gb/so-how-well-is-claude-playing-pokemon)

### 6. 2세대(금/은/크리스탈)
- 크리스탈 RL 프로젝트가 있음 ([PokemonCrystalExperiments](https://github.com/RahifMansoor/PokemonCrystalExperiments)), 골드 실험 글도 있음 ([Medium](https://medium.com/ordina-data/reinforcement-learning-201-ai-plays-all-the-pok%C3%A9mon-5957da084af9)). 규모나 결과는 확인 안 함
- **한국어판** 사례는 찾지 못함 → 메모리 주소를 직접 찾고 검증한 것(step2) 자체가 차별점

## 우리 방법의 원래 논문
| 우리 실험 | 원래 아이디어 | 출처 |
|---|---|---|
| RC, RC2 (목표 근처부터 시작, 넓혀감) | 역방향 커리큘럼 (Florensa 등 2017) | [arXiv 1707.05300](https://arxiv.org/abs/1707.05300) |
| AC, AC2, AC3 (가본 자리 저장 → 거기서 시작) | Go-Explore (Ecoffet 등 2019/2021) | [arXiv 1901.10995](https://arxiv.org/abs/1901.10995) |
| DEMO (시연 경로 끝부터 시작, 앞으로 당김) | 한 번의 시연으로 몬테주마 배우기 (Salimans & Chen 2018). Go-Explore의 "튼튼하게 만들기" 단계도 같은 방식 | [arXiv 1812.03381](https://arxiv.org/abs/1812.03381) |

## 비교표

| | 보상 | 단순화/비계 | 규모 | 어디까지 |
|---|---|---|---|---|
| Whidden | 여러 항목 + 탐험 | 적음 | 큼 (수만 시간 플레이 분량) | 블루시티 |
| Pleines 등 | 25개 이상 항목 | 단순화 환경 | 확인 안 함 | 블루시티 완료 |
| Rubinstein 등 | 탐험 중심 + 이벤트 | 스크립트 여러 개, 스웜 | 매우 큼 (초당 최대 1만 스텝, GPU) | **클리어** |
| PokeRL | 3단계 촘촘한 보상 + 맴돌기 벌점 | 과제별 세이브, 1칸 이동 | 우리와 비슷 (SB3, 환경 4개) | 첫 라이벌전 |
| **우리 B2** | 도착 + **BFS 거리** | 전투 자동, 마을 차단 | CPU, 환경 6개, 25만 | 무궁시티 88~96% |
| **우리 DEMO_TILE** | **도착만** | 시연 경로 1개(BFS), 전투 자동, 1칸 이동 | CPU, 환경 6개, 25만 | 무궁시티 (진단 대기) |

## 우리 프로젝트에 주는 의미
1. **길 찾기에서 고생하는 건 정상**. 모든 사례가 맴돌기, 벽 밀기, 보상 구멍을 겪음
2. 다른 사례는 거의 다 **촘촘한 보상**을 씀. 도착 보상만으로 103걸음 + U턴을 넘은 건 상대적으로 어려운 조건의 결과 → 포트폴리오에서 강조할 점 (단, 시연 경로라는 비계는 같이 명시)
3. 같은 문제를 같은 방식으로 푼 곳이 있음 → 방향이 맞다는 근거
   - 1칸 이동 = PokeRL의 이동 추상화
   - 세이브 상태 커리큘럼 = Rubinstein의 스웜, Go-Explore
   - 가본 칸 지도 = Pleines 등, PokeRL → 우리 "긴 기억" 후보의 근거
4. 끝까지 간 사례(Rubinstein)도 **스크립트를 다 끄고는 아직 불안정** → 전당등록까지 "비계를 명시하고 하나씩 걷어내기"가 현실적인 길이라는 우리 원칙과 같은 결론
5. Rubinstein이 직접 적은 "스웜(세이브 상태 공유)은 일반화를 해칠 수 있다" → 우리 DEMO의 "경로 위 상태를 외웠나" 걱정과 같은 지점. 일반화 시험이 필요한 이유
