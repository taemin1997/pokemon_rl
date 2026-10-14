# pokemon_rl — 포켓몬 금 강화학습

포켓몬 금(한국어판, 게임보이 컬러)을 무대로 한 강화학습 연구·포트폴리오 프로젝트.

- **트랙 B (진행)**: 목표 맵을 입력으로 받는 길 찾기 정책 하나(PPO)로 16개 구간을 이동하고, 상위 계획·회복 규칙·이벤트/전투 매크로와 묶어 스토리를 진행 (현재 M2: 첫 배지 준비)
- **트랙 A (연구)**: 처음 보는 구간에서 실패를 스스로 감지하고 탐색(Go-Explore)·학습으로 복구하는 자동화 (실험 E26)
- 사람이 만든 부분(BFS 시연, 구간 순서, 매크로, 세이브 상태 시작)은 결과마다 명시하고 하나씩 줄여 가는 중. "AI가 스스로 플레이한다"는 아직 아님

자세한 현황: [progress/README.md](progress/README.md) · 최신 요약: [progress/summary/latest.md](progress/summary/latest.md) · 실험 기록: [notes/04_experiments.md](notes/04_experiments.md)

## 실행 환경
- Python + [uv](https://github.com/astral-sh/uv), 에뮬레이터 [PyBoy](https://github.com/Baekalfen/PyBoy), Stable-Baselines3
- `uv sync` 후 `uv run python -m train.train_ppo ...` (명령 예시는 각 실험 기록에)

## 고지
이 저장소는 ROM 및 세이브 파일을 배포하지 않습니다. 실행하려면 이용에 필요한 권한을 확보한 포켓몬 금(한국어판) ROM을 사용자가 직접 준비해 roms/gold_rom.gb에 배치해야 합니다.
