# 끊겼을 때 다시 시작하기

## 1. 학습이 중간에 끊겼을 때

### 어떤 파일이 남아 있나
| 끊긴 방식 | 남는 것 |
|---|---|
| Ctrl+C (정상 중단) | `models/<실험>/final.zip` (그 순간까지) + 체크포인트들 |
| 창 닫기, 강제 종료, 정전, 재부팅 | 마지막 체크포인트 `ppo_<스텝>_steps.zip`만 (약 5만 스텝마다 저장) → 최대 5만 스텝 손해 |

### 이어서 학습 (한 줄)
```bash
uv run python -m train.train_ppo --env <구간> --reward <보상> --name <실험> --resume latest --total-steps <누적 목표>
```
- `--resume latest`: 그 실험 폴더에서 **가장 최근 정식 저장본**(final.zip 또는 ppo_*_steps.zip, live.zip 제외, 10-08)을 고름. 전선 상태도 그 모델 스텝 시점으로 복원 (`demo_ptr_history.jsonl`, 10-08 이후 학습부터)
- `--total-steps`: **누적 목표**. 이미 학습한 만큼 빼고 남은 만큼만 학습함. 이미 넘었으면 학습 안 하고 알려줌
- `--env`, `--reward`는 처음 학습할 때와 같게 줘야 함 (모델 파일엔 환경/보상 설정이 저장되지 않음)
- TensorBoard 곡선은 같은 `runs/<실험>_N` 폴더에 이어서 그려짐

예:
```bash
uv run python -m train.train_ppo --env route29 --reward task --name B2_route29 --resume latest --total-steps 250000
```

### 커리큘럼 학습(AC 계열)을 이어서 할 때
- 아카이브(연습 장소 후보, 익힌 자리)는 `runs/curriculum/<실험>/env_*.pkl`에 자동 저장됨 (판 100번마다 + 종료 시)
- `--resume`으로 이어서 하면 폴더의 파일을 전부 합쳐서 불러옴 → 익힌 자리부터 다시 시작
- 예외: 2026-10-06 14시 이전에 시작한 학습(AC3_route29 첫 실행 포함)은 이 기능이 없어서 저장 파일이 없음 → 이어서 해도 아카이브는 처음부터 다시 쌓임 (정책은 이어짐)
- 라이브 창용 `models/<실험>/live.zip`이 약 1만 스텝마다 덮어써짐 → `--resume latest`가 이걸 고를 수 있는데, 가장 최근 정책이라 문제없음
- 역방향 시연(DEMO 계열): 전선 위치가 `runs/curriculum/<실험>/demo_ptr.json`에 저장됨 (전선을 당길 때마다). `--resume`이면 거기서부터 이어감

### 주의
- **같은 실험 이름으로 `--resume` 없이 다시 실행하면** 안전장치가 막아줌 (기존 모델 덮어쓰기 방지)
- 학습 도중에 `envs/`, `train/` 코드를 고쳐도 **실행 중인 학습엔 영향 없음**. 이어서 할 때는 고친 코드로 돌아감 → 보상/환경을 고쳤다면 이어서 하지 말고 새 실험 이름으로 시작하는 게 맞음

## 2. 안 꺼지는 프로세스 정리
Ctrl+C를 여러 번 눌러도 안 꺼지면 (PyBoy 내부에서 신호를 삼키는 경우), PowerShell에서:
```powershell
Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -like '*train.*' } | Select-Object ProcessId, CommandLine
```
확인한 뒤 해당 번호를:
```powershell
Stop-Process -Id <번호> -Force
```
(강제 종료라 final.zip은 안 생김 → 이어서 할 땐 `--resume latest`가 마지막 체크포인트를 씀)

## 3. 작업 세션(대화)이 끊겼을 때
새 대화를 시작하면 이 순서로 보면 지금 어디인지 바로 알 수 있음
1. `progress/README.md`: 전체 현황, 마일스톤, 갱신 기록 (맨 아래가 가장 최근)
2. `progress/step4_training.md`: 실험 표와 "다음 할 일"
3. `progress/segments.md`: 구간 현황과 추가 절차
4. `notes/04_experiments.md`: 실험별 가설, 판정 기준, 결과 (결과 칸이 비어 있는 실험 = 진행 중이었던 것)
5. `models/`, `runs/`: 어떤 실험이 어디까지 학습됐는지 (파일 이름의 스텝 수)

새 대화에서 Claude에게는 "progress/README.md 보고 이어서 하자" 정도면 됨 (프로젝트 기억에도 이 폴더 구조가 저장돼 있음)

## 4. M2B_multi 학습 연쇄 (10-12 09:19 재개 기준)
- 스크립트 (세션 scratchpad, 지워지면 아래 내용으로 다시 만들기): `m2b_train5.sh` → 로그 `runs/m2b_train5.log`, `locked_eval5.sh` → `runs/m2b_locked5.log`
  - 학습: `train_ppo --env m2b_multi --reward goal_only --move tile --goal --hidden 256,256 --fast-battle --auto-event --resume latest --total-steps 1000000 --name M2B_multi --no-live` (로그 20분 정지 → 학습/워커 프로세스 종료 후 재시작, 최대 5번)
  - 끝나면 평가: 새 5구간(route30_north/route31 50판) / 기존 11구간 / 경로 밖 3구간 / `run_story --targets "26,10;24,5;10,7" --max-steps 6000` 20판
  - 시계 고정 평가: 학습 로그에 "끝"이 나오면 `POKE_RTC_LOCK=1`로 새 5구간 + M2A_r30 vs M2B 11구간 + 통합 실행 두 모델 (학습 연쇄가 "중단"이면 같이 끝남)
- 중간에 끊기면 같은 두 스크립트를 `nohup bash`로 다시 실행하면 됨 (`--resume latest`라 마지막 체크포인트부터)
