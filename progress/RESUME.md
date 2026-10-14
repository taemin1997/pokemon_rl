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

## 4. 현재 재개 지점 (10-13)
- M2B_multi 100만 학습과 E27 평가는 끝남. **M2B를 재개 학습하지 않음**
- E26 1단계 재현성 검증 완료: `runs/e26/stage1_report.md`
- 다음은 E26 2단계 기준선. `notes/experiments/E26_SELFREPAIR.md`의 순서를 따름
  1. E-lab/E-violet 평가 상태 생성·고정 및 SHA1/의미적 중복 기준 기록
  2. 아침·낮·밤 3조건 기준선 측정
  3. 정책 실패/규칙 개입 실패/구간 전환 오류/실험 무효 분리
- 기준선 결과를 먼저 보고하기 전에는 E26 학습, PPO 이어 학습, 모델 변경을 시작하지 않음
- 재현성 검증을 다시 돌릴 때만 PowerShell에서 `./scripts/e26_stage1.ps1` 사용.
  - 끊겨도 계속 돌게 띄우기 (10-14): `Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-Command','& .\scripts\e26_stage1.ps1 2>&1 | Out-File -Encoding utf8 runs\e26\stage1.log' -WindowStyle Hidden`
  - 주의: `*>` 리다이렉트는 UTF-16 로그가 됨. Windows PowerShell 5.1에선 Stop 모드 + 파이썬 stderr 경고가 스크립트를 멈춤 (스크립트 안에서 호출 구간만 Continue로 처리함)
  - 생성되는 `runs/stuck` 디버그 덤프와 캐시는 확인 후 정리

## 5. 저장소 안전장치 (10-14)
- 새로 받은 저장소에선 한 번: `uv run python scripts/check_assets.py --install` (커밋 전 저작물 검사 훅. ROM 헤더·금지 확장자·허용 안 된 이미지·2MB 넘는 파일을 막음)
- 게임 화면 캡처는 `notes/img_local/` (무시 목록, 저장소에 안 올라감). 직접 만든 그래프만 `docs/figures/`에
