# E26 1단계 재현성 검증 연쇄 (Windows). 학습·모델 변경 없음, 평가만.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"
$python = ".\.venv\Scripts\python.exe"
$model = "models/M2B_multi/final.zip"
$list = "runs/e26/stage1_runs.json"
New-Item -ItemType Directory -Force "runs/e26" | Out-Null
$runs = [ordered]@{}

function Stamp([string]$message) {
    Write-Output "== $(Get-Date -Format 'MM-dd HH:mm:ss') $message"
}

function Save-Runs {
    $runs | ConvertTo-Json | Set-Content -Encoding utf8 $list
}

function Diagnose([string]$key, [string]$clock, [string]$segment) {
    Stamp "$key ($clock, $segment)"
    $env:POKE_CLOCK = $clock
    # Windows PowerShell 5.1: Stop 상태에서 stderr 경고(SDL2 안내 등)를 2>&1로 받으면 오류로 멈춤 (10-14) → 이 호출만 Continue, 성공은 종료 코드로
    $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    $output = & $python -m train.diagnose $segment $model --runs 25 --workers 6 2>&1
    $ErrorActionPreference = $prev
    if ($LASTEXITCODE -ne 0) { throw ($output -join "`n") }
    $saved = $output | ForEach-Object { "$_" } | Where-Object { $_ -match '^저장:\s*' } | Select-Object -Last 1
    if (-not $saved) { throw "train.diagnose 저장 경로를 찾지 못함" }
    $path = $saved -replace '^저장:\s*', ''
    Write-Output "  → $path"
    $runs[$key] = $path
    Save-Runs
}

function Story([string]$key, [string]$clock) {
    Stamp "$key (통합 실행 3판, $clock)"
    $env:POKE_CLOCK = $clock
    # Windows PowerShell 5.1: Stop 상태에서 stderr 경고(SDL2 안내 등)를 2>&1로 받으면 오류로 멈춤 (10-14) → 이 호출만 Continue, 성공은 종료 코드로
    $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    $output = & $python -m train.run_story $model --headless --episodes 3 --quiet --targets "26,10;24,5;10,7" --max-steps 2500 --no-heal-at "10,5" 2>&1
    $ErrorActionPreference = $prev
    if ($LASTEXITCODE -ne 0) { throw ($output -join "`n") }
    $saved = $output | ForEach-Object { "$_" } | Where-Object { $_ -match '^저장:\s*' } | Select-Object -Last 1
    if (-not $saved) { throw "train.run_story 저장 경로를 찾지 못함" }
    $path = $saved -replace '^저장:\s*', ''
    Write-Output "  → $path"
    $runs[$key] = $path
    Save-Runs
}

Stamp "E26 1단계 재검증 시작"
Diagnose "V1a" "lock" "route30"
Diagnose "V1b" "lock" "route30"
$runs["V2_old"] = "runs/diagnose/route30_1012_1723_M2A_r30+M2B_multi_summary.json"
Save-Runs
foreach ($clock in @("morning", "day", "night")) {
    Diagnose "V3_route30_$clock" $clock "route30"
    Diagnose "V3_route30_${clock}_2" $clock "route30"
    Diagnose "V3_route29_$clock" $clock "route29"
}
Diagnose "V4_wall" "wall" "route30"
Story "V5a" "lock"
Story "V5b" "lock"

Stamp "V6 무효 판정 시험"
$ErrorActionPreference = "Continue"
& $python -m train.e26_stage1_v6
$ErrorActionPreference = "Stop"
if ($LASTEXITCODE -ne 0) { throw "V6 실행 실패" }
$runs["V6"] = "runs/e26/stage1_v6.json"
Save-Runs

Stamp "판정"
$ErrorActionPreference = "Continue"
& $python -m train.e26_stage1_check $list
$ErrorActionPreference = "Stop"
if ($LASTEXITCODE -ne 0) { throw "판정 실행 실패" }
Stamp "끝"
