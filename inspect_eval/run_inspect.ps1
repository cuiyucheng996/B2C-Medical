# C 端 + Inspect AI：先启动 run.py (:8002)，再本脚本。
$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$B2C = Split-Path $PSScriptRoot -Parent
$InspectSrc = Join-Path $B2C "External/agent-eval/inspect_ai"
$VenvPy = Join-Path $Root ".venv/Scripts/python.exe"
$Uv = "E:/uv/bin/uv.exe"

if (-not (Test-Path $VenvPy)) { throw "缺少 .venv: $VenvPy" }

$env:HTTP_PROXY = "http://127.0.0.1:9674"
$env:HTTPS_PROXY = "http://127.0.0.1:9674"

Write-Host "对齐 inspect 依赖 (botocore/aiobotocore)..."
& $Uv pip install -r (Join-Path $PSScriptRoot "inspect_ai_pip_pins.txt") --python $VenvPy -q

Write-Host "安装 inspect_ai (editable)..."
& $Uv pip install -e $InspectSrc --python $VenvPy -q

$base = if ($env:B2C_INSPECT_API_BASE) { $env:B2C_INSPECT_API_BASE } else { "http://127.0.0.1:8002" }
try {
    Invoke-WebRequest -Uri "$base/healthcheck" -UseBasicParsing -TimeoutSec 5 | Out-Null
} catch {
    throw "C 端未就绪 ($base/healthcheck)，请先: python `"B2C business/run.py`""
}

Set-Location (Join-Path $B2C "inspect_eval")
$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$task = if ($args.Count -gt 0) { $args[0] } else { "companion_api_task.py@companion_multi_intent_case" }
Write-Host "inspect eval $task ..."
& (Join-Path $Root ".venv/Scripts/inspect.exe") eval $task --model mockllm/model --log-dir $logDir --limit 5

Write-Host ""
Write-Host "可视化（另开终端）:"
Write-Host "  powershell -File `"$PSScriptRoot\run_inspect_view.ps1`""
Write-Host "或:"
Write-Host "  & `"$(Join-Path $Root '.venv/Scripts/inspect.exe')`" view --log-dir `"$logDir`""
