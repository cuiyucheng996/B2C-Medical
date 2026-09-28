# Inspect View。默认只打开「最新一条 multi-intent-case」，避免误点 companion-smoke。

param(

    [string]$Pick = "multi-intent-case",

    [switch]$AllLogs

)

$ErrorActionPreference = "Stop"

$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent

$Uv = "E:/uv/bin/uv.exe"

$VenvPy = Join-Path $Root ".venv/Scripts/python.exe"

$Inspect = Join-Path $Root ".venv/Scripts/inspect.exe"

$logDir = Join-Path $PSScriptRoot "logs"

if (-not (Test-Path $Inspect)) { throw "未找到 inspect.exe，请先 run_inspect.ps1 或 pip install -e External/agent-eval/inspect_ai" }

& $Uv pip install -r (Join-Path $PSScriptRoot "inspect_ai_pip_pins.txt") --python $VenvPy -q



$viewDir = $logDir

if (-not $AllLogs) {

    $all = @(Get-ChildItem -Path $logDir -Filter "*.eval" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)

    if ($all.Count -eq 0) { throw "logs 里没有 .eval，请先 run_inspect.ps1" }

    $chosen = @($all | Where-Object { $_.Name -like "*$Pick*" } | Select-Object -First 1)

    if ($chosen.Count -eq 0) {

        $chosen = @($all[0])

        Write-Host "未找到 *$Pick*，改用最新: $($chosen[0].Name)" -ForegroundColor Yellow

    } else {

        Write-Host "当前 View 只显示这一条（不是 companion-smoke 的「你好/谢谢」）:" -ForegroundColor Green

        Write-Host "  $($chosen[0].Name)" -ForegroundColor Green

    }

    $viewDir = Join-Path $PSScriptRoot "logs_view_single"

    if (Test-Path $viewDir) { Remove-Item -Recurse -Force $viewDir }

    New-Item -ItemType Directory -Path $viewDir | Out-Null

    Copy-Item -LiteralPath $chosen[0].FullName -Destination $viewDir

}



Set-Location $PSScriptRoot

Write-Host ""

Write-Host "Inspect View: http://127.0.0.1:7575"

Write-Host "打开后应只有 1 条 eval、1 个 Sample（长句：乏力口干…）"

Write-Host "要看全部历史（含 smoke）: run_inspect_view.ps1 -AllLogs"

Write-Host "（本窗口不要关）"

Write-Host ""

Start-Process "http://127.0.0.1:7575"

& $Inspect view --log-dir $viewDir --host 127.0.0.1 --port 7575 --recursive:$false

