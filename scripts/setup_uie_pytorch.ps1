# 拉取 UIE PyTorch 推理代码到 External/uie_pytorch（默认走本机 9674 代理）
$ErrorActionPreference = "Stop"
$proxy = if ($env:UIE_SETUP_PROXY) { $env:UIE_SETUP_PROXY } else { "http://127.0.0.1:9674" }
$b2c = Split-Path $PSScriptRoot -Parent
$dest = Join-Path $b2c "External\uie_pytorch"
if (Test-Path (Join-Path $dest "uie_predictor.py")) {
    Write-Host "已存在 $dest，跳过 clone"
    exit 0
}
if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
git -c "http.proxy=$proxy" -c "https.proxy=$proxy" clone --depth 1 https://github.com/HUSTAI/uie_pytorch.git $dest
Write-Host "完成: $dest"
