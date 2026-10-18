$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $Root
docker compose -f data/dev_resources/middleware_third_party/middleware/gateway/docker-compose.yml up -d
Write-Host "统一入口: http://127.0.0.1:8080 （静态 upstream；发现模式用 up-discovery.ps1）"
