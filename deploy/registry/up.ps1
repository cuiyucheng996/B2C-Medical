$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $Root
docker compose -f data/dev_resources/middleware_third_party/middleware/registry/docker-compose.yml up -d
Write-Host "Consul UI: http://127.0.0.1:8500"
Write-Host "Nacos:     http://127.0.0.1:8848/nacos (standalone, 无鉴权)"
