$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $Root
docker compose -f data/dev_resources/middleware_third_party/middleware/observability/docker-compose.yml up -d
Write-Host "Jaeger UI:        http://127.0.0.1:16686"
Write-Host "Prometheus:       http://127.0.0.1:9090"
Write-Host "Alertmanager:     http://127.0.0.1:9093"
Write-Host "探活: Prometheus → Status → Targets → b2c-blackbox-health"
