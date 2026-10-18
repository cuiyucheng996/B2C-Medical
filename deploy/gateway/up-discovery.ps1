$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $Root

Write-Host "1/3 启动 Consul + Nacos ..."
docker compose -f data/dev_resources/middleware_third_party/middleware/registry/docker-compose.yml up -d

Write-Host "2/3 发布共享配置到 Nacos（data/dev_resources/middleware_third_party/middleware/nacos）..."
uv run python deploy/scripts/publish_nacos_config.py --merge-dotenv

Write-Host "3/3 构建并启动 Consul 感知网关 ..."
docker compose -f data/dev_resources/middleware_third_party/middleware/gateway/docker-compose.discovery.yml up -d --build

Write-Host ""
Write-Host "入口: http://127.0.0.1:8080"
Write-Host "请在 .env 启用: B2C_NACOS_CONFIG=true, B2C_SERVICE_DISCOVERY=consul,nacos"
Write-Host "网关模式注册地址: B2C_SERVICE_REGISTER_HOST=host.docker.internal"
Write-Host "然后启动: uv run python run.py 及各 server/*/run.py"
