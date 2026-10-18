# B2C 网关

| 模式 | 命令 | 说明 |
|------|------|------|
| 静态 | `up.ps1` | `nginx.conf` 固定 `host.docker.internal:端口` |
| 发现 | `up-discovery.ps1` | **consul-template** 按 Consul 健康实例生成 upstream；Nacos 管共享 env |

**企业链路（发现模式）**

1. `docker compose -f deploy/registry` 起 Consul + Nacos  
2. `uv run python deploy/scripts/publish_nacos_config.py --merge-dotenv` 把共享项推到 Nacos `b2c-business.env`  
3. 业务 `.env` 开 `B2C_NACOS_CONFIG=true`，启动时拉 Nacos（不覆盖本机已设变量）  
4. 业务开 `B2C_SERVICE_DISCOVERY=consul,nacos`，`B2C_SERVICE_REGISTER_HOST=host.docker.internal` 注册实例  
5. `up-discovery.ps1` 起网关；**只暴露 8080**，按路径反代，后端 IP 来自 Consul 健康检查  

服务间 HTTP 仍可用 `resolve_base_url()` 走 Consul/Nacos；配置以 Nacos 为准，网关不认 Nacos 服务列表。
