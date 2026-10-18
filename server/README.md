# server 目录

| 目录 | 说明 |
|------|------|
| `builtin_server/` | 自建 FastAPI 旁路服务（报告 8012、入库 8013、采集 8014、记忆 8015 等） |
| `mcp_server/` | MCP 形态服务（如 Graphiti） |

注册发现（Consul/Nacos 客户端）在 **`src/infra/registry/`**，不是独立服务。

启动示例：

```powershell
uv run python server/builtin_server/report_read/run.py
uv run python server/mcp_server/graphiti/run.py
```
