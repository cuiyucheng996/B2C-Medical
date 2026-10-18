# 可观测

```powershell
.\deploy\observability\up.ps1
uv sync --group observability
```

`.env`：

```env
B2C_OTEL_ENABLED=true
B2C_OTEL_ENDPOINT=http://127.0.0.1:4318/v1/traces
```

再起 `run.py` / 子服务。Jaeger UI：**http://127.0.0.1:16686**，按 `service.name` 查 `b2c-companion` 等。

- HTTP 请求：FastAPI 自动 Span  
- 陪聊一轮：`companion.turn` + `log_trace_event` 事件（工具/MCP 等）  
- 跨服务：`X-Trace-Id` + W3C `traceparent`（`internal_headers`）
