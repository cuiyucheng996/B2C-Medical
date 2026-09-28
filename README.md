# B2C business（C 端陪聊）

**单一生产栈：Python**。陪聊 API、LangGraph 编排、Agent、会话与 Redis 记忆均在本目录 `src/`；由 `run.py` 启动 **Uvicorn + FastAPI**（默认 **8002**）。

与 uvgraph 主仓关系：运行时 `sys.path` 先挂 **uvgraph 根**（共享 `src.redis`、`src.llm`、Neo4j 用户记忆等），再挂 **B2C `src/`**（`api` / `agent` / `graph` / `session`）。Agent 规范见 `.agents/skills/b2c-business/SKILL.md`。

## 技术栈

| 组件 | 说明 |
|------|------|
| Python | `>=3.11`，依赖由仓库根 `uv` / `pyproject.toml` 管理 |
| API | FastAPI `api.server:app` |
| 编排 | LangGraph（`graph/companion.py` 等） |
| LLM / Embedding | 主项目 `src.llm`、DashScope（`.env`） |
| 短期记忆 | Redis `conversation:short|compress:{session_id}`、槽位 `conversation:slots:{user_id}` |
| 长期记忆 | Neo4j / Graphiti 形态（`memory/graphiti_store.py`） |
| MCP | `tools/mcp_bridge.py`，见 `doc/mcp.md` |

历史上曾规划 **Spring Boot + ONNX** 双栈，**C 端已不再采用**；纠错与训练脚本留在 `src/models/`、`src/runner/`（Python 离线），不进独立 Java 服务。

## 目录结构

```
B2C business/
├── run.py                 # 生产入口（默认 127.0.0.1:8002）
├── doc/                   # mcp.md 等
├── tests/
└── src/
    ├── api/server.py      # REST + 静态调试页
    ├── agent/             # create_agent、子意图 Worker
    ├── graph/             # LangGraph 节点与 checkpoint
    ├── session.py         # SessionStore
    ├── algorithm/         # 业务算法（与图解耦）
    ├── memory/            # ingest / search 封装
    ├── tools/               # 本地工具 + MCP bridge
    ├── middleware/
    ├── configs/paths_b2c.py
    ├── models/            # BERT 等（训练/评测，非 Java ONNX 路径）
    └── web/               # companion 调试台
```

## 运行

在 **uvgraph 仓库根**（有 `.venv` 与 `.env`）：

```powershell
uv run python "B2C business/run.py"
# 或指定端口
uv run python "B2C business/run.py" --port 8002 --reload
```

启动后：

- 健康检查：`GET http://127.0.0.1:8002/healthcheck`
- 陪聊调试页：`http://127.0.0.1:8002/`（`companion.html`）
- 工具登记：`GET http://127.0.0.1:8002/api/tools`

## 生产部署（Python 单进程 / 多 worker）

| 项 | 建议 |
|----|------|
| 进程 | `uvicorn api.server:app`，`app_dir=B2C business/src`，`--workers N` 时 **同用户会话须粘滞或外置 SessionStore**（当前默认进程内 `SessionStore`） |
| 配置 | 仓库根 `.env`：`DASHSCOPE_API_KEY`、`KIMI_*`、`REDIS_*`、Neo4j 等（与主项目一致，业务代码用 `src/config.py`，勿散落 `load_dotenv`） |
| 监控 | HTTP `GET /healthcheck`；应用日志（`logging`）；可选对 Redis / Neo4j 做依赖探针（未内置则需网关或 sidecar） |
| 图谱 / RAG | 优先走本服务内算法与 MCP；需 B2B 问诊能力时用 `b2c_b2b_bridge`，**不**再经 Java 8080 |

## 常用命令

```powershell
# 图编译自检
uv run python -m graph.companion   # 需在 B2C src 为 cwd 或经 run.py 同 path 设置

# 单测
D:\program\uvgraph\uvgraph\.venv\Scripts\python.exe "B2C business\tests\test_session_memory.py" -v
```

## 延伸阅读

- 架构与请求路径：`.agents/skills/b2c-business/architecture.md`
- MCP： `doc/mcp.md`
