---
name: b2c-business
description: >-
  B2C C 端陪聊：Python 生产栈（FastAPI 8002 + LangGraph + Agent + Redis/Neo4j 记忆）。
  Use when editing B2C business/, companion API, session memory, MCP bridge,
  intent_worker, spell check training under src/models, or deploying/monitoring the Python service.
---

# B2C business 规范（Python 生产）

## 核心原则

| 层 | 位置 | 职责 |
|----|------|------|
| **生产运行时** | `B2C business/src` + `vendor/uvgraph`（`import src.*`） | REST、Session、LangGraph、Agent、MCP Client、Redis 短期记忆 |
| **离线 ML** | `src/models/`、`src/runner/` | BERT 纠错等训练/预测脚本，**不**单独起 Java 推理服务 |
| **配置** | `data/dev_resources/middleware_third_party/config.yaml` + `.env` | API Key、Redis、Neo4j；路径见 `configs/data_paths.py`（不再依赖上级 uvgraph `sys.path`） |

**C 端生产只部署 Python（Uvicorn），不部署 Spring Boot / JDK。** 文档与 Agent 禁止再把「Java 生产 + Python 训练」写成默认架构。

## 目录与分层

```
B2C business/
├── run.py
├── src/
│   ├── api/           # FastAPI，薄：校验、调 session/agent/graph
│   ├── graph/         # LangGraph 节点、routes、checkpointer
│   ├── agent/         # create_agent、子 thread_id
│   ├── session.py     # SessionStore（进程内）
│   ├── algorithm/     # 意图、压缩、槽位、关场 release（单一入口 session_memory）
│   ├── memory/        # Graphiti 形态 ingest/search
│   ├── tools/         # registry、mcp_bridge、companion_tools
│   ├── middleware/    # envelope、compress、tool_trace
│   └── configs/paths_b2c.py
├── doc/mcp.md
└── tests/
```

约定与 uvgraph 主项目对齐：

- 图节点要薄；复杂分支进 `algorithm/` 或 `graph/*_route.py`
- 环境变量只经主项目 `get_app_settings()` / `src/config.py`，B2C 业务不散落 `os.getenv`
- Redis 键：`conversation:short|compress:{session_id}`；槽位 `conversation:slots:{user_id}` + `session_id` 校验（见 `algorithm/session_memory.py`）
- Checkpoint `thread_id`：主图 `session_id`；子 Agent `agent:{session_id}:{major}:{handler}`

## 单一维护入口（记忆 / 关场）

| 类别 | 权威源 |
|------|--------|
| 关场清 Redis + checkpoint | `algorithm/session_release.release_companion_session` → `session_memory.release_session_runtime_state` |
| 新开替换旧场 | `session.SessionStore.create_new` |
| MCP 连接与登记 | `uvgraph/src/configs/config.yaml` `mcp.servers` → `tools/mcp_yaml_registry.register_mcp_from_yaml` + `mcp_bridge` |
| 意图清单 / 规则 / 训练模板 | `middleware_third_party/intent_catalog.yaml` → `algorithm/intent_manifest.py` |
| 细分意图 SOP（子图 Skill） | `src/prompt/intent_skills/{handler}.md` → `intent_skill_loader` → `companion_dispatch` |
| 结构化输出（schema 末尾 + API + 自修复 + 降级） | `prompt/structured_block.py` + `algorithm/structured_output.py` + `algorithm/json_sanitize.py` + `tools/tool_envelope.validate_envelope_json` |
| 工具绑定 | `tools/setup.py` `INTENT_BIND` + `CompanionToolRegistry` |

## Agent 行为规范

### 必须

1. 改 API/Agent/图 → 只动 Python `B2C business/src`（及共享的 uvgraph `src/`）
2. 先搜索是否已有同类逻辑，再扩展 `algorithm/` 或 `tools/`
3. 保留已有注释；最小改动
4. 改完后可验证：`uv run python "B2C business/run.py"` 或针对性 `tests/test_*.py`
5. 部署/监控文档写 **8002 + /healthcheck + 日志**，不写 Java Actuator

### 禁止

1. 在 C 端路径新增 `src/main/java`、`pom.xml`、Spring 控制器
2. 把「生产必须 JAR」写进 README / SKILL / architecture
3. 在节点里堆业务常量；勿把 LangGraph `RouterState` 放进无必要的 `schemas` 混用（B2C 用 `graph/state.py`）

### 新增功能决策

| 需求 | 做法 |
|------|------|
| 新 REST | `api/server.py` + `algorithm/` |
| 新意图 / 子图 | `graph/intent_worker.py`、`algorithm/intent_catalog.py` |
| 外部能力 | MCP Server + YAML；或 `companion_tools` 本地工具 |
| BERT / 纠错 | Python `src/runner/` 或推理挂到 algorithm，不经 Java ONNX |

## 部署与监控（统一模型）

```powershell
# 开发 / 单机生产示例
cd <uvgraph-root>
uv run python "B2C business/run.py" --host 0.0.0.0 --port 8002
```

| 信号 | 用途 |
|------|------|
| `GET /healthcheck` | 存活探针 |
| 结构化 `logging` | 排障、工具 trace |
| Redis / Neo4j | 依赖健康（需在运维层或后续加 `/ready` 时再扩展） |

多 worker：注意进程内 `SessionStore` 与 LangGraph memory checkpointer；扩实例前需 Redis 会话或粘性路由。

## 常用命令

```powershell
uv run python "B2C business/run.py"
uv run pytest "B2C business/tests/companion/test_companion_graph.py"
uv run pytest "B2C business/tests/memory/test_session_memory.py" -v
```

## 延伸阅读

- [architecture.md](architecture.md) — 请求路径与和 uvgraph 边界
- [doc/mcp.md](../../doc/mcp.md) — MCP Host/Client/Server
