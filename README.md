# B2C 陪聊

Python 陪聊服务（FastAPI + LangGraph）。仓库根目录就是工程根，克隆后在本目录部署。

**目录分块**（前端 / 后端 / 部署 / 测试）：见 [doc/project-layout.md](doc/project-layout.md)。

## 环境

- Python 3.11、3.12 或 3.13
- [uv](https://docs.astral.sh/uv/)（装依赖并启动）
- Redis：`localhost:6379`（短期对话，见 `config.yaml` 的 `redis`）
- 可选：Neo4j `neo4j://localhost:7687`（图记忆；密码放 `.env` 的 `NEO4J_PASSWORD`）
- 可选：Chroma。主配置里 `chroma.server_host` 有值时要先起服务（`data/dev_resources/middleware_third_party/middleware/chroma/docker-compose.yml`，默认端口 8000）。留空则用本地嵌入式库，不必单独起进程

GPU、MinerU、语音 ASR/TTS 不是打开陪聊页的前提。报告解析、语音外壳按 `config.yaml` 里对应段再配。

## 安装与配置

```powershell
cd "B2C business"
uv sync
copy .env.example .env
```

编辑 `.env`，至少填写：

```
DEEPSEEK_API_KEY=你的密钥
```

主配置在 `data/dev_resources/middleware_third_party/config.yaml`（可提交，非密钥）。不要把 `.env` 提交到 Git。`.cache/`、`.venv/` 是本机缓存和虚拟环境，也不要提交。

## 启动

```powershell
uv run python run.py
```

默认 `http://127.0.0.1:8002`。换端口：`uv run python run.py --port 8002`。

- 健康检查：`http://127.0.0.1:8002/healthcheck`（返回 `{"status":"healthy"}`）
- 陪聊页：`http://127.0.0.1:8002/chat`

上下文压缩和工具后思考默认走通义（`config.yaml` 的 `llm.context_compress` / `llm.think`）。要用的话在 `.env` 增加 `DASHSCOPE_API_KEY`；不用可在 `config.yaml` 把对应 `enabled` 改为 `false`。
