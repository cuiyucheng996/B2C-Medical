"""启动 / 停止 Graphiti 官方 Docker 全栈。

在 B2C business 目录：
    uv run python server/mcp_server/graphiti/run.py
    uv run python server/mcp_server/graphiti/run.py --down

默认映射（容器内仍为官方 8000/3000/6379）：
    MCP:        http://127.0.0.1:8016/mcp/
    图 UI:      http://127.0.0.1:3016
    FalkorDB:   127.0.0.1:6380

前置：
    1. 安装并启动 Docker Desktop
    2. 复制 server/mcp_server/graphiti/.env.example → server/mcp_server/graphiti/.env，填写 OPENAI_API_KEY
    3. 启用陪聊 MCP 时，将 config.yaml mcp.servers.graphiti.url 设为上述 MCP 地址
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

GRAPHITI_DIR = Path(__file__).resolve().parent
B2C_ROOT = GRAPHITI_DIR.parents[1]
COMPOSE_FILE = (
    B2C_ROOT
    / "data/dev_resources/middleware_third_party/third_party/graphiti/docker-compose.yml"
)


def _load_b2c_env() -> None:
    """把 B2C .env 里已有的 LLM Key 注入当前进程，供 compose 变量替换。"""
    env_path = B2C_ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"} and key not in os.environ:
            os.environ[key] = val.strip().strip('"').strip("'")


def _docker_compose_base() -> list[str]:
    if shutil.which("docker"):
        try:
            r = subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if r.returncode == 0:
                return ["docker", "compose"]
        except (OSError, subprocess.TimeoutExpired):
            pass
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    print("未找到 docker compose，请先安装 Docker Desktop。", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Graphiti 官方 FalkorDB 全栈（Docker）")
    parser.add_argument("--down", action="store_true", help="停止并移除容器")
    parser.add_argument("--pull", action="store_true", help="拉取最新镜像后再启动")
    parser.add_argument("-d", "--detach", action="store_true", default=True, help="后台运行（默认）")
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="前台运行（等同 docker compose up 不加 -d）",
    )
    args = parser.parse_args()
    detach = args.detach and not args.foreground

    _load_b2c_env()
    local_env = GRAPHITI_DIR / ".env"
    if not local_env.is_file() and not os.getenv("OPENAI_API_KEY"):
        print(
            "警告：未配置 OPENAI_API_KEY。请创建 server/mcp_server/graphiti/.env 或在 B2C business/.env 中设置。",
            file=sys.stderr,
        )

    base = _docker_compose_base()
    cmd = base + ["-f", str(COMPOSE_FILE)]
    if args.down:
        subprocess.check_call(cmd + ["down"], cwd=str(GRAPHITI_DIR))
        print("Graphiti 栈已停止。")
        return
    if args.pull:
        subprocess.check_call(cmd + ["pull"], cwd=str(GRAPHITI_DIR))
    up = cmd + ["up"]
    if detach:
        up.append("-d")
    subprocess.check_call(up, cwd=str(GRAPHITI_DIR))
    print("Graphiti 栈已启动：")
    print("  MCP:     http://127.0.0.1:8016/mcp/")
    print("  图 UI:   http://127.0.0.1:3016")
    print("  健康检查: curl http://127.0.0.1:8016/health")


if __name__ == "__main__":
    main()
