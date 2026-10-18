"""8014 定向采集：直连 server/mcp_server/yingdao（stdio），不经 8002 陪聊 MCP。"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

from configs.paths_b2c import B2C_ROOT, ensure_b2c_runtime_path

ensure_b2c_runtime_path()

logger = logging.getLogger(__name__)

_MCP_ENTRY = B2C_ROOT / "server" / "mcp_server" / "yingdao" / "dist" / "index.js"


def mcp_script_path() -> Path:
    override = (os.getenv("YINGDAO_MCP_SCRIPT") or "").strip()
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = (B2C_ROOT / p).resolve()
        return p
    return _MCP_ENTRY.resolve()


def _yingdao_env() -> dict[str, str]:
    from tools.mcp_bridge import _yingdao_stdio_env

    from src.configs import get_yaml_config

    cfg = get_yaml_config()
    mcp = cfg.get("mcp") if isinstance(cfg, dict) else {}
    servers = mcp.get("servers") if isinstance(mcp, dict) else {}
    spec = servers.get("yingdao_rpa") if isinstance(servers, dict) else {}
    if not isinstance(spec, dict):
        spec = {}
    return _yingdao_stdio_env(spec)


def connection_ready() -> bool:
    from tools.mcp_bridge import _yingdao_rpa_ready

    script = mcp_script_path()
    if not script.is_file():
        return False
    node = (os.getenv("YINGDAO_NODE") or "").strip() or shutil.which("node") or shutil.which("node.exe")
    if not node:
        return False
    return _yingdao_rpa_ready(_yingdao_env())


def stdio_connection() -> dict[str, Any]:
    script = mcp_script_path()
    if not script.is_file():
        raise FileNotFoundError(f"影刀 MCP 未构建：{script}（在 server/mcp_server/yingdao 执行 npm run build）")
    node = (os.getenv("YINGDAO_NODE") or "").strip() or shutil.which("node") or shutil.which("node.exe")
    if not node:
        raise RuntimeError("未找到 node（可设置 YINGDAO_NODE）")
    return {
        "transport": "stdio",
        "command": node,
        "args": [str(script)],
        "env": _yingdao_env(),
    }


async def invoke_run_app(app_uuid: str, app_params: dict[str, Any]) -> str:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    if not connection_ready():
        raise RuntimeError("影刀 MCP 未就绪（检查 .env 与 server/mcp_server/yingdao/dist/index.js）")
    conn = stdio_connection()
    client = MultiServerMCPClient({"yingdao_directed_crawl": conn}, tool_name_prefix=True)
    tools = await client.get_tools()
    run_tool = None
    for tool in tools:
        name = str(getattr(tool, "name", "") or "")
        if name.endswith("runApp"):
            run_tool = tool
            break
    if run_tool is None:
        raise RuntimeError("影刀 MCP 无 runApp 工具")
    payload = {"appUuid": (app_uuid or "").strip(), "appParams": app_params or {}}
    result = await run_tool.ainvoke(payload)
    return str(result)
