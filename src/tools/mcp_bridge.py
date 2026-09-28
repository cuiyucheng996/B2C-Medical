"""从已配置的 MCP 服务器拉 LangChain 工具，按大类白名单并入陪聊本轮工具集。"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

from configs.paths_b2c import ensure_uvgraph_root

logger = logging.getLogger(__name__)

__all__ = ["mcp_tools_for_major", "is_mcp_tool_name", "mcp_server_for_tool", "split_bound_tools", "tag_tool_trace_row"]

_cache: dict[str, list[BaseTool]] | None = None
_B2C_SRC = Path(__file__).resolve().parents[1]
_NPX_TO_MODULE = {
    "@modelcontextprotocol/server-time": "companion_mcp.stdio_time",
    "@modelcontextprotocol/server-fetch": "companion_mcp.stdio_fetch",
}
_PYTHON_MODULES = {
    "companion_mcp.stdio_bailian_kb",
    "companion_mcp.stdio_dashscope_search",
    "companion_mcp.stdio_oss",
    "companion_mcp.stdio_time",
    "companion_mcp.stdio_fetch",
}


def _auth_headers(spec: dict[str, Any]) -> dict[str, str]:
    headers: dict[str, str] = {}
    raw = spec.get("headers")
    if isinstance(raw, dict):
        headers.update({str(k): str(v) for k, v in raw.items()})
    env_key = str(spec.get("auth_bearer_env") or "").strip()
    if env_key:
        token = (os.getenv(env_key) or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
    return headers


def _resolve_url(spec: dict[str, Any]) -> str:
    url = str(spec.get("url") or "").strip()
    if url:
        return url
    if not spec.get("bailian_mcp"):
        return ""
    ensure_uvgraph_root()
    from src.configs import get_yaml_config

    block = get_yaml_config().get("mcp")
    aliyun = block.get("aliyun") if isinstance(block, dict) and isinstance(block.get("aliyun"), dict) else {}
    ws = (os.getenv("BAILIAN_WORKSPACE_ID") or str(aliyun.get("bailian_workspace_id") or "")).strip()
    region = str(aliyun.get("bailian_region") or "cn-beijing").strip()
    if not ws:
        return ""
    return f"https://{ws}.{region}.maas.aliyuncs.com/api/v1/indices/rag/mcp"


def _mcp_yaml() -> dict[str, Any]:
    ensure_uvgraph_root()
    from src.configs import get_yaml_config

    block = get_yaml_config().get("mcp")
    return dict(block) if isinstance(block, dict) else {}


def _npx_bin() -> str | None:
    return shutil.which("npx") or shutil.which("npx.cmd")


def _python_stdio(module: str) -> dict[str, Any]:
    root = str(_B2C_SRC.parent.parent)
    src = str(_B2C_SRC)
    path = os.pathsep.join([root, src, os.environ.get("PYTHONPATH", "")]).strip(os.pathsep)
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", module],
        "cwd": src,
        "env": {**os.environ, "PYTHONPATH": path},
    }


def _stdio_conn(command: str, args: list[Any]) -> dict[str, Any]:
    argv = [str(item) for item in args]
    if command.lower() in {"npx", "npx.cmd"}:
        pkg = next((item for item in argv if item.startswith("@")), "")
        module = _NPX_TO_MODULE.get(pkg)
        if module and not _npx_bin():
            logger.info("未找到 npx，改用 %s 启动 %s", sys.executable, module)
            return _python_stdio(module)
        npx = _npx_bin() or command
        if sys.platform == "win32":
            comspec = os.environ.get("COMSPEC") or "cmd.exe"
            return {"transport": "stdio", "command": comspec, "args": ["/c", npx, *argv]}
        return {"transport": "stdio", "command": npx, "args": argv}
    if command.lower() in {"python", "python3", "py"}:
        if len(argv) >= 2 and argv[0] == "-m":
            return _python_stdio(argv[1])
        return {"transport": "stdio", "command": sys.executable, "args": argv}
    resolved = shutil.which(command) or shutil.which(command + ".cmd") or shutil.which(command + ".exe") or command
    if sys.platform == "win32" and command.lower().endswith((".cmd", ".bat")):
        comspec = os.environ.get("COMSPEC") or "cmd.exe"
        return {"transport": "stdio", "command": comspec, "args": ["/c", resolved, *argv]}
    return {"transport": "stdio", "command": resolved, "args": argv}


def _connections() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    cfg = _mcp_yaml()
    if not bool(cfg.get("enabled", True)):
        return {}, {}
    servers = cfg.get("servers") if isinstance(cfg.get("servers"), dict) else {}
    conns: dict[str, dict[str, Any]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for name, spec in servers.items():
        if not isinstance(spec, dict) or not spec.get("enabled"):
            continue
        transport = str(spec.get("transport") or "streamable_http").strip()
        url = _resolve_url(spec) or str(spec.get("url") or "").strip()
        command = str(spec.get("command") or "").strip()
        if transport in {"stdio"} and command:
            conns[name] = _stdio_conn(command, list(spec.get("args") or []))
        elif url:
            conn: dict[str, Any] = {"transport": transport, "url": url}
            headers = _auth_headers(spec)
            if headers:
                conn["headers"] = headers
            conns[name] = conn
        else:
            continue
        meta[name] = {
            "majors": [str(item) for item in (spec.get("majors") or [])],
            "allow_tools": [str(item) for item in (spec.get("allow_tools") or [])],
        }
    return conns, meta


def mcp_server_names() -> frozenset[str]:
    conns, _ = _connections()
    return frozenset(conns.keys())


def is_mcp_tool_name(name: str) -> bool:
    raw = (name or "").strip()
    if not raw:
        return False
    lowered = raw.lower()
    if lowered.startswith("mcp:"):
        return True
    for server in mcp_server_names():
        prefix = f"{server}_".lower()
        if lowered.startswith(prefix):
            return True
    return False


def mcp_server_for_tool(name: str) -> str:
    raw = (name or "").strip()
    lowered = raw.lower()
    for server in sorted(mcp_server_names(), key=len, reverse=True):
        prefix = f"{server}_".lower()
        if lowered.startswith(prefix):
            return server
    return ""


def split_bound_tools(names: list[str]) -> tuple[list[str], list[str]]:
    local: list[str] = []
    mcp: list[str] = []
    for item in names:
        n = str(item or "").strip()
        if not n:
            continue
        if is_mcp_tool_name(n):
            mcp.append(n)
        else:
            local.append(n)
    return local, mcp


def tag_tool_trace_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    name = str(out.get("name") or "")
    if is_mcp_tool_name(name):
        out["source"] = "mcp"
        server = mcp_server_for_tool(name)
        if server:
            out["mcp_server"] = server
    else:
        out["source"] = "local"
    return out


async def _load_all() -> dict[str, list[BaseTool]]:
    global _cache
    if _cache is not None:
        return _cache
    conns, meta = _connections()
    grouped: dict[str, list[BaseTool]] = {name: [] for name in conns}
    if not conns:
        _cache = grouped
        return grouped
    from langchain_mcp_adapters.client import MultiServerMCPClient

    for name, conn in conns.items():
        try:
            client = MultiServerMCPClient({name: conn}, tool_name_prefix=True)
            tools = await client.get_tools()
        except Exception:
            logger.warning("MCP 服务 %s 加载失败，跳过", name, exc_info=True)
            grouped[name] = []
            continue
        allow = list((meta.get(name) or {}).get("allow_tools") or [])
        kept: list[BaseTool] = []
        for tool in tools:
            raw = str(getattr(tool, "name", "") or "")
            short = raw.split("_", 1)[1] if "_" in raw else raw
            if allow and short not in allow and raw not in allow:
                continue
            kept.append(tool)
        grouped[name] = kept
    _cache = grouped
    return grouped


async def mcp_tools_for_major(major: str) -> list[BaseTool]:
    """按大类取出允许的 MCP 工具；服务未开或失败则空列表。"""
    from tools.registry import ensure_companion_tools_registered

    ensure_companion_tools_registered()
    conns, yaml_meta = _connections()
    if not conns:
        return []
    grouped = await _load_all()
    wanted = (major or "").strip()
    catalog = ensure_companion_tools_registered().mcp_catalog_for(wanted)
    allow_by_server = {
        entry.meta.mcp_server: set(entry.meta.allow_tools) for entry in catalog if entry.meta.mcp_server
    }
    picked: list[BaseTool] = []
    seen: set[str] = set()
    for server, tools in grouped.items():
        yaml_majors = list((yaml_meta.get(server) or {}).get("majors") or [])
        if server not in allow_by_server and yaml_majors and wanted not in yaml_majors:
            continue
        if catalog and server not in allow_by_server:
            continue
        allow = allow_by_server.get(server) or set((yaml_meta.get(server) or {}).get("allow_tools") or [])
        for tool in tools:
            name = str(getattr(tool, "name", "") or "")
            short = name.split("_", 1)[1] if "_" in name else name
            if allow and short not in allow and name not in allow:
                continue
            if not name or name in seen:
                continue
            seen.add(name)
            picked.append(tool)
    return picked
