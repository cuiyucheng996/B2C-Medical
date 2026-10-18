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


def _expand_env_token(raw: Any) -> str:
    val = str(raw or "").strip()
    if val.startswith("${") and val.endswith("}"):
        return (os.getenv(val[2:-1].strip()) or "").strip()
    if val.startswith("$") and len(val) > 1 and not val.startswith("${"):
        return (os.getenv(val[1:].strip()) or "").strip()
    return val


def _yingdao_rpa_ready(env: dict[str, str]) -> bool:
    model = (env.get("RPA_MODEL") or os.getenv("RPA_MODEL") or "openApi").strip().lower()
    if model == "local":
        bot = (env.get("SHADOWBOT_PATH") or "").strip()
        folder = (env.get("USER_FOLDER") or "").strip()
        return bool(bot and folder)
    key_id = (env.get("ACCESS_KEY_ID") or "").strip()
    secret = (env.get("ACCESS_KEY_SECRET") or "").strip()
    return bool(key_id and secret)


def _yingdao_stdio_env(spec: dict[str, Any]) -> dict[str, str]:
    base = _stdio_env(spec.get("env") if isinstance(spec.get("env"), dict) else None)
    model = (base.get("RPA_MODEL") or os.getenv("RPA_MODEL") or "openApi").strip()
    base["RPA_MODEL"] = model
    if model.lower() == "local":
        base["SHADOWBOT_PATH"] = (
            base.get("SHADOWBOT_PATH")
            or (os.getenv("YINGDAO_SHADOWBOT_PATH") or os.getenv("SHADOWBOT_PATH") or "")
        ).strip()
        base["USER_FOLDER"] = (
            base.get("USER_FOLDER") or (os.getenv("YINGDAO_USER_FOLDER") or os.getenv("USER_FOLDER") or "")
        ).strip()
    else:
        base["ACCESS_KEY_ID"] = (
            base.get("ACCESS_KEY_ID") or (os.getenv("YINGDAO_ACCESS_KEY_ID") or os.getenv("ACCESS_KEY_ID") or "")
        ).strip()
        base["ACCESS_KEY_SECRET"] = (
            base.get("ACCESS_KEY_SECRET")
            or (os.getenv("YINGDAO_ACCESS_KEY_SECRET") or os.getenv("ACCESS_KEY_SECRET") or "")
        ).strip()
    return base


def _default_yingdao_mcp_script() -> str:
    candidates = [
        (os.getenv("YINGDAO_MCP_SCRIPT") or "").strip(),
        str(Path(__file__).resolve().parents[2] / "server" / "mcp_server" / "yingdao" / "dist" / "index.js"),
    ]
    for item in candidates:
        if item and Path(item).is_file():
            return item
    return (os.getenv("YINGDAO_MCP_SCRIPT") or str(Path(__file__).resolve().parents[2] / "server" / "mcp_server" / "yingdao" / "dist" / "index.js")).strip()


def _stdio_env(spec: dict[str, Any] | None) -> dict[str, str]:
    base = dict(os.environ)
    if not isinstance(spec, dict):
        return base
    for key, raw in spec.items():
        name = str(key).strip()
        if not name:
            continue
        val = str(raw or "").strip()
        if val.startswith("${") and val.endswith("}"):
            val = (os.getenv(val[2:-1].strip()) or "").strip()
        elif val.startswith("$") and len(val) > 1:
            val = (os.getenv(val[1:].strip()) or "").strip()
        if val:
            base[name] = val
    return base


def _stdio_conn(command: str, args: list[Any], *, env: dict[str, str] | None = None) -> dict[str, Any]:
    argv = [str(item) for item in args]
    if command.lower() in {"npx", "npx.cmd"}:
        pkg = next((item for item in argv if item.startswith("@")), "")
        module = _NPX_TO_MODULE.get(pkg)
        if module and not _npx_bin():
            logger.info("未找到 npx，改用 %s 启动 %s", sys.executable, module)
            conn = _python_stdio(module)
            if env:
                conn["env"] = {**conn.get("env", {}), **env}
            return conn
        npx = _npx_bin() or command
        if sys.platform == "win32":
            comspec = os.environ.get("COMSPEC") or "cmd.exe"
            out = {"transport": "stdio", "command": comspec, "args": ["/c", npx, *argv]}
        else:
            out = {"transport": "stdio", "command": npx, "args": argv}
        if env:
            out["env"] = env
        return out
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
        if name == "brave_search" and not (os.getenv("BRAVE_API_KEY") or "").strip():
            logger.info("MCP brave_search 未配置 BRAVE_API_KEY，跳过")
            continue
        if name == "yingdao_rpa":
            child_env = _yingdao_stdio_env(spec)
            if not _yingdao_rpa_ready(child_env):
                logger.info("MCP yingdao_rpa 未配置 RPA_MODEL/密钥或影刀路径，跳过")
                continue
            script = _default_yingdao_mcp_script()
            if not Path(script).is_file():
                logger.info("MCP yingdao_rpa 入口不存在（请先 npm run build）：%s", script)
                continue
            node = (os.getenv("YINGDAO_NODE") or "").strip() or shutil.which("node") or shutil.which("node.exe")
            if not node:
                logger.info("MCP yingdao_rpa 未找到 node（可设 YINGDAO_NODE）")
                continue
            conns[name] = {
                "transport": "stdio",
                "command": node,
                "args": [script],
                "env": child_env,
            }
            meta[name] = {
                "majors": [str(item) for item in (spec.get("majors") or [])],
                "allow_tools": [str(item) for item in (spec.get("allow_tools") or [])],
                "agent_expose": bool(spec.get("agent_expose", True)),
            }
            continue
        transport = str(spec.get("transport") or "streamable_http").strip()
        url = _resolve_url(spec) or str(spec.get("url") or "").strip()
        command = str(spec.get("command") or "").strip()
        if transport in {"stdio"} and command:
            child_env = _stdio_env(spec.get("env") if isinstance(spec.get("env"), dict) else None)
            argv = [_expand_env_token(item) for item in (spec.get("args") or [])]
            conns[name] = _stdio_conn(command, argv, env=child_env)
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
            "agent_expose": bool(spec.get("agent_expose", True)),
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
        except Exception as exc:
            logger.warning("MCP 服务 %s 加载失败，跳过: %s", name, exc)
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
        if not (yaml_meta.get(server) or {}).get("agent_expose", True):
            continue
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
