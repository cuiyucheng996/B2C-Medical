"""MCP 工具登记：仅从 config.yaml mcp.servers 读取，避免与 register_mcp 双写。"""

from __future__ import annotations

from typing import Any

from tools.tool_entry import CompanionToolMeta

__all__ = ["mcp_server_specs_from_yaml", "register_mcp_from_yaml", "validate_mcp_yaml_registry"]


def _mcp_block() -> dict[str, Any]:
    from configs.paths_b2c import ensure_uvgraph_root

    ensure_uvgraph_root()
    from src.configs import get_yaml_config

    block = get_yaml_config().get("mcp")
    return dict(block) if isinstance(block, dict) else {}


def mcp_server_specs_from_yaml() -> dict[str, dict[str, Any]]:
    """enabled 的 server → 原始 spec（含 majors / allow_tools / 展示字段）。"""
    cfg = _mcp_block()
    servers = cfg.get("servers") if isinstance(cfg.get("servers"), dict) else {}
    out: dict[str, dict[str, Any]] = {}
    for name, spec in servers.items():
        if not isinstance(spec, dict):
            continue
        if not spec.get("enabled"):
            continue
        out[str(name)] = spec
    return out


def register_mcp_from_yaml(registry: Any) -> None:
    for server, spec in mcp_server_specs_from_yaml().items():
        majors = tuple(str(m) for m in (spec.get("majors") or []))
        allow_tools = tuple(str(t) for t in (spec.get("allow_tools") or []))
        title = str(spec.get("title") or server).strip() or server
        summary = str(spec.get("summary") or "").strip() or f"MCP 服务 {server}"
        hint = str(spec.get("operation_hint") or "").strip()
        registry.register_mcp(
            server,
            CompanionToolMeta(
                title=title,
                summary=summary,
                category="mcp",
                majors=majors,
                mcp_server=server,
                allow_tools=allow_tools,
                operation_hint=hint,
            ),
        )


def validate_mcp_yaml_registry() -> list[str]:
    """登记与 YAML 一致性；未注册 registry 时只校验 YAML 结构。"""
    issues: list[str] = []
    specs = mcp_server_specs_from_yaml()
    for server, spec in specs.items():
        if not spec.get("majors"):
            issues.append(f"mcp.servers.{server} 缺少 majors")
        allow = spec.get("allow_tools")
        if allow is not None and not isinstance(allow, list):
            issues.append(f"mcp.servers.{server} allow_tools 必须是 list")
    from tools.registry import reset_companion_registry, ensure_companion_tools_registered

    reset_companion_registry()
    reg = ensure_companion_tools_registered()
    yaml_names = set(specs.keys())
    reg_names = {
        entry.meta.mcp_server
        for entry in reg._tools.values()
        if entry.meta.source == "mcp" and entry.meta.mcp_server
    }
    for name in yaml_names - reg_names:
        issues.append(f"YAML 已启用 MCP {name}，但注册表未登记")
    for name in reg_names - yaml_names:
        issues.append(f"注册表 MCP {name} 不在 YAML enabled servers 中")
    for server, spec in specs.items():
        entry = reg.get(f"mcp:{server}")
        if entry is None:
            continue
        yaml_majors = set(str(m) for m in (spec.get("majors") or []))
        reg_majors = set(entry.meta.majors or ())
        if yaml_majors != reg_majors:
            issues.append(f"MCP {server} majors 不一致：yaml={sorted(yaml_majors)} registry={sorted(reg_majors)}")
        yaml_allow = tuple(str(t) for t in (spec.get("allow_tools") or []))
        reg_allow = tuple(entry.meta.allow_tools or ())
        if yaml_allow != reg_allow:
            issues.append(f"MCP {server} allow_tools 不一致")
    reset_companion_registry()
    return issues
