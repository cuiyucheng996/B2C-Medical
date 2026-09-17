"""C 端工具注册表：name → CompanionToolEntry（对齐主项目 ToolRegistry 单例）。"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from tools.tool_entry import CompanionToolEntry, CompanionToolMeta

__all__ = [
    "CompanionToolRegistry",
    "ensure_companion_tools_registered",
    "get_companion_registry",
    "list_companion_registry",
    "reset_companion_registry",
    "tools_from_registry",
]

_global: CompanionToolRegistry | None = None


class CompanionToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, CompanionToolEntry] = {}

    def register(self, entry: CompanionToolEntry) -> None:
        if entry.name in self._tools:
            raise ValueError(f"工具已注册：{entry.name}")
        self._tools[entry.name] = entry

    def register_function(self, tool: BaseTool, meta: CompanionToolMeta) -> None:
        name = str(getattr(tool, "name", "") or "").strip()
        if not name:
            raise ValueError("LangChain 工具缺少 name")
        self.register(CompanionToolEntry(name=name, meta=meta, langchain_tool=tool))

    def register_mcp(self, server: str, meta: CompanionToolMeta) -> None:
        name = f"mcp:{server}"
        self.register(
            CompanionToolEntry(
                name=name,
                meta=CompanionToolMeta(
                    title=meta.title,
                    summary=meta.summary,
                    category=meta.category or "mcp",
                    majors=meta.majors,
                    handlers=meta.handlers,
                    source="mcp",
                    mcp_server=server,
                    allow_tools=meta.allow_tools,
                    operation_hint=meta.operation_hint,
                ),
                langchain_tool=None,
            )
        )

    def get(self, name: str) -> CompanionToolEntry | None:
        return self._tools.get(name)

    def tools_for(self, *, major: str, handler: str) -> list[BaseTool]:
        from tools.setup import INTENT_BIND

        key = ((major or "").strip(), (handler or "").strip())
        names = INTENT_BIND.get(key) or INTENT_BIND.get((key[0], ""))
        if not names:
            return []
        bound: list[BaseTool] = []
        for name in names:
            entry = self._tools.get(name)
            if entry is None or entry.langchain_tool is None:
                continue
            bound.append(entry.langchain_tool)
        return bound

    def mcp_catalog_for(self, major: str) -> list[CompanionToolEntry]:
        wanted = (major or "").strip()
        return [
            entry
            for entry in self._tools.values()
            if entry.meta.source == "mcp" and (not entry.meta.majors or wanted in entry.meta.majors)
        ]

    def summaries(self) -> list[dict]:
        return [entry.to_summary_dict() for entry in self._tools.values()]


def get_companion_registry() -> CompanionToolRegistry:
    global _global
    if _global is None:
        _global = CompanionToolRegistry()
    return _global


def reset_companion_registry() -> None:
    global _global
    _global = None


def ensure_companion_tools_registered() -> CompanionToolRegistry:
    registry = get_companion_registry()
    if registry.summaries():
        return registry
    from tools.setup import register_all

    register_all(registry)
    return registry


def tools_from_registry(*, major: str, handler: str) -> list[BaseTool]:
    return ensure_companion_tools_registered().tools_for(major=major, handler=handler)


def list_companion_registry() -> list[dict]:
    return ensure_companion_tools_registered().summaries()
