"""C 端工具注册项（对齐主项目 ToolEntry：name + 元数据 + LangChain 工具）。"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.tools import BaseTool

__all__ = [
    "CompanionToolEntry",
    "CompanionToolMeta",
    "mark_companion_tool",
    "items_for_domain",
]

_DOMAIN_MAJORS: dict[str, tuple[str, ...]] = {
    "safety": ("风险拦截",),
    "knowledge": ("公共知识问答",),
    "memory": ("个人记忆问答",),
    "file": ("近况陈述",),
    "tongue": ("图片识别解析",),
    "report": ("报告指标解读",),
    "task": ("具体任务",),
}

_MARKED: list[tuple[BaseTool, str, CompanionToolMeta]] = []


def mark_companion_tool(
    tool: BaseTool,
    domain: str,
    *,
    title: str = "",
    summary: str = "",
    majors: tuple[str, ...] | None = None,
    handlers: tuple[str, ...] = (),
    operation_hint: str = "",
) -> BaseTool:
    """实现写完后挂域；注册表元数据缺省用 docstring。"""
    desc = str(getattr(tool, "description", "") or "").strip().split("\n")[0]
    meta = CompanionToolMeta(
        title=title or desc[:24] or str(tool.name),
        summary=summary or desc,
        category=domain,
        majors=majors if majors is not None else _DOMAIN_MAJORS.get(domain, ()),
        handlers=handlers,
        operation_hint=operation_hint,
    )
    _MARKED.append((tool, domain, meta))
    return tool


def items_for_domain(domain: str) -> tuple[tuple[BaseTool, CompanionToolMeta], ...]:
    return tuple((tool, meta) for tool, name, meta in _MARKED if name == domain)


@dataclass(frozen=True)
class CompanionToolMeta:
    title: str
    summary: str
    category: str = "lookup"
    majors: tuple[str, ...] = ()
    handlers: tuple[str, ...] = ()
    source: str = "local"
    mcp_server: str = ""
    allow_tools: tuple[str, ...] = ()
    operation_hint: str = ""


@dataclass
class CompanionToolEntry:
    name: str
    meta: CompanionToolMeta
    langchain_tool: BaseTool | None = field(default=None)

    def matches(self, *, major: str, handler: str) -> bool:
        majors = self.meta.majors
        if majors and major not in majors:
            return False
        handlers = self.meta.handlers
        if handlers and handler not in handlers:
            return False
        return True

    def to_summary_dict(self) -> dict[str, str | list[str]]:
        return {
            "name": self.name,
            "title": self.meta.title,
            "summary": self.meta.summary,
            "category": self.meta.category,
            "source": self.meta.source,
            "majors": list(self.meta.majors),
            "handlers": list(self.meta.handlers),
            "mcp_server": self.meta.mcp_server,
            "allow_tools": list(self.meta.allow_tools),
            "operation_hint": self.meta.operation_hint,
        }
