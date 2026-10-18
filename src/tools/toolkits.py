"""按域打包：工具实现与域标识在 companion_tools，这里只组 Toolkit。"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.tools import BaseTool

from tools.companion_tools import mark_companion_tools
from tools.registry import CompanionToolRegistry
from tools.tool_entry import CompanionToolMeta, items_for_domain

__all__ = [
    "CompanionToolkit",
    "ALL_TOOLKITS",
    "FILE_TOOLKIT",
    "KNOWLEDGE_TOOLKIT",
    "MEMORY_TOOLKIT",
    "REPORT_TOOLKIT",
    "SAFETY_TOOLKIT",
    "TASK_TOOLKIT",
    "TONGUE_TOOLKIT",
]


@dataclass(frozen=True)
class CompanionToolkit:
    """对齐 LangChain Toolkit：get_tools() 给整包；pick() 给意图绑定。"""

    name: str
    items: tuple[tuple[BaseTool, CompanionToolMeta], ...]

    def get_tools(self) -> list[BaseTool]:
        return [tool for tool, _ in self.items]

    def names(self) -> tuple[str, ...]:
        return tuple(str(tool.name) for tool, _ in self.items)

    def pick(self, *tool_names: str) -> tuple[str, ...]:
        allowed = set(self.names())
        missing = [name for name in tool_names if name not in allowed]
        if missing:
            raise ValueError(f"工具包 {self.name} 不含：{missing}")
        return tool_names

    def register(self, registry: CompanionToolRegistry) -> None:
        for tool, meta in self.items:
            registry.register_function(tool, meta)


mark_companion_tools()

SAFETY_TOOLKIT = CompanionToolkit(name="safety", items=items_for_domain("safety"))
KNOWLEDGE_TOOLKIT = CompanionToolkit(name="knowledge", items=items_for_domain("knowledge"))
MEMORY_TOOLKIT = CompanionToolkit(name="memory", items=items_for_domain("memory"))
FILE_TOOLKIT = CompanionToolkit(name="file", items=items_for_domain("file"))
TONGUE_TOOLKIT = CompanionToolkit(name="tongue", items=items_for_domain("tongue"))
REPORT_TOOLKIT = CompanionToolkit(name="report", items=items_for_domain("report"))
TASK_TOOLKIT = CompanionToolkit(name="task", items=items_for_domain("task"))

ALL_TOOLKITS: tuple[CompanionToolkit, ...] = (
    SAFETY_TOOLKIT,
    KNOWLEDGE_TOOLKIT,
    MEMORY_TOOLKIT,
    FILE_TOOLKIT,
    TONGUE_TOOLKIT,
    REPORT_TOOLKIT,
    TASK_TOOLKIT,
)
