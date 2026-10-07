"""8.3 注册表本体 + 8.4 全局单例。"""



from __future__ import annotations

from typing import Any



from langchain_core.tools import BaseTool



from ...schemas.agent_protocol import EmitFn, SubAgent, SubAgentResult

from .tool_entry import ToolEntry, ToolExecutor, ToolMeta

from .tool_spec import ToolSpec



CATEGORY_LABELS: dict[str, str] = {

    "rag": "语义检索",

    "lookup": "结构化查询",

    "subagent": "子 Agent",

}



_SUB_AGENT_VISIBLE: dict[str, str] = {

    "disease_retrieval": "subagent:disease_retrieval",

    "symptom_match": "subagent:symptom_match",

}



__all__ = [

    "CATEGORY_LABELS",

    "ToolRegistry",

    "ensure_tools_registered",

    "format_capabilities_for_prompt",

    "get_openai_tools_for_sub_agent",

    "get_registry",
    "get_tool_by_name",
    "get_tool_detail_payload",
    "get_tool_registry",

    "get_tool_spec",

    "get_tool_specs_for_sub_agent",

    "get_tool_specs_for_visible",

    "get_tools_for_phase",

    "get_tools_for_sub_agent",

    "list_registry_entries",

    "list_registry_payload",

    "list_registry_summaries",

    "list_tool_specs",

    "reset_registry",

    "reset_tool_registry",

]





class ToolRegistry:

    """工具注册表：name → ToolEntry。"""



    def __init__(self) -> None:

        self._tools: dict[str, ToolEntry] = {}



    def _put(self, entry: ToolEntry) -> None:

        if entry.name in self._tools:

            raise ValueError(f"工具已注册：{entry.name}")

        self._tools[entry.name] = entry



    def register(self, entry: ToolEntry) -> None:

        """底层注册：直接挂载完整 ToolEntry。"""

        self._put(entry)



    def register_subagent(

        self,

        subagent: SubAgent,

        spec: ToolSpec,

        *,

        meta: ToolMeta | None = None,

    ) -> None:

        """注册子 Agent 型工具。"""

        self._put(ToolEntry(spec=spec, subagent=subagent, meta=meta))



    def register_function(

        self,

        spec: ToolSpec,

        executor: ToolExecutor,

        *,

        meta: ToolMeta | None = None,

        langchain_tool: BaseTool | None = None,

    ) -> None:

        """注册普通函数型工具。"""

        self._put(

            ToolEntry(

                spec=spec,

                executor=executor,

                meta=meta,

                langchain_tool=langchain_tool,

            )

        )



    def unregister(self, name: str) -> bool:

        if name not in self._tools:

            return False

        del self._tools[name]

        return True



    def get(self, name: str) -> ToolEntry | None:

        return self._tools.get(name)



    def has(self, name: str) -> bool:

        return name in self._tools



    def list_names(self) -> list[str]:

        return list(self._tools.keys())



    def all(self) -> list[ToolEntry]:

        return list(self._tools.values())



    def schemas(self, *, names: list[str] | None = None) -> list[dict[str, Any]]:

        """生成 OpenAI tools schema 列表。"""

        entries = self.all()

        if names is not None:

            wanted = set(names)

            entries = [entry for entry in entries if entry.name in wanted]

        return [entry.spec.to_openai_schema() for entry in entries]



    def execute(self, name: str, args: dict[str, Any], emit: EmitFn | None = None) -> SubAgentResult:
        """按名同步调度（LangChain / 无事件循环）。"""
        entry = self.get(name)
        if entry is None:
            return SubAgentResult.failed(f"未知工具: {name}")
        return entry.execute_sync(args, emit)

    async def aexecute(self, name: str, args: dict[str, Any], emit: EmitFn | None = None) -> SubAgentResult:
        """按名异步调度（模板签名）。"""
        entry = self.get(name)
        if entry is None:
            return SubAgentResult.failed(f"未知工具: {name}")
        return await entry.execute(args, emit)



    # --- 问诊业务查询（模板之外，本项目保留）---



    def by_visible(self, visible_to: str) -> list[ToolEntry]:

        return [entry for entry in self._tools.values() if entry.spec.visible_to == visible_to]



    def by_sub_agent(self, sub_agent_id: str) -> list[ToolEntry]:

        visible = _SUB_AGENT_VISIBLE.get(sub_agent_id, "")

        return self.by_visible(visible)



    def by_phase(self, major_round: int) -> list[ToolEntry]:

        return [

            entry

            for entry in self._tools.values()

            if entry.meta is not None and major_round in entry.meta.phases

        ]



    def langchain_tools_for_sub_agent(self, sub_agent_id: str) -> list[BaseTool]:

        return [

            entry.to_langchain_tool()

            for entry in self.by_sub_agent(sub_agent_id)

            if entry.executor is not None

        ]



    def langchain_tools_all(self) -> list[BaseTool]:

        """主模型可见的全部工具（含子 Agent 入口）。"""

        tools: list[BaseTool] = []

        for entry in self.all():

            try:

                tools.append(entry.to_langchain_tool())

            except ValueError:

                continue

        return tools



    def openai_schemas_for_sub_agent(self, sub_agent_id: str) -> list[dict[str, Any]]:

        names = [entry.name for entry in self.by_sub_agent(sub_agent_id) if entry.executor is not None]

        return self.schemas(names=names)



    def summaries(self) -> list[dict[str, Any]]:

        return [entry.to_summary_dict(category_labels=CATEGORY_LABELS) for entry in self.all()]



    def payloads(self) -> list[dict[str, Any]]:

        return [entry.to_detail_dict(category_labels=CATEGORY_LABELS) for entry in self.all()]





_global_registry: ToolRegistry | None = None
_initialized = False


def get_registry() -> ToolRegistry:
    """8.4 获取全局 ToolRegistry 单例。"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def reset_registry() -> None:
    """重置全局注册表（测试用）。"""
    global _global_registry, _initialized
    _global_registry = None
    _initialized = False


def get_tool_registry() -> ToolRegistry:
    """兼容别名 → get_registry。"""
    return get_registry()


def ensure_tools_registered() -> ToolRegistry:
    """8.5 开业前挂载：幂等注册全部 ToolEntry。"""
    global _initialized
    registry = get_registry()
    if not _initialized:
        from .setup import register_all_agents

        register_all_agents(registry)
        _initialized = True
    return registry


def reset_tool_registry() -> None:
    """兼容别名 → reset_registry。"""
    reset_registry()





# ---------------------------------------------------------------------------

# 兼容门面

# ---------------------------------------------------------------------------





def list_registry_entries() -> list[ToolEntry]:

    return ensure_tools_registered().all()





def list_tool_specs() -> list[ToolSpec]:

    return [entry.spec for entry in ensure_tools_registered().all()]





def get_tool_spec(name: str) -> ToolSpec | None:

    entry = ensure_tools_registered().get(name)

    return entry.spec if entry else None





def get_tool_specs_for_visible(visible_to: str) -> list[ToolSpec]:

    return [entry.spec for entry in ensure_tools_registered().by_visible(visible_to)]





def get_tool_specs_for_sub_agent(sub_agent_id: str) -> list[ToolSpec]:

    return [entry.spec for entry in ensure_tools_registered().by_sub_agent(sub_agent_id)]





def get_tools_for_sub_agent(sub_agent_id: str) -> list[BaseTool]:

    return ensure_tools_registered().langchain_tools_for_sub_agent(sub_agent_id)





def get_openai_tools_for_sub_agent(sub_agent_id: str) -> list[dict[str, Any]]:

    return ensure_tools_registered().openai_schemas_for_sub_agent(sub_agent_id)





def list_registry_summaries() -> list[dict[str, Any]]:

    return ensure_tools_registered().summaries()





def list_registry_payload() -> list[dict[str, Any]]:

    return ensure_tools_registered().payloads()





def get_tool_by_name(name: str) -> ToolEntry | None:

    return ensure_tools_registered().get(name)





def get_tool_detail_payload(name: str) -> dict[str, Any] | None:

    entry = get_tool_by_name(name)

    if entry is None:

        return None

    return entry.to_detail_dict(category_labels=CATEGORY_LABELS)





def get_tools_for_phase(major_round: int) -> list[BaseTool]:

    phase = major_round if major_round in (1, 2, 3, 4) else 1

    return [entry.to_langchain_tool() for entry in ensure_tools_registered().by_phase(phase)]





def format_capabilities_for_prompt(major_round: int) -> str:

    lines: list[str] = []

    for entry in ensure_tools_registered().by_phase(major_round):

        meta = entry.meta

        if meta is None:

            continue

        lines.append(

            f"- **{entry.name}**（{meta.title}）：{meta.summary}\n"

            f"  何时用：{meta.operation_hint}\n"

            f"  可见范围：{entry.spec.visible_to}"

        )

    return "\n".join(lines) if lines else "（当前阶段无可用工具）"


