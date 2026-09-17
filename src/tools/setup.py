"""开业注册：场景 Toolkit 进注册表，INTENT_BIND 再按细意图抽子集。"""

from __future__ import annotations

from tools.registry import CompanionToolRegistry
from tools.tool_entry import CompanionToolMeta
from tools.toolkits import (
    ALL_TOOLKITS,
    FILE_TOOLKIT,
    KNOWLEDGE_TOOLKIT,
    MEMORY_TOOLKIT,
    REPORT_TOOLKIT,
    SAFETY_TOOLKIT,
    TASK_TOOLKIT,
)

__all__ = ["INTENT_BIND", "register_all"]

_GRAPH = KNOWLEDGE_TOOLKIT.pick("query_consult_graph")
_HANDBOOK = KNOWLEDGE_TOOLKIT.pick("search_exam_handbook")
_HANDBOOK_THEN_MILVUS = KNOWLEDGE_TOOLKIT.pick("search_exam_handbook", "search_milvus_knowledge")

# (major, handler) → 本地工具名；handler 空表示该大类默认
INTENT_BIND: dict[tuple[str, str], tuple[str, ...]] = {
    ("风险拦截", ""): SAFETY_TOOLKIT.names(),
    ("公共知识问答", "kg_part_diseases"): _GRAPH,
    ("公共知识问答", "kg_disease_symptoms"): _GRAPH,
    ("公共知识问答", "kg_symptom_parts"): _GRAPH,
    ("公共知识问答", "kg_disease_plain"): _GRAPH,
    ("公共知识问答", "kg_syndrome_signs"): _GRAPH,
    ("公共知识问答", "kg_exam_meaning"): _HANDBOOK_THEN_MILVUS,
    ("公共知识问答", "kg_lab_range"): _HANDBOOK_THEN_MILVUS,
    ("公共知识问答", "kg_symptom_care"): _HANDBOOK,
    ("公共知识问答", ""): KNOWLEDGE_TOOLKIT.names(),
    ("个人记忆问答", "mem_stated_consult"): MEMORY_TOOLKIT.pick("lookup_stated_consult"),
    ("个人记忆问答", "mem_companion_fact"): MEMORY_TOOLKIT.pick("lookup_life_facts"),
    ("个人记忆问答", ""): MEMORY_TOOLKIT.pick("lookup_stated_consult", "lookup_life_facts"),
    ("近况陈述", "file_image"): FILE_TOOLKIT.pick("describe_uploaded_image"),
    ("近况陈述", "file_document"): FILE_TOOLKIT.pick("describe_uploaded_document"),
    ("近况陈述", ""): MEMORY_TOOLKIT.pick("search_long_memory", "lookup_life_facts"),
    ("报告指标解读", "report_upload"): REPORT_TOOLKIT.pick(
        "explain_uploaded_report",
        "extract_report_tables",
    ),
    ("报告指标解读", "report_tables"): REPORT_TOOLKIT.pick("extract_report_tables"),
    ("报告指标解读", "report_explain"): REPORT_TOOLKIT.pick("explain_mentioned_report"),
    ("报告指标解读", "report_compare"): REPORT_TOOLKIT.pick("compare_lab_history"),
    ("报告指标解读", ""): REPORT_TOOLKIT.names(),
    ("具体任务", "task_remind"): TASK_TOOLKIT.pick("note_task_remind"),
    ("具体任务", "task_checkin"): TASK_TOOLKIT.pick("note_task_checkin"),
    ("具体任务", "task_medication"): TASK_TOOLKIT.pick("note_task_medication"),
    ("具体任务", "task_appointment"): TASK_TOOLKIT.pick("note_task_appointment"),
    ("具体任务", "task_report"): TASK_TOOLKIT.pick("note_task_report"),
    ("具体任务", ""): TASK_TOOLKIT.names(),
}


def register_all(registry: CompanionToolRegistry) -> None:
    for toolkit in ALL_TOOLKITS:
        toolkit.register(registry)
    registry.register_mcp(
        "time",
        CompanionToolMeta(
            title="MCP 时间",
            summary="第三方 Time MCP：当前时区与日期换算。",
            category="mcp",
            majors=("具体任务",),
            mcp_server="time",
            allow_tools=("get_current_time", "convert_time"),
            operation_hint="复查提醒需要落到具体日期时用。",
        ),
    )
    registry.register_mcp(
        "fetch",
        CompanionToolMeta(
            title="MCP 网页抓取",
            summary="第三方 Fetch MCP：拉取公开页面文本。",
            category="mcp",
            majors=("公共知识问答",),
            mcp_server="fetch",
            allow_tools=("fetch",),
            operation_hint="本轮已绑工具都查空后再抓网页，不当诊断。",
        ),
    )
    registry.register_mcp(
        "graphiti",
        CompanionToolMeta(
            title="MCP Graphiti 记忆",
            summary="Graphiti MCP：search_memory_facts / search_nodes。",
            category="mcp",
            majors=("个人记忆问答", "近况陈述", "报告指标解读"),
            mcp_server="graphiti",
            allow_tools=("search_memory_facts", "search_nodes"),
            operation_hint="补长期事实；禁止 add_memory。",
        ),
    )
