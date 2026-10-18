"""Supervisor 图状态：messages + 陪聊上下文 + reducer 合并 worker 结果。"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from graph.state import merge_dicts

__all__ = ["CompanionSupervisorState", "major_to_agent_name", "agent_name_to_major"]

_MAJOR_AGENT: dict[str, str] = {
    "公共知识问答": "knowledge_qa",
    "个人记忆问答": "memory_qa",
    "具体任务": "task_exec",
    "报告指标解读": "report_read",
    "图片识别解析": "image_parse",
    "风险拦截": "risk_guard",
    "近况陈述": "life_status",
    "闲聊": "chat",
    "other": "other",
}


def major_to_agent_name(major: str) -> str:
    return _MAJOR_AGENT.get(str(major or "").strip(), "other")


def agent_name_to_major(agent_name: str) -> str:
    for major, name in _MAJOR_AGENT.items():
        if name == agent_name:
            return major
    return "other"


class CompanionSupervisorState(TypedDict, total=False):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    companion_ctx: dict[str, Any]
    worker_outputs: Annotated[list[dict[str, Any]], operator.add]
    facts: Annotated[list[dict[str, Any]], operator.add]
    reply: str
    trace: Annotated[dict[str, Any], merge_dicts]
