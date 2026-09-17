"""C 端陪聊 Agent 工具：长期记忆检索。"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from langchain_core.tools import tool

from memory.graphiti_store import search_facts
from tools.tool_envelope import dump_envelope

__all__ = [
    "companion_user_id",
    "companion_last_facts",
    "companion_parsed_report",
    "search_long_memory",
]

companion_user_id: ContextVar[str] = ContextVar("companion_user_id", default="")
companion_group_id = companion_user_id
companion_last_facts: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "companion_last_facts",
    default=None,
)
companion_parsed_report: ContextVar[dict[str, Any] | None] = ContextVar(
    "companion_parsed_report",
    default=None,
)


@tool
def search_long_memory(query: str) -> str:
    """检索当前用户仍有效的长期事实（既往症状、地点、心情、事件等）。

    用户在说近况、症状、身体不适、地点或心情变化时必须先调用。
    纯寒暄（你好、谢谢、嗯）不要调用。
    query 请带上近几轮对话要点，不要只用单个语气词。
    """
    uid = companion_user_id.get()
    facts = search_facts(user_id=uid, query=(query or "").strip())
    companion_last_facts.set(list(facts))
    if not facts:
        return dump_envelope(
            tool="search_long_memory",
            summary="暂无仍有效的长期事实",
            data={"facts": []},
            usable=False,
            retryable=True,
        )
    return dump_envelope(
        tool="search_long_memory",
        summary=f"{len(facts)} 条长期事实",
        data={"facts": facts},
    )
