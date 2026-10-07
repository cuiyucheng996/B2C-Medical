"""Agent 工具调用轨迹（对应书中 assistant.tool_calls + tool 消息）。"""

from __future__ import annotations

from typing_extensions import TypedDict

__all__ = ["ToolTraceEntry"]


class ToolTraceEntry(TypedDict, total=False):
    """单次工具调用摘要，写入 RouterState.tool_trace。"""

    tool_call_id: str
    tool_name: str
    arguments: str
    content: str
    summary: str
    source: str
