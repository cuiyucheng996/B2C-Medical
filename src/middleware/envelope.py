"""工具结果统一信封；不吞异常，失败交给官方 ToolRetry / ToolError。"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from tools.tool_envelope import normalize_tool_content

__all__ = ["CompanionToolEnvelopeMiddleware"]


class CompanionToolEnvelopeMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request: ToolCallRequest, handler: Any) -> Any:
        return _normalize_result(handler(request), request)

    async def awrap_tool_call(self, request: ToolCallRequest, handler: Any) -> Any:
        return _normalize_result(await handler(request), request)


def _normalize_result(result: Any, request: ToolCallRequest) -> Any:
    if isinstance(result, Command):
        return result
    name = str((request.tool_call or {}).get("name") or "")
    call_id = str((request.tool_call or {}).get("id") or "")
    if isinstance(result, ToolMessage):
        name = str(result.name or name)
        call_id = str(result.tool_call_id or call_id)
        content = result.content
        status = str(getattr(result, "status", "") or "")
    else:
        content = result
        status = ""
    from middleware.middleware_trace import append_middleware_trace

    append_middleware_trace(
        "CompanionToolEnvelopeMiddleware",
        f"归一化工具回执为 CompanionToolEnvelope：{name or '？'}",
    )
    return ToolMessage(
        content=normalize_tool_content(name, content, status=status),
        tool_call_id=call_id,
        name=name,
    )
