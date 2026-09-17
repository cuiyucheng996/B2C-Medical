"""拦截危险 MCP 写操作，不交给模型去调。"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage

from tools.tool_envelope import dump_envelope

__all__ = ["McpGuardMiddleware", "is_denied_mcp_tool"]

_DENY_PARTS = (
    "add_memory",
    "add_episode",
    "delete",
    "clear_graph",
    "drop_",
    "wipe",
    "remove_memory",
)


def is_denied_mcp_tool(name: str) -> bool:
    lowered = (name or "").lower()
    return any(part in lowered for part in _DENY_PARTS)


class McpGuardMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request: ToolCallRequest, handler: Any) -> Any:
        blocked = self._block(request)
        return blocked if blocked is not None else handler(request)

    async def awrap_tool_call(self, request: ToolCallRequest, handler: Any) -> Any:
        blocked = self._block(request)
        return blocked if blocked is not None else await handler(request)

    @staticmethod
    def _block(request: ToolCallRequest) -> ToolMessage | None:
        name = str((request.tool_call or {}).get("name") or "")
        if not is_denied_mcp_tool(name):
            return None
        call_id = str((request.tool_call or {}).get("id") or "")
        return ToolMessage(
            content=dump_envelope(
                tool=name,
                ok=False,
                error="该 MCP 工具不允许在陪聊中调用（写图/删除）",
                usable=False,
                retryable=False,
            ),
            tool_call_id=call_id,
            name=name,
        )
